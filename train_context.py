import argparse
import os
import math
import torch
import numpy as np
import wandb
from datetime import datetime
from functools import partial
from tqdm import trange, tqdm
from torch import nn
from torch.nn import functional as F
from model import Transformer
from data_loader import TokenizedDataset
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
from contextlib import nullcontext
from transformers import AutoModel, BitsAndBytesConfig

from model import TransformerContext

def cosine_decay_schedule(epoch, total_epochs, warmup_epochs, lr_max, lr_min):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        n = total_epochs - warmup_epochs - 1
        epoch -= warmup_epochs
        coeff = 0.5 * (1 + math.cos(epoch / n * math.pi))
        lr = lr_min + (lr_max - lr_min) * coeff
        return lr / lr_max
    
def cycle(iterable):
    # itertools.cycle does not shuffle the data after each iteration
    while True:
        for x in iterable:
            yield x

def calc_y_context(y, future_context_size, vocab_size):
    B, extended_seq_len = y.shape
    seq_len = extended_seq_len - future_context_size + 1
    # build indices for context target
    indices = torch.arange(y.shape[1])
    indices = (indices.view(-1, 1).repeat((1, y.shape[1])) - indices.view(1, -1))
    indices = (indices[future_context_size-1:, :future_context_size] % y.shape[1]).flip(1)

    # get context targets
    y_context = torch.gather(y, dim=1, index=indices.view(1, -1).expand(B, -1)).view(B, seq_len, -1)
    return F.one_hot(y_context, num_classes=vocab_size).sum(-2)

# TO SET
DEVICE_TYPE = 'cuda'
DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler


def main():
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DTYPE]
    ctx = nullcontext() if DEVICE_TYPE == 'cpu' else torch.amp.autocast(device_type=DEVICE_TYPE, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == 'float16'))

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    with open("cfg.yml", "r") as f:
        cfg = YAML().load(f)
    model_cfg = argparse.Namespace(**cfg['model_cfg_llama7b'])
    train_cfg = argparse.Namespace(**cfg['train_cfg'])

    setattr(model_cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)

    path = '/local00/bioinf/hauzenbe/slimpajama_sampled'
    filepaths_train = [os.path.join(path, 'train', f) for f in os.listdir(os.path.join(path, 'train'))]
    filepaths_val = [os.path.join(path, 'validation', f) for f in os.listdir(os.path.join(path, 'validation'))]
    ds_train = TokenizedDataset(filepaths=filepaths_train, context_length=model_cfg.context_length, predict_n_tokens=train_cfg.future_context_size)
    ds_val = TokenizedDataset(filepaths=filepaths_val, context_length=model_cfg.context_length, predict_n_tokens=train_cfg.future_context_size)

    dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size, pin_memory=True, pin_memory_device='cuda:2')
    dl_val = DataLoader(ds_val, batch_size=train_cfg.batch_size, pin_memory=True, pin_memory_device='cuda:2')
    dl_train = cycle(dl_train)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_kv_groups = f'{model_cfg.kv_groups}' if model_cfg.k_config != 'mha' else ''
    v_kv_groups = f'{model_cfg.kv_groups}' if model_cfg.v_config != 'mha' else ''
    wandb.init(
        project = "transformer-experiments-context",
        config = train_cfg.__dict__,
        name = f"context_{model_cfg.context_length}_{model_cfg.model_dim}_{model_cfg.n_blocks}_{model_cfg.n_attn_heads}_k{model_cfg.k_config}{k_kv_groups}_v{model_cfg.v_config}{v_kv_groups}_ts_{ts}",
        # mode = 'disabled'
    )
    checkpoint_dir = os.path.join('/local00/bioinf/hauzenbe/checkpoints', wandb.run.dir.split("/")[-2])
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    print(wandb.run.dir)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit = True,
        load_in_4bit = False,
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_use_double_quant = False,
        bnb_4bit_compute_dtype = ptdtype
    )
    device_map = {
        'embed_tokens': 'cuda:0',
        'norm': 'cuda:1',
        **{f'layers.{i}': f'cuda:{i // 16}' for i in range(32)}
    }

    # quantization_config=quantization_config
    base_model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", device_map=device_map)
    setattr(base_model, 'cfg', model_cfg)
    model = TransformerContext(model_cfg, model = base_model)
    model.lm_head.to('cuda:2')
    model.context_head.to('cuda:2')
    model.freeze_base_model(True)
    model.freeze_head(True)

    param_groups = [
        {'params': model.lm_head.parameters(), 'lr': 1e-6},
        {'params': model.context_head.parameters(), 'lr': 1e-4}
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas = (train_cfg.adam_b1, train_cfg.adam_b2),
        eps = 1e-08,
        weight_decay = train_cfg.adam_weight_decay
    )

    total_steps=int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps
    warmup_steps=int(train_cfg.warmup_iters) // train_cfg.gradient_accumulation_steps
    lr_lambda = partial(
        cosine_decay_schedule,
        total_epochs=total_steps,
        warmup_epochs=warmup_steps,
        lr_max=train_cfg.lr,
        lr_min=train_cfg.lr_min
    )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_str = "training - step {}, loss: {:7.5f}, last_eval_loss: {:7.5f}"
    train_steps = int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps
    train_iterator = trange(train_steps, leave=False, position=0)

    eval_loss = math.inf
    best_eval_loss = math.inf

    model.train()

    for step in train_iterator:

        if step >= warmup_steps:
            model.freeze_head(False)

        loss = 0.0
        for _ in range(train_cfg.gradient_accumulation_steps):
            
            x, y = next(dl_train)
            y_context = calc_y_context(y, train_cfg.future_context_size, model_cfg.vocab_size)

            x = x.to('cuda:0', non_blocking=(DEVICE_TYPE=='cuda'))
            y = y.to('cuda:2', non_blocking=(DEVICE_TYPE=='cuda'))
            y_context = y_context.to('cuda:3', non_blocking=(DEVICE_TYPE=='cuda'))
            
            with ctx:
                out = model(x, y, y_context, 'cuda:3')
                loss += out['loss']
                loss += out['context_loss']
                loss /= train_cfg.gradient_accumulation_steps
        scaler.scale(loss).backward()

        if train_cfg.max_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad)
        
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)
        
        lr_schedule.step()

        if step % train_cfg.eval_interval == 0:

            eval_iters = train_cfg.eval_iters if train_cfg.eval_iters is not None else len(dl_val)
            eval_str = "evaluating - step {}"
            eval_iterator = tqdm(dl_val, desc=eval_str.format(0), total=eval_iters, leave=False, position=1)

            model.eval()

            losses = np.array([])
            context_losses = np.array([])

            for eval_step, (x, y) in enumerate(eval_iterator):

                y_context = calc_y_context(y, train_cfg.future_context_size, model_cfg.vocab_size)

                x = x.to('cuda:0', non_blocking=(DEVICE_TYPE=='cuda'))
                y = y.to('cuda:2', non_blocking=(DEVICE_TYPE=='cuda'))
                y_context = y_context.to('cuda:3', non_blocking=(DEVICE_TYPE=='cuda'))

                with torch.no_grad():
                    out_eval = model(x, y, y_context, 'cuda:3')
                losses = np.append(losses, out_eval['loss'].item())
                context_losses = np.append(losses, out_eval['context_loss'].item())

                if eval_step + 1 == eval_iters:
                    break

            eval_loss = losses.mean()
            eval_context_loss = context_losses.mean()

            if eval_loss < best_eval_loss:
                model_state = {
                    'lm_head': model.lm_head.state_dict(),
                    'context_head': model.context_head.state_dict(),
                    'cfg': model.cfg
                }
                train_state = {
                    "optimizer": optimizer.state_dict(),
                    "lr_schedule": lr_schedule.state_dict(),
                    "step": step
                }
                print(f"--> saving model ...")
                torch.save(model_state, os.path.join(checkpoint_dir, "model.pt"))
                print(f"--> saving train state ...")
                torch.save(train_state, os.path.join(checkpoint_dir, "train_state.pt"))

            log_dict = {
                "train_loss": out['loss'].item(),
                'train_context_loss': out['context_loss'].item(),
                "eval_loss": eval_loss,
                "eval_context_loss": eval_context_loss
            }
            wandb.log(log_dict)

            model.train()

        train_iterator.set_description(
            train_str.format(step, log_dict['train_loss'], log_dict['eval_loss']),
            refresh=True
        )


if __name__ == '__main__':
    main()