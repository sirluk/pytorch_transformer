# /system/user/publicwork/hauzenbe/hf/misc/accelerate/default_config.yaml

import argparse
import os
import math
import torch
import numpy as np
import wandb
from datetime import datetime
from functools import partial
from tqdm import trange, tqdm
from torch.nn import functional as F
from data_loader import TokenizedDataset
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model import TransformerContext, TransformerBlock

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

dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size,)
dl_val = DataLoader(ds_val, batch_size=train_cfg.batch_size)
dl_train = cycle(dl_train)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
k_kv_groups = f'{model_cfg.kv_groups}' if model_cfg.k_config != 'mha' else ''
v_kv_groups = f'{model_cfg.kv_groups}' if model_cfg.v_config != 'mha' else ''
wandb.init(
    project = "transformer-experiments-context",
    config = train_cfg.__dict__,
    name = f"context_{model_cfg.context_length}_{model_cfg.model_dim}_{model_cfg.n_blocks}_{model_cfg.n_attn_heads}_k{model_cfg.k_config}{k_kv_groups}_v{model_cfg.v_config}{v_kv_groups}_ts_{ts}",
    mode = 'disabled'
)
checkpoint_dir = os.path.join('/local00/bioinf/hauzenbe/checkpoints', wandb.run.dir.split("/")[-2])
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
print(wandb.run.dir)

transformer_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        TransformerBlock,
    }
)
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
    auto_wrap_policy=transformer_wrap_policy
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin, gradient_accumulation_steps=train_cfg.gradient_accumulation_steps)

model = TransformerContext.load_hf_llama2("meta-llama/Llama-2-7b-hf")
model = accelerator.prepare(model)

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
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    optimizer, dl_train, dl_val, lr_scheduler
)

train_str = "training - step {}, loss: {:7.5f}, last_eval_loss: {:7.5f}"
train_steps = int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps
train_iterator = trange(train_steps, leave=False, position=0)
bytes_to_mb = lambda x: x / 1e6
eval_loss = math.inf
best_eval_loss = math.inf

model.train()

for step in train_iterator:

    if step >= warmup_steps:
        model.freeze_head(False)

    x, y = next(dl_train)
    y_context = calc_y_context(y, train_cfg.future_context_size, model_cfg.vocab_size)

    out = model(x, y, y_context)
    loss = out['loss'] + out['context_loss']

    accelerator.backward(loss)

    if train_cfg.max_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad)

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    if step % train_cfg.eval_interval == 0:

        eval_iters = train_cfg.eval_iters if train_cfg.eval_iters is not None else len(dl_val)
        eval_str = "evaluating - step {}"
        eval_iterator = tqdm(dl_val, desc=eval_str.format(0), total=eval_iters, leave=False, position=1)

        model.eval()

        losses = np.array([])
        context_losses = np.array([])

        for eval_step, (x, y) in enumerate(eval_iterator):

            y_context = calc_y_context(y, train_cfg.future_context_size, model_cfg.vocab_size)

            with torch.no_grad():
                out_eval = model(x, y, y_context)
            losses = np.append(losses, out_eval['loss'].item())
            context_losses = np.append(losses, out_eval['context_loss'].item())

            if eval_step + 1 == eval_iters:
                break

        eval_loss = losses.mean()
        eval_context_loss = context_losses.mean()

        if eval_loss < best_eval_loss:
            print(f"--> saving model ...")
            model_state = {
                'lm_head': accelerator.get_state_dict(model.lm_head),
                'context_head': accelerator.get_state_dict(model.context_head),
                'cfg': model.cfg
            }
            torch.save(model_state, os.path.join(checkpoint_dir, "model.pt"))
            print(f"--> saving train state ...")
            accelerator.save_state(os.path.join(checkpoint_dir, "train_state.pt"))


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

