import argparse
import math
import torch
import numpy as np
from functools import partial
from tqdm import trange
from torch import nn
from torch.nn import functional as F
from model import Transformer
from data_loader import TokenizedDataset
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
from contextlib import nullcontext


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

# TO SET
DEVICE = 'cuda'
DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler


device_type = 'cuda' if 'cuda' in DEVICE else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DTYPE]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == 'float16'))

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

with open("cfg.yml", "r") as f:
    cfg = YAML().load(f)
model_cfg = argparse.Namespace(**cfg['model_cfg'])
train_cfg = argparse.Namespace(**cfg['train_cfg'])

setattr(model_cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)

ds_train = TokenizedDataset(filenames='train.bin', context_length=model_cfg.context_length)
ds_val = TokenizedDataset(filenames='test.bin', context_length=model_cfg.context_length)

dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size, pin_memory=(device_type=='cuda'), pin_memory_device=DEVICE)
dl_val = DataLoader(ds_val, batch_size=train_cfg.batch_size, pin_memory=(device_type=='cuda'), pin_memory_device=DEVICE)
dl_train = cycle(dl_train)

model = Transformer(model_cfg)

model = torch.compile(model)

model.to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = train_cfg.lr,
    betas = (train_cfg.adam_b1, train_cfg.adam_b2),
    eps = 1e-08,
    weight_decay = train_cfg.adam_weight_decay
)

lr_lambda = partial(
    cosine_decay_schedule,
    total_epochs=train_cfg.train_iters,
    warmup_epochs=train_cfg.warmup_iters,
    lr_max=train_cfg.lr,
    lr_min=train_cfg.lr_min
)
lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_str = "training - step {}, loss: {:7.5f}, last_eval_loss: {:7.5f}"
train_steps = int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps
train_iterator = trange(train_steps, leave=False, position=0)

for step in train_iterator:

    x, y = next(dl_train)
    x = x.to(DEVICE, non_blocking=(device_type=='cuda'))
    y = y.to(DEVICE, non_blocking=(device_type=='cuda'))

    train_loss = 0
    for sub_step in range(train_cfg.gradient_accumulation_steps):
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            loss /= train_cfg.gradient_accumulation_steps
        scaler.scale(loss).backward()
        train_loss += loss.item()

    if train_cfg.max_grad is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad)
    
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)
    
    lr_schedule.step()

    if step % train_cfg.eval_interval == 0:

        model.eval()

        for eval_step, (x, y) in enumerate(dl_val):

            x = x.to(DEVICE, non_blocking=(device_type=='cuda'))
            y = y.to(DEVICE, non_blocking=(device_type=='cuda'))

            losses = np.array([])
            with torch.no_grad():
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            np.append(losses, loss.item())

            if train_cfg.eval_iters is not None and eval_step + 1 >= train_cfg.eval_iters:
                break

        eval_loss = losses.mean()

        model.train()

    train_iterator.set_description(train_str.format(step, train_loss, eval_loss), refresh=True)





import IPython; IPython.embed(); exit(1)
