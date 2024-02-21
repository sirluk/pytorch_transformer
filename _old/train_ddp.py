import os
import argparse
import math
import numpy as np
from functools import partial
from tqdm import trange
from ruamel.yaml import YAML
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader

from model import Transformer
from data_loader import TokenizedDataset


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


def run(main_func, backend, num_machines, num_gpus, machine_rank, dist_url, args=()):
    world_size = num_machines * num_gpus

    mp.spawn(
        distributed_worker,
        nprocs=num_gpus,
        args=(
            main_func,
            backend,
            world_size,
            num_gpus,
            machine_rank,
            dist_url,
            args,
        ),
        daemon=False,
    )

# TO SET
DEVICE = 'cuda'
DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
DDP_BACKEND = 'nccl' # 'nccl', 'gloo', etc.

with open("cfg.yml", "r") as f:
    cfg = YAML().load(f)
model_cfg = argparse.Namespace(**cfg['model_cfg'])
train_cfg = argparse.Namespace(**cfg['train_cfg'])

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=DDP_BACKEND)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert train_cfg.gradient_accumulation_steps % ddp_world_size == 0
    train_cfg.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DTYPE]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(DTYPE == 'float16'))

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

setattr(model_cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)

ds_train = TokenizedDataset(filenames='data/train.bin', context_length=model_cfg.context_length)
ds_val = TokenizedDataset(filenames='data/test.bin', context_length=model_cfg.context_length)

dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size, pin_memory=(device_type=='cuda'), pin_memory_device=DEVICE)
dl_val = DataLoader(ds_val, batch_size=train_cfg.batch_size, pin_memory=(device_type=='cuda'), pin_memory_device=DEVICE)
dl_train = cycle(dl_train)

model = Transformer(model_cfg)

model = torch.compile(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
else:
    model.to(DEVICE)
    raw_model = model

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = train_cfg.lr,
    betas = (train_cfg.adam_b1, train_cfg.adam_b2),
    eps = 1e-08,
    weight_decay = train_cfg.adam_weight_decay
)

lr_lambda = partial(
    cosine_decay_schedule,
    total_epochs=int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps,
    warmup_epochs=int(train_cfg.warmup_iters) // train_cfg.gradient_accumulation_steps,
    lr_max=train_cfg.lr,
    lr_min=train_cfg.lr_min
)
lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_str = "training - step {}, loss: {:7.5f}, last_eval_loss: {:7.5f}"
train_steps = int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps
train_iterator = trange(train_steps, leave=False, position=0)

model.train()

for step in train_iterator:

    x, y = next(dl_train)
    x = x.to(DEVICE, non_blocking=(device_type=='cuda'))
    y = y.to(DEVICE, non_blocking=(device_type=='cuda'))

    train_loss = 0
    for sub_step in range(train_cfg.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (sub_step == train_cfg.gradient_accumulation_steps - 1)    
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

        losses = np.array([])

        for eval_step, (x, y) in enumerate(dl_val):

            x = x.to(DEVICE, non_blocking=(device_type=='cuda'))
            y = y.to(DEVICE, non_blocking=(device_type=='cuda'))

            with torch.no_grad():
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            losses = np.append(losses, loss.item())

            if train_cfg.eval_iters is not None and eval_step + 1 >= train_cfg.eval_iters:
                break

        eval_loss = losses.mean()

        model.train()

    train_iterator.set_description(train_str.format(step, train_loss, eval_loss), refresh=True)

if ddp:
    destroy_process_group()



import IPython; IPython.embed(); exit(1)
