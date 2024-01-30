import argparse
import os
import functools
import math
import numpy as np
import wandb
from ruamel.yaml import YAML
from tqdm import trange

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from model import Transformer, TransformerBlock
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


def func(global_rank, local_rank, train_cfg, model_cfg):

    # if global_rank == 0: wandb.init(project="pytorch-transformer", config = train_cfg.__dict_)

    device = f"cuda:{local_rank}"
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[train_cfg.dtype]
    #ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    #scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.dtype == 'float16'))

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    ds_train = TokenizedDataset(filenames='data/train.bin', context_length=model_cfg.context_length)
    ds_val = TokenizedDataset(filenames='data/test.bin', context_length=model_cfg.context_length)

    dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size, pin_memory=True, pin_memory_device=device)
    dl_val = DataLoader(ds_val, batch_size=train_cfg.batch_size, pin_memory=True, pin_memory_device=device)
    dl_train = cycle(dl_train)

    transformer_wrap_policy = ModuleWrapPolicy(module_classes=[TransformerBlock])
    cpu_offload = CPUOffload(offload_params=False)
    mixed_precision = MixedPrecision(param_dtype=ptdtype, reduce_dtype=torch.float32)

    model = Transformer(model_cfg)
    model.to(device)
    model = FSDP(
        model,
        cpu_offload=cpu_offload,
        auto_wrap_policy=transformer_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device()
    )
    # model = torch.compile(model)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = train_cfg.lr,
        betas = (train_cfg.adam_b1, train_cfg.adam_b2),
        eps = 1e-08,
        weight_decay = train_cfg.adam_weight_decay
    )

    lr_lambda = functools.partial(
        cosine_decay_schedule,
        total_epochs=int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps,
        warmup_epochs=int(train_cfg.warmup_iters) // train_cfg.gradient_accumulation_steps,
        lr_max=train_cfg.lr,
        lr_min=train_cfg.lr_min
    )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_str = "training - step {}, loss: {:7.5f}, last_eval_loss: {:7.5f}, mem_allocated: {}, mem_reserved: {}, mma: {}, mmr: {}"
    train_steps = int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps
    train_iterator = trange(train_steps, leave=False, position=0, disable=(global_rank==0))
    bytes_to_mb = lambda x: x / 1e6

    for step in train_iterator:

        x, y = next(dl_train)

        print(ptdtype)
        logits = model(x)
        y = y.to(device, non_blocking=True)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        loss.backward()

        if global_rank == 0:
            print('memory allocated', bytes_to_mb(torch.cuda.memory_allocated(device=device)))
            print('memory reserved', bytes_to_mb(torch.cuda.memory_reserved(device=device)))
            print('max memory allocated', bytes_to_mb(torch.cuda.max_memory_allocated(device=device)))
            print('max memory reserved', bytes_to_mb(torch.cuda.max_memory_reserved(device=device)))

            print('memory summary', torch.cuda.memory_summary(device=device))

        break


def distributed_worker(
    local_rank,
    main_func,
    ddp_backend,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    ddp_init_method,
    main_func_args
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your installation.")
    if num_gpus_per_machine > torch.cuda.device_count():
        raise RuntimeError("Parameter 'num_gpus_per_machine' exceeds torch.cuda.device_count()")

    global_rank = machine_rank * num_gpus_per_machine + local_rank
    
    dist.init_process_group(
        backend=ddp_backend,
        init_method=ddp_init_method,
        world_size=world_size,
        rank=global_rank
    )

    torch.cuda.set_device(local_rank)

    print(f"starting process with global rank {global_rank}")

    main_func(global_rank, local_rank, *main_func_args)

    dist.destroy_process_group()
    

def main():
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"

    print(f"CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()} - cudaNCCL {torch.cuda.nccl.version()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", help="'gloo' or 'nccl'.")
    parser.add_argument("--num-gpus", type=int, default=1, help="# GPUs per machine.")
    parser.add_argument("--num-machines", type=int, default=1, help="# of machines.")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine).")
    parser.add_argument("--master-addr", type=str, default='127.0.0.1', help="the ip address of the main machine")
    parser.add_argument("--master-port", type=int, default=1234, help="the port of the main machine")#")
    args = parser.parse_args()

    with open("cfg.yml", "r") as f:
        cfg = YAML().load(f)
    model_cfg = argparse.Namespace(**cfg['model_cfg'])
    train_cfg = argparse.Namespace(**cfg['train_cfg'])

    # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    setattr(train_cfg, 'dtype', 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16')
    setattr(model_cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)
    
    dist_url = f"tcp://{args.master_addr}:{args.master_port}"
    world_size = args.num_machines * args.num_gpus

    mp.spawn(
        distributed_worker,
        args = (
            func,
            args.backend,
            world_size,
            args.num_gpus,
            args.machine_rank,
            dist_url,
            (train_cfg, model_cfg)
        ),
        nprocs=args.num_gpus,
        daemon=False
    )


if __name__ == '__main__':
    main()