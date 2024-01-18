import argparse
import os
import functools
import math
import numpy as np
import wandb
from datetime import datetime
from ruamel.yaml import YAML
from tqdm import trange, tqdm
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
    FullOptimStateDictConfig
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model import Transformer, TransformerBlock
from model_qq import TransformerQQ, TransformerBlock as TransformerBlockQQ
from data_loader import TokenizedDataset


def cosine_decay_schedule(step, total_steps, warmup_steps, lr_max, lr_min):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        n = total_steps - warmup_steps - 1
        step -= warmup_steps
        coeff = 0.5 * (1 + math.cos(step / n * math.pi))
        lr = lr_min + (lr_max - lr_min) * coeff
        return lr / lr_max
    
def cycle(iterable):
    # itertools.cycle does not shuffle the data after each iteration
    while True:
        for x in iterable:
            yield x


def func(global_rank, local_rank, world_size, train_cfg, model_cfg, model_cls, model_block_cls):

    if global_rank == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_cfg.k_config == 'qq' or model_cfg.v_config == 'qq':
            qq_config = f'{model_cfg.controller_alpha}_{model_cfg.controller_k}_{model_cfg.controller_temperature}_'
        else:
            qq_config = ''
        wandb.init(
            project = "pytorch-transformer",
            config = train_cfg.__dict__,
            name = f"{model_cfg.context_length}_{model_cfg.model_dim}_{model_cfg.n_blocks}_{model_cfg.n_attn_heads}_k{model_cfg.k_config}_v{model_cfg.v_config}_{qq_config}ts_{ts}"
        )
        checkpoint_dir = os.path.join('/local00/bioinf/hauzenbe/checkpoints', wandb.run.dir.split("/")[-2])
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        print(wandb.run.dir)

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[train_cfg.dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == "float16"))

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # path = '/system/user/publicdata/slimpajama_sampled/pretokenized'
    path = '/local00/bioinf/hauzenbe/slimpajama_sampled'
    filepaths_train = [os.path.join(path, 'train', f) for f in os.listdir(os.path.join(path, 'train'))]
    filepaths_val = [os.path.join(path, 'validation', f) for f in os.listdir(os.path.join(path, 'validation'))]
    ds_train = TokenizedDataset(filepaths=filepaths_train, context_length=model_cfg.context_length)
    ds_val = TokenizedDataset(filepaths=filepaths_val, context_length=model_cfg.context_length)

    dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size, pin_memory=True, pin_memory_device=device)
    dl_val = DataLoader(ds_val, batch_size=train_cfg.batch_size, pin_memory=True, pin_memory_device=device)
    dl_train = cycle(dl_train)

    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            model_block_cls,
        }
    )
    cpu_offload = CPUOffload(offload_params=True)
    mixed_precision = MixedPrecision(param_dtype=ptdtype)

    model = model_cls(model_cfg)
    model.to(device)
    model = FSDP(
        model,
        cpu_offload=cpu_offload,
        auto_wrap_policy=transformer_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # BackwardPrefetch.BACKWARD_PRE or BackwardPrefetch.BACKWARD_POST
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
        total_steps=int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps,
        warmup_steps=int(train_cfg.warmup_iters) // train_cfg.gradient_accumulation_steps,
        lr_max=train_cfg.lr,
        lr_min=train_cfg.lr_min
    )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_str = "training - step {}, loss: {:7.5f}, last_eval_loss: {:7.5f}, mem_allocated (gb): {}, mem_reserved (gb): {}"
    train_steps = int(train_cfg.train_iters) // train_cfg.gradient_accumulation_steps
    train_iterator = trange(train_steps, desc=train_str, leave=False, position=0, disable=(global_rank!=0))
    bytes_to_gb = lambda x: round(x / 1e9, ndigits=4)
    eval_loss = math.inf
    best_eval_loss = math.inf

    for step in train_iterator:

        x, y = next(dl_train)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with model.no_sync(), ctx:
            loss = 0.0
            for _ in range(train_cfg.gradient_accumulation_steps):
                out = model(x, y)
                partial_loss = out['loss']
                if 'aux_loss' in out:
                    partial_loss += out['aux_loss']
                loss += partial_loss / train_cfg.gradient_accumulation_steps
        scaler.scale(loss).backward()

        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            train_loss = loss.item() / world_size
        else:
            train_loss = loss.item()

        if train_cfg.max_grad is not None:
            scaler.unscale_(optimizer)
            model.clip_grad_norm_(train_cfg.max_grad)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        lr_schedule.step()

        if step % train_cfg.eval_interval == 0:

            model.eval()

            eval_iters = train_cfg.eval_iters if train_cfg.eval_iters is not None else len(dl_val)
            eval_str = "evaluating - step {}"
            eval_iterator = tqdm(dl_val, desc=eval_str.format(0), total=eval_iters, leave=False, position=1, disable=(global_rank!=0))

            losses = torch.zeros((eval_iters,), device=device)
            aux_losses = torch.zeros((eval_iters,), device=device)

            with torch.no_grad():
                for eval_step, (x, y) in enumerate(eval_iterator):

                    eval_iterator.set_description(
                        eval_str.format(eval_step),
                        refresh=True
                    )

                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    out = model(x, y)

                    losses[eval_step] = out['loss']
                    if 'aux_loss' in out:
                        aux_losses[eval_step] = out['aux_loss']

                    if eval_step + 1 == eval_iters:
                        break

            if world_size > 1:
                dist.all_reduce(losses, op=dist.ReduceOp.SUM)
                if 'aux_loss' in out:
                    dist.all_reduce(aux_losses, op=dist.ReduceOp.SUM)

            losses = losses / world_size
            aux_losses = aux_losses / world_size
            eval_loss = (losses + aux_losses).mean()

            model.train()

            if eval_loss < best_eval_loss:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT,
                    state_dict_config = save_policy,
                    optim_state_dict_config = optim_save_policy
                ):   
                    save_dict = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_schedule": lr_schedule.state_dict(),
                        "step": step
                    }
                    if global_rank == 0:
                        torch.save(save_dict, os.path.join(checkpoint_dir, "best_model.pt"))
                best_eval_loss = eval_loss

        log_dict = {
            "train_loss": train_loss,
            "eval_loss": losses.mean(),
            "eval_loss_aux": aux_losses.mean(),
            "memory_allocated": bytes_to_gb(torch.cuda.memory_allocated(device=device)),
            "memory_reserved": bytes_to_gb(torch.cuda.memory_reserved(device=device))
        }
        
        if global_rank == 0: wandb.log(log_dict)

        train_iterator.set_description(
            train_str.format(step, *log_dict.values()),
            refresh=True
        )


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

    print(f"starting process with global rank {global_rank}")

    main_func(global_rank, local_rank, world_size, *main_func_args)

    dist.barrier()

    dist.destroy_process_group()
    

def main():
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    print(f"CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()} - cudaNCCL {torch.cuda.nccl.version()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", help="'gloo' or 'nccl'.")
    parser.add_argument("--num-gpus", type=int, default=1, help="# GPUs per machine.")
    parser.add_argument("--num-machines", type=int, default=1, help="# of machines.")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine).")
    parser.add_argument("--master-addr", type=str, default='127.0.0.1', help="the ip address of the main machine")
    parser.add_argument("--master-port", type=int, default=1234, help="the port of the main machine")
    parser.add_argument("--model-cfg", type=str, default='model_cfg', help="name of the model config")
    args = parser.parse_args()

    with open("cfg.yml", "r") as f:
        cfg = YAML().load(f)
    model_cfg = argparse.Namespace(**cfg[args.model_cfg])
    train_cfg = argparse.Namespace(**cfg['train_cfg'])

    # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    setattr(train_cfg, 'dtype', 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16')
    setattr(model_cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)
    setattr(model_cfg, 'cfg_name', args.model_cfg)
    
    dist_url = f"tcp://{args.master_addr}:{args.master_port}"
    world_size = args.num_machines * args.num_gpus

    model_cls = TransformerQQ if 'qq' in args.model_cfg else Transformer
    model_block_cls = TransformerBlockQQ if 'qq' in args.model_cfg else TransformerBlock

    mp.spawn(
        distributed_worker,
        args = (
            func,
            args.backend,
            world_size,
            args.num_gpus,
            args.machine_rank,
            dist_url,
            (train_cfg, model_cfg, model_cls, model_block_cls)
        ),
        nprocs=args.num_gpus,
        daemon=False
    )


if __name__ == '__main__':
    main()