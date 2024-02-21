import argparse
import os
import functools
import math
import json
import numpy as np
import wandb
import pickle
import itertools
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

from transformers import AutoModel, AutoTokenizer


def func(global_rank, local_rank, world_size, train_cfg):

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[train_cfg.dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    path = '/local00/bioinf/hauzenbe/slimpajama_sampled'
    filepaths_train = [os.path.join(path, 'train', f) for f in os.listdir(os.path.join(path, 'train'))]
    ds_train = TokenizedDataset(filepaths=filepaths_train, context_length=4096)

    dl_train = DataLoader(ds_train, batch_size=train_cfg.batch_size, pin_memory=True, pin_memory_device=device)
    dl_train = itertools.cycle(dl_train)

    cpu_offload = CPUOffload(offload_params=True)
    mixed_precision = MixedPrecision(param_dtype=ptdtype)

    model = AutoModel.from_pretrained('facebook/opt-350m')
    model.to(device)

    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            model.decoder.layers[0].__class__,
        }
    )

    model = FSDP(
        model,
        cpu_offload=cpu_offload,
        auto_wrap_policy=transformer_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # BackwardPrefetch.BACKWARD_PRE or BackwardPrefetch.BACKWARD_POST
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=False
    )
    # model = torch.compile(model)
    model.eval()

    def quantize(x):
        x = x.to(torch.bfloat16)
        nd = 254 / (x.max() - x.min())
        xi8 = (nd * x).round().to(torch.int8)
        return xi8, nd.item()


    
            


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
    args = parser.parse_args()

    with open("cfg.yml", "r") as f:
        cfg = YAML().load(f)
    train_cfg = argparse.Namespace(**cfg['train_cfg'])

    # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    setattr(train_cfg, 'dtype', 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16')
    
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
            (train_cfg,)
        ),
        nprocs=args.num_gpus,
        daemon=False
    )


if __name__ == '__main__':
    main()