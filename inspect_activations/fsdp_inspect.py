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

from model import Transformer, TransformerBlock
from model_qq import TransformerQQ, TransformerBlock as TransformerBlockQQ
from data_loader import TokenizedDataset


PATH_NAME = 'activations'


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

    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            TransformerBlock,
        }
    )
    cpu_offload = CPUOffload(offload_params=True)
    mixed_precision = MixedPrecision(param_dtype=ptdtype)

    model = Transformer.load_meta_llama2("../llama/llama-2-7b")
    model.to(device)
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
    model.train()

    def quantize(x):
        x = x.to(torch.bfloat16)
        nd = 254 / (x.max() - x.min())
        xi8 = (nd * x).round().to(torch.int8)
        return xi8, nd.item()


    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    model.embedding.register_forward_hook(get_activation('embedding'))
    for i, block in enumerate(model.blocks):
        # block.attn_norm.register_forward_hook(get_activation(f'block{i}_attn_norm'))
        block.attn_layer.attn_proj_q.register_forward_hook(get_activation(f'block{i}_attn_proj_q'))
        block.attn_layer.attn_proj_k.register_forward_hook(get_activation(f'block{i}_attn_proj_k'))
        block.attn_layer.attn_proj_v.register_forward_hook(get_activation(f'block{i}_attn_proj_v'))
        block.attn_layer.register_forward_hook(get_activation(f'block{i}_attn'))
        # block.ffn_norm.register_forward_hook(get_activation(f'block{i}_ffn_norm'))
        # block.ffn_layer.ffn_linear1.register_forward_hook(get_activation(f'block{i}_ffn_linear1'))
        # block.ffn_layer.ffn_linear2.register_forward_hook(get_activation(f'block{i}_ffn_linear2'))
        # block.ffn_layer.ffn_linear3.register_forward_hook(get_activation(f'block{i}_ffn_linear3'))
        block.ffn_layer.register_forward_hook(get_activation(f'block{i}_ffn'))
    
    if not os.path.exists(PATH_NAME):
        os.makedirs(PATH_NAME)

    inference_steps = 32
    inference_iterator = trange(inference_steps, leave=False, position=0, disable=(global_rank!=0))

    with torch.no_grad():
        quantization_scalars = {}
        for step in inference_iterator:

            x, _ = next(dl_train)

            out = model(x)

            for i in range(train_cfg.batch_size):
                micro_step = step*train_cfg.batch_size+i
                if global_rank==0:
                    inputs_i8, ndi = quantize(x[i].view(-1).cpu())
                    outputs_i8, ndo = quantize(out['logits'][i].view(-1).cpu())
                    
                    np.save(os.path.join(PATH_NAME, f'inputs_sample{micro_step}.npy'), inputs_i8)
                    np.save(os.path.join(PATH_NAME, f'outputs_sample{micro_step}.npy'), outputs_i8)
                    quantization_scalars[f'inputs_sample{micro_step}.npy'] = ndi
                    quantization_scalars[f'outputs_sample{micro_step}.npy'] = ndo
                for k in list(activations.keys()):
                    act = activations[k][i]
                    if global_rank==0:
                        activations_i8, nda = quantize(act.view(-1))
                        np.save(os.path.join(PATH_NAME, f'activations_sample{micro_step}_{k}.npy'), activations_i8.numpy())
                        quantization_scalars[f'activations_sample{micro_step}_{k}.npy'] = nda
            activations = {}

        if global_rank == 0:
            with open(os.path.join(PATH_NAME, 'quantization_scalars.json'), 'w') as fp:
                json.dump(quantization_scalars, fp)
            


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