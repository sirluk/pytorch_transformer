import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta

from data_loader import TokenizedDataset


def func():
    # print(torch.randn(1))
    ds_val = TokenizedDataset(filenames='data/test.bin', context_length=1024)
    x, y = next(iter(ds_val))


def run(main_func, ddp_backend, num_machines, num_gpus, machine_rank, ddp_init_method):
    world_size = num_machines * num_gpus

    mp.spawn(
        distributed_worker,
        args=(
            main_func,
            ddp_backend,
            world_size,
            num_gpus,
            machine_rank,
            ddp_init_method
        ),
        nprocs=num_gpus,
        daemon=False,
    )


def distributed_worker(
    local_rank,
    main_func,
    ddp_backend,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    ddp_init_method
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

    dist.barrier()

    print(f"Global rank {global_rank}.")
    print("Synchronized GPUs.")

    torch.cuda.set_device(local_rank)

    main_func()

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
    parser.add_argument("--master-port", type=int, default=1234, help="the port of the main machine")
    args = parser.parse_args()

    dist_url = f"tcp://{args.master_addr}:{args.master_port}"

    run(
        main_func=func,
        ddp_backend=args.backend,
        num_machines=args.num_machines,
        num_gpus=args.num_gpus,
        machine_rank=args.machine_rank,
        ddp_init_method=dist_url
    )


if __name__ == '__main__':
    main()