import os
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(config):
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        config.RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        config.WORLDSIZE = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        config.GPU = config.RANK % torch.cuda.device_count()
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.RANK = int(os.environ["RANK"])
        config.WORLDSIZE = int(os.environ['WORLD_SIZE'])
        config.GPU = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        config.RANK = int(os.environ['SLURM_PROCID'])
        config.GPU = config.RANK % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        config.DISTRIBUTED = False
        return

    config.DISTRIBUTED = True

    torch.cuda.set_device(config.GPU)
    config.DISTBACKEND = 'nccl'
    print('| distributed init (rank {}): {}'.format(config.RANK, config.DISTURL), flush=True)
    torch.distributed.init_process_group(backend=config.DISTBACKEND, init_method=config.DISTURL, world_size=config.WORLDSIZE, rank=config.RANK)
    torch.distributed.barrier()
    setup_for_distributed(config.RANK == 0)
