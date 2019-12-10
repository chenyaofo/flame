import os
import typing
import torch
import torch.distributed as dist


def init(backend="nccl", init_method="env://"):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if dist.is_available():
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ['WORLD_SIZE'])

            master_addr = os.environ["MASTER_ADDR"]
            master_port = os.environ["MASTER_PORT"]

            dist.init_process_group(backend=backend,
                                    init_method=init_method,
                                    world_size=world_size,
                                    rank=rank)
            print(f"Init distributed mode(backend={backend}, "
                  f"init_mothod={master_addr}:{master_port}, "
                  f"rank={rank}, pid={os.getpid()}, world_size={world_size}, "
                  f"is_master={is_master()}).")
            return backend, init_method, rank, local_rank, world_size, master_addr, master_port
        else:
            print("Fail to init distributed because torch.distributed is unavailable.")
        return None, None, 0, 0, 1, None, None


def is_dist_avail_and_init():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist_avail_and_init() else 0


def local_rank():
    return int(os.environ["LOCAL_RANK"]) if is_dist_avail_and_init() else 0


def world_size():
    return dist.get_world_size() if is_dist_avail_and_init() else 1


def is_master():
    return rank() == 0


_str_2_reduceop = dict(
    sum=dist.ReduceOp.SUM,
    mean=dist.ReduceOp.SUM,
    product=dist.ReduceOp.PRODUCT,
    min=dist.ReduceOp.MIN,
    max=dist.ReduceOp.MAX,
)


def all_reduce_array(*args, reduction="sum"):
    t = torch.tensor(args, dtype=torch.float).cuda()
    dist.all_reduce(t, op=_str_2_reduceop[reduction])
    array = t.tolist()
    if reduction == "mean":
        world_size = dist.get_world_size()
        array = [item/world_size for item in array]
    return array
