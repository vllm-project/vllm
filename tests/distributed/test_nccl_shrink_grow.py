import os
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def unique_id_to_bytes(unique_id: ncclUniqueId) -> bytes:
    return bytes(unique_id.internal)


def unique_id_from_bytes(data: bytes) -> ncclUniqueId:
    unique_id = ncclUniqueId()
    for i, byte in enumerate(data):
        unique_id.internal[i] = byte
    return unique_id


def has_symbol(comm: PyNcclCommunicator, name: str) -> bool:
    return name in comm.nccl._funcs


def sync_and_time(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def print_max_ms(rank: int, name: str, value_ms: float) -> None:
    tensor = torch.tensor([value_ms], dtype=torch.float64)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"{name}: {float(tensor.item()):.3f} ms")


def local_rank(active_ranks: list[int], rank: int) -> int:
    return active_ranks.index(rank)


def check_all_reduce(
    comm: PyNcclCommunicator,
    rank: int,
    active_ranks: list[int],
    phase: str,
) -> None:
    expected = sum(active_rank + 1 for active_rank in active_ranks)
    tensor = torch.ones(1, dtype=torch.float32, device=comm.device) * (rank + 1)
    output = comm.all_reduce(tensor)
    torch.cuda.synchronize()
    actual = float(output.cpu().item())
    assert actual == expected, (phase, "all_reduce", rank, actual, expected)


def check_all_gather(
    comm: PyNcclCommunicator,
    rank: int,
    active_ranks: list[int],
    phase: str,
) -> None:
    elems = 8
    input_tensor = (
        torch.arange(elems, dtype=torch.float32, device=comm.device) + rank * 100
    )
    output = torch.empty(
        elems * len(active_ranks),
        dtype=torch.float32,
        device=comm.device,
    )
    comm.all_gather(output, input_tensor)
    torch.cuda.synchronize()

    expected = torch.cat(
        [
            torch.arange(elems, dtype=torch.float32, device=comm.device)
            + active_rank * 100
            for active_rank in active_ranks
        ]
    )
    torch.testing.assert_close(output, expected, msg=f"{phase} all_gather rank={rank}")


def check_reduce_scatter(
    comm: PyNcclCommunicator,
    rank: int,
    active_ranks: list[int],
    phase: str,
) -> None:
    elems_per_rank = 8
    world_size = len(active_ranks)
    total_elems = elems_per_rank * world_size
    input_tensor = (
        torch.arange(total_elems, dtype=torch.float32, device=comm.device)
        + rank * 1000
    )
    output = torch.empty(elems_per_rank, dtype=torch.float32, device=comm.device)
    comm.reduce_scatter(output, input_tensor)
    torch.cuda.synchronize()

    lr = local_rank(active_ranks, rank)
    start = lr * elems_per_rank
    end = start + elems_per_rank
    expected = sum(
        (
            torch.arange(total_elems, dtype=torch.float32, device=comm.device)
            + active_rank * 1000
        )[start:end]
        for active_rank in active_ranks
    )
    torch.testing.assert_close(
        output,
        expected,
        msg=f"{phase} reduce_scatter rank={rank}",
    )


def check_broadcast(
    comm: PyNcclCommunicator,
    rank: int,
    active_ranks: list[int],
    phase: str,
) -> None:
    for root_local, root_global in enumerate(active_ranks):
        tensor = torch.empty(1, dtype=torch.float32, device=comm.device)
        if rank == root_global:
            tensor.fill_(root_global + 11)
        comm.broadcast(tensor, src=root_local)
        torch.cuda.synchronize()
        actual = float(tensor.cpu().item())
        expected = root_global + 11
        assert actual == expected, (
            phase,
            "broadcast",
            rank,
            root_global,
            actual,
            expected,
        )


def check_send_recv(
    comm: PyNcclCommunicator,
    rank: int,
    active_ranks: list[int],
    phase: str,
) -> None:
    lr = local_rank(active_ranks, rank)
    if len(active_ranks) < 2 or lr >= 2:
        return

    if lr == 0:
        tensor = torch.ones(4, dtype=torch.float32, device=comm.device) * (rank + 17)
        comm.send(tensor, dst=1)
    else:
        tensor = torch.empty(4, dtype=torch.float32, device=comm.device)
        comm.recv(tensor, src=0)
        expected = torch.ones_like(tensor) * (active_ranks[0] + 17)
        torch.testing.assert_close(
            tensor,
            expected,
            msg=f"{phase} send_recv rank={rank}",
        )
    torch.cuda.synchronize()


def check_phase(
    comm: PyNcclCommunicator,
    rank: int,
    active_ranks: list[int],
    phase: str,
) -> None:
    if rank not in active_ranks:
        return

    assert comm.world_size == len(active_ranks), (
        phase,
        rank,
        comm.world_size,
        len(active_ranks),
    )
    assert comm.rank == local_rank(active_ranks, rank), (
        phase,
        rank,
        comm.rank,
        local_rank(active_ranks, rank),
    )

    check_all_reduce(comm, rank, active_ranks, phase)
    check_all_gather(comm, rank, active_ranks, phase)
    check_reduce_scatter(comm, rank, active_ranks, phase)
    check_broadcast(comm, rank, active_ranks, phase)
    check_send_recv(comm, rank, active_ranks, phase)


def worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    comm = PyNcclCommunicator(group=dist.group.WORLD, device=device)

    try:
        if rank == 0:
            print(f"NCCL version: {comm.nccl.ncclGetVersion()}")
            print(
                "symbols: "
                f"shrink={has_symbol(comm, 'ncclCommShrink')}, "
                f"get_grow_id={has_symbol(comm, 'ncclCommGetUniqueId')}, "
                f"grow={has_symbol(comm, 'ncclCommGrow')}"
            )

        full_ranks = list(range(world_size))
        retained_ranks = list(range(world_size // 2))
        excluded_ranks = list(range(world_size // 2, world_size))

        dist.barrier()
        check_phase(comm, rank, full_ranks, "initial")
        dist.barrier()
        if rank == 0:
            print("initial correctness passed")

        if not has_symbol(comm, "ncclCommShrink"):
            if rank == 0:
                print("skip shrink/grow: ncclCommShrink is unavailable")
            return

        retained = False

        def do_shrink() -> None:
            nonlocal retained
            retained = comm.shrink(excluded_ranks, destroy_old=False)

        dist.barrier()
        shrink_ms = sync_and_time(do_shrink)
        print_max_ms(rank, "shrink rebuild", shrink_ms if retained else 0.0)
        dist.barrier()

        check_phase(comm, rank, retained_ranks, "shrink")
        dist.barrier()
        if rank == 0:
            print("shrink correctness passed")

        if not (
            has_symbol(comm, "ncclCommGetUniqueId")
            and has_symbol(comm, "ncclCommGrow")
        ):
            if rank == 0:
                print("skip grow: ncclCommGetUniqueId/ncclCommGrow unavailable")
            return

        if rank == retained_ranks[0]:
            obj = [unique_id_to_bytes(comm.get_grow_unique_id())]
        else:
            obj = [None]
        dist.broadcast_object_list(obj, src=retained_ranks[0])
        grow_id = unique_id_from_bytes(obj[0])

        def do_grow() -> None:
            if retained:
                comm.grow(new_world_size=world_size, destroy_old=False)
            else:
                comm.grow(
                    new_world_size=world_size,
                    grow_unique_id=grow_id,
                    new_rank=rank,
                    destroy_old=False,
                )

        dist.barrier()
        grow_ms = sync_and_time(do_grow)
        print_max_ms(rank, "grow rebuild", grow_ms)
        dist.barrier()

        check_phase(comm, rank, full_ranks, "grow")
        dist.barrier()
        if rank == 0:
            print("grow correctness passed")

    finally:
        try:
            comm.destroy()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    world_size = int(os.environ.get("NCCL_SHRINK_GROW_WORLD_SIZE", "4"))
    assert world_size >= 2
    assert world_size % 2 == 0
    assert torch.cuda.device_count() >= world_size
    mp.spawn(worker, args=(world_size, find_free_port()), nprocs=world_size)
