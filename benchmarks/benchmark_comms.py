# pyright: basic
import argparse
import itertools
import multiprocessing
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from tqdm import tqdm
from typing_extensions import Self

import vllm.distributed.parallel_state as pstate
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_open_port

logger = init_logger(__name__)


def get_device_idx(i):
    return pstate.get_device_idx(i)


def alloc_dev_buf(size: int, device: torch.device) -> torch.Tensor:
    return torch.rand(size // 2, dtype=torch.float16, device=device)


@dataclass
class BMResult:
    bm_name: str
    minimum: float
    maximum: float
    median: float
    average: float

    @staticmethod
    def p2p_bw_from_latencies(latencies: List[float], msize: int):
        bwlst = list(map(lambda x: msize * 1e-9 / x, latencies))
        bwlst.sort()
        return BMResult("p2p",
                        minimum=bwlst[0],
                        maximum=bwlst[-1],
                        median=bwlst[len(bwlst) // 2],
                        average=sum(bwlst) / len(bwlst))

    @staticmethod
    def ag_busbw_from_latencies(latencies: List[float], msize: int,
                                ndevs: int):
        # B = S/t * (n-1)/n = algbw * (n-1)/n
        # S is total buffer, i.e. received data
        bwlst = list(
            map(lambda x: (msize * 1e-9 * (ndevs) / x) * (ndevs - 1) / ndevs,
                latencies))
        bwlst.sort()
        return BMResult("all_gather",
                        minimum=bwlst[0],
                        maximum=bwlst[-1],
                        median=bwlst[len(bwlst) // 2],
                        average=sum(bwlst) / len(bwlst))

    @staticmethod
    def ar_busbw_from_latencies(latencies: List[float], msize: int,
                                ndevs: int):
        # B = S/t * (2*(n-1)/n) = algbw * (2*(n-1)/n)
        bwlst = list(
            map(lambda x: (msize * 1e-9 / x) * 2 * (ndevs - 1) / ndevs,
                latencies))
        bwlst.sort()
        return BMResult("all_reduce",
                        minimum=bwlst[0],
                        maximum=bwlst[-1],
                        median=bwlst[len(bwlst) // 2],
                        average=sum(bwlst) / len(bwlst))

    @staticmethod
    def g_busbw_from_latencies(latencies: List[float], msize: int):
        # B = S/t * n = algbw * n
        bwlst = list(map(lambda x: (msize * 1e-9 / x), latencies))
        bwlst.sort()
        return BMResult("gather",
                        minimum=bwlst[0],
                        maximum=bwlst[-1],
                        median=bwlst[len(bwlst) // 2],
                        average=sum(bwlst) / len(bwlst))

    @staticmethod
    def b_busbw_from_latencies(latencies: List[float], msize: int):
        # B = S/t * n = algbw * n
        bwlst = list(map(lambda x: (msize * 1e-9 / x), latencies))
        bwlst.sort()
        return BMResult("broadcast",
                        minimum=bwlst[0],
                        maximum=bwlst[-1],
                        median=bwlst[len(bwlst) // 2],
                        average=sum(bwlst) / len(bwlst))


@dataclass
class BMConfig:
    ndevs: int
    tp_size: int
    pp_size: int
    distributed_init_method: str
    iters: int
    warmups: int
    msize: int
    metric: str
    bm_name: str
    bm_print: Callable[[List[BMResult], Self], None] | None
    bm_fn: Callable[[Self], List[BMResult]] | None


# g0 and g1 are ranks in gc
# when using pytorch, g0 and g1 are global_ranks, regardless of group argument
# this is identical to pytorch's documentation.
# IMO, this is dumb, but I'm sure there's a reason for it...
# There also seem to be some lockups w/ passing in pg when doing pp...
def measure_p2p_bw(buff: torch.Tensor, g0: int, g1: int, bmcfg: BMConfig,
                   gc: GroupCoordinator) -> BMResult:
    rank = gc.rank_in_group
    device = gc.device
    latencies = []

    assert rank in (g0, g1)
    assert buff.device == device

    sync_word = alloc_dev_buf(4, device)

    for i in range(bmcfg.iters + bmcfg.warmups):
        if rank == g0:
            gc.recv(sync_word.shape, sync_word.dtype, g1)
            torch.cuda.synchronize(device)
            ts = time.perf_counter()
            gc.send(buff, g1)
        else:
            gc.send(sync_word, g0)
            torch.cuda.synchronize(device)
            ts = time.perf_counter()
            buff = gc.recv(buff.shape, buff.dtype, g0)

        torch.cuda.synchronize(device)
        tf = time.perf_counter() - ts

        if i >= bmcfg.warmups:
            latencies.append(tf)

    assert len(latencies) == bmcfg.iters

    return BMResult.p2p_bw_from_latencies(latencies, bmcfg.msize)


def measure_tdict_bw(tdict: Dict[str, Union[torch.Tensor, Any]], g0: int,
                     g1: int, bmcfg: BMConfig) -> BMResult:
    ppgc = pstate.get_pp_group()
    tpgc = pstate.get_tp_group()
    pprank = ppgc.rank_in_group
    tprank = tpgc.rank_in_group
    device = pstate.get_world_group().device
    latencies = []

    assert pprank in (g0, g1)
    assert all([tdict[k].device == device for k in tdict])
    # tensors are reshaped based on all-gather group, sending less data overall
    assert sum([tdict[k].nbytes for k in tdict]) == bmcfg.msize

    sync_word = alloc_dev_buf(4, device)

    tdict_p2p_iter = range(bmcfg.iters + bmcfg.warmups)
    if tprank == 0 and pprank == g0:
        tdict_p2p_iter = tqdm(tdict_p2p_iter,
                              desc="send/recv_tensor_dict iterations")

    for i in tdict_p2p_iter:
        if pprank == g0:
            ppgc.recv(sync_word.shape, sync_word.dtype, g1)
            torch.cuda.synchronize(device)
            ts = time.perf_counter()
            ppgc.send_tensor_dict(tdict, g1, tpgc)
        else:
            ppgc.send(sync_word, g0)
            torch.cuda.synchronize(device)
            ts = time.perf_counter()
            ppgc.recv_tensor_dict(g0, tpgc)

        torch.cuda.synchronize(device)
        tf = time.perf_counter() - ts

        if i >= bmcfg.warmups:
            latencies.append(tf)

    assert len(latencies) == bmcfg.iters

    return BMResult.p2p_bw_from_latencies(latencies, bmcfg.msize)


def measure_all_gather(buff: torch.Tensor, bmcfg: BMConfig,
                       gc: GroupCoordinator) -> BMResult:
    assert buff.nbytes == bmcfg.msize

    ag_bm_iter = range(bmcfg.iters + bmcfg.warmups)
    if gc.rank == 0:
        ag_bm_iter = tqdm(ag_bm_iter, "All_gather")

    latencies = []
    device = gc.device
    for i in ag_bm_iter:
        gc.barrier()
        ts = time.perf_counter()
        gc.all_gather(buff)
        torch.cuda.synchronize(device)
        tf = time.perf_counter() - ts

        if i >= bmcfg.warmups:
            latencies.append(tf)

    assert len(latencies) == bmcfg.iters

    return BMResult.ag_busbw_from_latencies(latencies, bmcfg.msize,
                                            bmcfg.tp_size)


def measure_all_reduce(buff: torch.Tensor, bmcfg: BMConfig,
                       gc: GroupCoordinator) -> BMResult:
    assert buff.nbytes == bmcfg.msize

    ar_bm_iter = range(bmcfg.iters + bmcfg.warmups)
    if gc.rank == 0:
        ar_bm_iter = tqdm(ar_bm_iter, "All_reduce")

    device = gc.device
    latencies = []
    for i in ar_bm_iter:
        gc.barrier()
        ts = time.perf_counter()
        gc.all_reduce(buff)
        torch.cuda.synchronize(device)
        tf = time.perf_counter() - ts
        if i >= bmcfg.warmups:
            latencies.append(tf)

    assert len(latencies) == bmcfg.iters

    return BMResult.ar_busbw_from_latencies(latencies, bmcfg.msize,
                                            bmcfg.tp_size)


def measure_gather(buff: torch.Tensor, bmcfg: BMConfig,
                   gc: GroupCoordinator) -> BMResult:
    assert buff.nbytes == bmcfg.msize

    ar_bm_iter = range(bmcfg.iters + bmcfg.warmups)
    if gc.rank == 0:
        ar_bm_iter = tqdm(ar_bm_iter, "Gather")

    device = gc.device
    latencies = []
    for i in ar_bm_iter:
        gc.barrier()
        ts = time.perf_counter()
        gc.gather(buff)
        torch.cuda.synchronize(device)
        tf = time.perf_counter() - ts
        if i >= bmcfg.warmups:
            latencies.append(tf)

    assert len(latencies) == bmcfg.iters

    return BMResult.g_busbw_from_latencies(latencies, bmcfg.msize)


def measure_broadcast(buff: torch.Tensor, bmcfg: BMConfig,
                      gc: GroupCoordinator) -> BMResult:
    assert buff.nbytes == bmcfg.msize

    ar_bm_iter = range(bmcfg.iters + bmcfg.warmups)
    if gc.rank == 0:
        ar_bm_iter = tqdm(ar_bm_iter, "Broadcast")

    device = gc.device
    latencies = []
    for i in ar_bm_iter:
        gc.barrier()
        ts = time.perf_counter()
        gc.broadcast(buff)
        torch.cuda.synchronize(device)
        tf = time.perf_counter() - ts
        if i >= bmcfg.warmups:
            latencies.append(tf)

    assert len(latencies) == bmcfg.iters

    return BMResult.b_busbw_from_latencies(latencies, bmcfg.msize)


def print_p2p_bm(results: List[BMResult], bmcfg: BMConfig):
    assert len(results) == bmcfg.ndevs * bmcfg.ndevs
    assert all([r.bm_name == "p2p" for r in results])
    ndevs = bmcfg.ndevs

    print(f"NCCL P2P Bandwidth msize: {bmcfg.msize}")
    header = f"{bmcfg.metric:<13}"
    for g in range(ndevs):
        header += f"R{g:<11}"
    print(header)

    for g0 in range(ndevs):
        row = f"R{g0:<4}"
        for g1 in range(ndevs):
            val = getattr(results[g0 * ndevs + g1], bmcfg.metric)
            row += f"{val:8.2f}GB/s"
        print(row)


def bm_p2p_bw(bmcfg: BMConfig) -> List[BMResult]:
    wgc = pstate.get_world_group()
    world_size = wgc.world_size
    rank = wgc.rank
    device = wgc.device
    sendbuff = alloc_dev_buf(bmcfg.msize, device)

    disable_tqdm = rank != 0

    results = []
    for g0 in tqdm(range(world_size), desc="Send rank", disable=disable_tqdm):
        for g1 in tqdm(range(world_size),
                       desc="Recv rank",
                       disable=disable_tqdm):
            if g0 == g1:
                if rank == g0:
                    results.append(BMResult("p2p", 0.0, 0.0, 0.0, 0.0))
                continue

            wgc.barrier()

            if rank in (g0, g1):
                res = measure_p2p_bw(sendbuff, g0, g1, bmcfg, wgc)
                if rank == g0:
                    results.append(res)

            wgc.barrier()

    return results


def print_tp_colls(results: List[BMResult], bmcfg: BMConfig):
    assert len(results) == bmcfg.ndevs * 4

    header_str = "TP xCCL Collective Bus-Bandwidth, "
    header_str += f"msize: {bmcfg.msize} tp_size: {bmcfg.tp_size} "
    header_str += f"pp_size: {bmcfg.pp_size}"
    print(header_str)

    for i in range(bmcfg.ndevs):
        ag_res = results[i * 4]
        ar_res = results[i * 4 + 1]
        g_res = results[i * 4 + 2]
        b_res = results[i * 4 + 3]

        rank = i
        dev_idx = get_device_idx(i)
        tp_group = i // bmcfg.tp_size
        print(f"Rank: {rank:<3} dev: {dev_idx:<3} tp_group: {tp_group:<3} ",
              end="")
        for m in [bmcfg.metric]:
            ag_m = getattr(ag_res, m)
            ar_m = getattr(ar_res, m)
            g_m = getattr(g_res, m)
            b_m = getattr(b_res, m)

            line_str = f"{m:8} all_gather: {ag_m:6.2f}GB/s, "
            line_str += f"all_reduce: {ar_m:6.2f}GB/s, gather: {g_m:6.2f}GB/s, "
            line_str += f"broadcast: {b_m:6.2f}GB/s"
            print(line_str)


def bm_tp_colls(bmcfg: BMConfig) -> List[BMResult]:

    tpgc = pstate.get_tp_group()

    msize = bmcfg.msize
    device = tpgc.device

    buff = alloc_dev_buf(msize, device)

    ag_results = measure_all_gather(buff, bmcfg, tpgc)
    ar_results = measure_all_reduce(buff, bmcfg, tpgc)
    g_results = measure_gather(buff, bmcfg, tpgc)
    b_results = measure_broadcast(buff, bmcfg, tpgc)
    # TODO

    tpgc.barrier()

    return [ag_results, ar_results, g_results, b_results]


def print_pp_p2p(results: List[BMResult], bmcfg: BMConfig):
    return _print_pp_p2p(results, bmcfg, "send/recv")


def print_pp_tdict(results: List[BMResult], bmcfg: BMConfig):
    return _print_pp_p2p(results, bmcfg, "tensor_dict send/recv")


def _print_pp_p2p(results: List[BMResult], bmcfg: BMConfig, name: str):

    assert len(
        results
    ) == bmcfg.ndevs, f"len(results): {len(results)}, expected: {bmcfg.ndevs}"
    assert all([r.bm_name == "p2p" or r.bm_name == "none" for r in results])

    header_str = f"PP xCCL {name} bandwidth, msize: {bmcfg.msize} "
    header_str += f"tp_size: {bmcfg.tp_size} pp_size: {bmcfg.pp_size}"
    print(header_str)

    num_pp_groups = bmcfg.ndevs // bmcfg.pp_size
    for i in range(num_pp_groups):
        for j in range(bmcfg.pp_size - 1):
            s = i + j * bmcfg.tp_size
            r = i + (j + 1) * bmcfg.tp_size
            sdev_idx = get_device_idx(s)
            rdev_idx = get_device_idx(r)
            bw = getattr(results[s], bmcfg.metric)

            line = f"rank: {s:<3} cuda:{sdev_idx:<3} ==>"
            line += f" rank: {r:<3} cuda:{rdev_idx:<3}"
            line += f" {bw:<8.2f}GB/s"
            print(line)


def bm_pp_p2p(bmcfg: BMConfig) -> List[BMResult]:
    ppgc = pstate.get_pp_group()
    device = ppgc.device
    rank = ppgc.rank
    rank_in_group = ppgc.rank_in_group

    buff = alloc_dev_buf(bmcfg.msize, device)

    pipeline_iter = range(bmcfg.pp_size - 1)
    if rank == 0:
        pipeline_iter = tqdm(pipeline_iter)

    results = []
    for i in pipeline_iter:
        if rank_in_group == i or rank_in_group == i + 1:
            res = measure_p2p_bw(buff, i, i + 1, bmcfg, ppgc)
            if rank_in_group == i:
                results.append(res)
        ppgc.barrier()

    if len(results) == 0:
        results.append(BMResult("none", 0.0, 0.0, 0.0, 0.0))

    assert len(results) == 1

    return results


def bm_pp_tdict(bmcfg: BMConfig) -> List[BMResult]:
    ppgc = pstate.get_pp_group()
    device = ppgc.device
    rank_in_group = ppgc.rank_in_group

    # 'inspired' by vllm/model_executor/models/llamapy::345
    tdict = {
        "hidden_states": alloc_dev_buf(bmcfg.msize // 2, device),
        "residual": alloc_dev_buf(bmcfg.msize // 2, device),
    }

    results = []
    pipeline_iter = range(bmcfg.pp_size - 1)
    if ppgc.rank == 0:
        pipeline_iter = tqdm(pipeline_iter, desc="Pipeline stages")

    for i in pipeline_iter:
        if rank_in_group == i or rank_in_group == i + 1:
            res = measure_tdict_bw(tdict, i, i + 1, bmcfg)
            if rank_in_group == i:
                results.append(res)
        ppgc.barrier()

    if len(results) == 0:
        results.append(BMResult("none", 0.0, 0.0, 0.0, 0.0))

    assert len(results) == 1

    return results


def proc_fn(cfgtpl: Tuple[BMConfig, int, torch.device]):
    bmcfg = cfgtpl[0]
    rank = cfgtpl[1]
    device = cfgtpl[2]
    backend = "nccl"
    torch.cuda.set_device(device)
    pstate.init_distributed_environment(
        world_size=bmcfg.ndevs,
        rank=rank,
        distributed_init_method=bmcfg.distributed_init_method,
        local_rank=rank,
        device=device,
        backend=backend)

    pstate.initialize_model_parallel(bmcfg.tp_size, bmcfg.pp_size, backend)

    assert pstate.get_world_group().device == pstate.get_pp_group().device
    assert pstate.get_world_group().device == pstate.get_tp_group().device
    assert rank == pstate.get_world_group().rank

    print(f"rank: {rank}, device: {device}")

    if bmcfg.bm_name == "all":
        results = []
        for bm in bm_dict:
            results += bm_dict[bm][0](bmcfg)
    else:
        results = bm_dict[bmcfg.bm_name][0](bmcfg)

    pstate.destroy_model_parallel()
    pstate.destroy_distributed_environment()

    return results


bm_dict = {
    "p2p_bw": (bm_p2p_bw, print_p2p_bm),
    "tp_colls": (bm_tp_colls, print_tp_colls),
    "pp_p2p": (bm_pp_p2p, print_pp_p2p),
    "pp_tdict": (bm_pp_tdict, print_pp_tdict),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--msize", type=int, default=1 << 28)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--metric",
                        type=str,
                        choices=["average", "minimum", "maximum", "median"],
                        default="median")
    parser.add_argument("--bm-name",
                        type=str,
                        choices=list(bm_dict.keys()) + ["all"],
                        default="all")
    args = parser.parse_args()

    argsdict = vars(args)
    argsdict["ndevs"] = min(torch.cuda.device_count(),
                            argsdict["pp_size"] * argsdict["tp_size"])
    argsdict["bm_fn"], argsdict["bm_print"] = bm_dict[
        argsdict["bm_name"]] if argsdict["bm_name"] != "all" else None, None
    argsdict["distributed_init_method"] = get_distributed_init_method(
        "127.0.0.1", get_open_port())

    bmcfg = BMConfig(**argsdict)
    print(bmcfg)

    cfgs = []
    for i in range(bmcfg.ndevs):
        cfgs.append((bmcfg, i, torch.device(f"cuda:{get_device_idx(i)}")))

    assert len(cfgs) == bmcfg.ndevs

    p = multiprocessing.Pool(bmcfg.ndevs)
    results = p.map(proc_fn, cfgs)

    if bmcfg.bm_name == "all":
        n_p2p_bw_res = bmcfg.ndevs
        n_tp_coll_res = 4
        n_pp_p2p_res = 1
        n_pp_tdict_res = 1

        p2p_bw_eidx = bmcfg.ndevs * n_p2p_bw_res
        tp_coll_eidx = p2p_bw_eidx + bmcfg.ndevs * n_tp_coll_res
        pp_p2p_eidx = tp_coll_eidx + bmcfg.ndevs * n_pp_p2p_res
        pp_tdict_eidx = pp_p2p_eidx + bmcfg.ndevs * n_pp_tdict_res

        flat_reslist = []
        for r in results:
            flat_reslist += r[0:n_p2p_bw_res]
        for r in results:
            flat_reslist += r[n_p2p_bw_res:n_p2p_bw_res + n_tp_coll_res]
        for r in results:
            flat_reslist += r[n_p2p_bw_res + n_tp_coll_res:n_p2p_bw_res +
                              n_tp_coll_res + n_pp_p2p_res]
        for r in results:
            flat_reslist += r[n_p2p_bw_res + n_tp_coll_res +
                              n_pp_p2p_res:n_p2p_bw_res + n_tp_coll_res +
                              n_pp_p2p_res + n_pp_tdict_res]

        for bm in bm_dict:
            if bm == "p2p_bw":
                sidx = 0
                eidx = p2p_bw_eidx
            elif bm == "tp_colls":
                sidx = p2p_bw_eidx
                eidx = tp_coll_eidx
            elif bm == "pp_p2p":
                sidx = tp_coll_eidx
                eidx = pp_p2p_eidx
            elif bm == "pp_tdict":
                sidx = pp_p2p_eidx
                eidx = pp_tdict_eidx

            bm_dict[bm][1](flat_reslist[sidx:eidx], bmcfg)

    else:
        bm_dict[bmcfg.bm_name][1](list(itertools.chain(*results)), bmcfg)

    p.close()
    p.join()


if __name__ == "__main__":
    main()
