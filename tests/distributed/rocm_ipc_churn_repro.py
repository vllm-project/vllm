# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone RCCL group churn + HIP IPC repro (no vLLM imports).

Phases (each in fresh processes):
  A: create/all_reduce/destroy RCCL groups of 4 and 2 ranks
  B: same, with most of the free GPU memory allocated
  C: same as B, with that memory registered with NIXL (UCX backend)
  D: same as C, plus NIXL READs from every peer's registered memory each round
"""

import argparse
import ctypes
import os
import socket
import sys
import time
import traceback
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ipc_probe(hip) -> int:
    ptr = ctypes.c_void_p()
    rc = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(2 << 20))
    if rc != 0:
        return 1000 + rc
    handle = (ctypes.c_byte * 64)()
    rc = hip.hipIpcGetMemHandle(handle, ptr)
    hip.hipFree(ptr)
    return rc


def _log(phase, rank, msg):
    print(f"[REPRO] phase={phase} rank={rank} {msg}", flush=True)


READ_BYTES = 8 << 20


def _nixl_connect(agent, rank, hold, cpu_group):
    src = hold[READ_BYTES : 2 * READ_BYTES]
    src.fill_(rank + 1)
    torch.accelerator.synchronize()
    info = (agent.get_agent_metadata(), src.data_ptr())
    gathered: list = [None] * WORLD
    dist.all_gather_object(gathered, info, group=cpu_group)
    peers = {}
    for peer in range(WORLD):
        if peer != rank:
            name = agent.add_remote_agent(gathered[peer][0])
            peers[peer] = (name, gathered[peer][1])
    return peers


def _nixl_read_all(agent, rank, hold, peers):
    dst = hold[:READ_BYTES]
    for peer, (name, remote_ptr) in peers.items():
        dst.zero_()
        torch.accelerator.synchronize()
        local = agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            agent.get_xfer_descs([(dst.data_ptr(), READ_BYTES, rank)], "VRAM"),
        )
        remote = agent.prep_xfer_dlist(
            name, agent.get_xfer_descs([(remote_ptr, READ_BYTES, peer)], "VRAM")
        )
        idx = np.arange(1, dtype=np.int32)
        xfer = agent.make_prepped_xfer("READ", local, idx, remote, idx)
        agent.transfer(xfer)
        deadline = time.time() + 60
        while True:
            state = agent.check_xfer_state(xfer)
            if state == "DONE":
                break
            if state != "PROC" or time.time() > deadline:
                raise RuntimeError(f"NIXL READ from rank {peer} state={state}")
            time.sleep(0.001)
        agent.release_xfer_handle(xfer)
        agent.release_dlist_handle(local)
        agent.release_dlist_handle(remote)
        got = int(dst[0].item()), int(dst[-1].item())
        if got != (peer + 1, peer + 1):
            raise RuntimeError(f"NIXL READ from rank {peer} returned {got}")


def _worker(rank, port, phase, rounds, mem_frac, results):
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    hip = ctypes.CDLL("libamdhip64.so")
    stage = "init"
    r = -1
    try:
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD,
            timeout=timedelta(seconds=180),
            device_id=device,
        )
        t = torch.ones(1 << 20, device=device)
        dist.all_reduce(t)
        torch.accelerator.synchronize()

        hold = None
        agent = None
        peers = None
        if phase in ("B", "C", "D"):
            stage = "alloc"
            free, total = torch.accelerator.get_memory_info(device)
            hold = torch.empty(int(free * mem_frac), dtype=torch.uint8, device=device)
            _log(
                phase,
                rank,
                f"holding {hold.numel() / 2**30:.1f} GiB of {total / 2**30:.1f} GiB",
            )
        if phase in ("C", "D") and hold is not None:
            stage = "nixl"
            from nixl_rocm._api import nixl_agent, nixl_agent_config

            agent = nixl_agent(
                f"repro-{rank}", nixl_agent_config(capture_telemetry=False)
            )
            descs = agent.get_reg_descs(
                [(hold.data_ptr(), hold.numel(), rank, "")], "VRAM"
            )
            agent.register_memory(descs)
            _log(phase, rank, "registered held memory with NIXL")
        if phase == "D" and agent is not None:
            stage = "nixl_connect"
            cpu_group = dist.new_group(backend="gloo")
            peers = _nixl_connect(agent, rank, hold, cpu_group)
            stage = "nixl_read"
            _nixl_read_all(agent, rank, hold, peers)
            _log(phase, rank, "NIXL READ from all peers ok")

        rc = _ipc_probe(hip)
        _log(phase, rank, f"ipc probe before churn rc={rc}")

        for r in range(rounds):
            ranks = list(range(WORLD)) if r % 2 == 0 else [0, 1]
            stage = f"new_group{ranks}"
            group = dist.new_group(ranks, backend="nccl")
            if rank in ranks:
                stage = f"all_reduce{ranks}"
                x = torch.ones(1 << 20, device=device)
                dist.all_reduce(x, group=group)
                torch.accelerator.synchronize()
                if x[0].item() != len(ranks):
                    raise RuntimeError(f"bad all_reduce result {x[0].item()}")
            stage = "destroy"
            dist.destroy_process_group(group)
            rc = _ipc_probe(hip)
            if rc != 0:
                raise RuntimeError(f"hipIpcGetMemHandle failed rc={rc} after group")
            if peers is not None:
                stage = "nixl_read"
                _nixl_read_all(agent, rank, hold, peers)
            stage = "barrier"
            dist.barrier(device_ids=[rank])
            if r % 10 == 9:
                _log(phase, rank, f"round {r + 1}/{rounds} ok")

        results.put((rank, "PASS", rounds, ""))
        dist.destroy_process_group()
    except Exception as e:
        _log(phase, rank, f"FAILED round={r} stage={stage}: {e!r}")
        traceback.print_exc()
        results.put((rank, "FAIL", r, f"{stage}: {e!r}"))
        sys.stdout.flush()
        os._exit(1)


def run_phase(phase, rounds, mem_frac, timeout_s):
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, port, phase, rounds, mem_frac, results))
        for rank in range(WORLD)
    ]
    start = time.time()
    for p in procs:
        p.start()
    deadline = start + timeout_s
    for p in procs:
        p.join(max(1, deadline - time.time()))
    timed_out = [p for p in procs if p.is_alive()]
    for p in timed_out:
        p.kill()
        p.join()
    got = {}
    while not results.empty():
        rank, status, r, msg = results.get()
        got[rank] = (status, r, msg)
    elapsed = time.time() - start
    ok = (
        not timed_out
        and len(got) == WORLD
        and all(v[0] == "PASS" for v in got.values())
    )
    print(
        f"[REPRO] ===== phase {phase}: {'PASS' if ok else 'FAIL'} ({elapsed:.0f}s)",
        flush=True,
    )
    for rank in range(WORLD):
        status = got.get(rank)
        if status is None:
            code = procs[rank].exitcode
            why = "timeout" if procs[rank] in timed_out else f"exit {code}"
            print(f"[REPRO]   rank {rank}: no result ({why})", flush=True)
        else:
            print(
                f"[REPRO]   rank {rank}: {status[0]} round={status[1]} {status[2]}",
                flush=True,
            )
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="A,B,C,D")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--mem-frac", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    print(
        f"[REPRO] torch {torch.__version__} hip {torch.version.hip} "
        f"rccl {torch.cuda.nccl.version()} gpus {torch.accelerator.device_count()}",
        flush=True,
    )
    for k in sorted(os.environ):
        if k.startswith(("HSA_", "NCCL_", "RCCL_", "HIP_", "UCX_")):
            print(f"[REPRO] env {k}={os.environ[k]}", flush=True)

    summary = {}
    for phase in args.phases.split(","):
        summary[phase] = run_phase(phase, args.rounds, args.mem_frac, args.timeout)
    print(
        "[REPRO] SUMMARY "
        + " ".join(f"{p}={'PASS' if v else 'FAIL'}" for p, v in summary.items()),
        flush=True,
    )
    sys.exit(0 if all(summary.values()) else 1)


if __name__ == "__main__":
    main()
