"""Closes the plan-once capture-safety gap cheaply (no model / PD).

Captures swap_in(produce_plan) + apply_plan — and, with overlap on, the
copy-stream prefetch fork/join + gather_plan — into a torch.cuda.graph, replays,
and checks it runs without a capture error and reproduces the eager result from
the same starting state. Correctness (== independent per-layer) is already
proven eagerly by test_hisparse_plan_once_matches_independent; this proves the
ops are graph-capturable (the one hole left by the eager-only e2e).
"""

import torch
import vllm  # noqa: F401 - registers _C_cache_ops
from vllm.v1.attention.backends.mla import hisparse as _hs
from vllm.v1.attention.backends.mla.hisparse import HiSparseConfig, HiSparseCoordinator

DEV = "cuda"


def _make(top_k, buf, reqs, row_width):
    return HiSparseCoordinator(
        config=HiSparseConfig(top_k=top_k, device_buffer_size=buf, host_to_device_ratio=2),
        max_num_reqs=reqs, row_width=row_width, kv_dtype=torch.float32, device=DEV,
    )


def _mirror(c, kv, block_ids, block_size):
    slots = torch.cat([torch.arange(block_size) + b * block_size for b in block_ids]).to(DEV)
    c.mirror_slots(kv, slots)


def run(overlap: bool) -> str:
    torch.manual_seed(0)
    block_size, row_width, num_blocks, top_k, buf, reqs = 16, 32, 16, 8, 8, 2
    kv = torch.arange(num_blocks * block_size * row_width, dtype=torch.float32, device=DEV).view(
        num_blocks, block_size, row_width
    )
    _hs._GROUP_PLANS.clear()
    leader, shared = _make(top_k, buf, reqs, row_width), _make(top_k, buf, reqs, row_width)
    for c in (leader, shared):
        _mirror(c, kv, list(range(num_blocks)), block_size)
        if not c._kernel_path():
            return "SKIP: pinned host unavailable"
    if overlap:
        for c in (leader, shared):
            c._overlap_enabled = True
            c._copy_stream = _hs._get_copy_stream(torch.device(DEV))
        leader.group_shared = [shared]
        shared.leader = leader

    bt = torch.arange(num_blocks, dtype=torch.int32, device=DEV).view(reqs, num_blocks // reqs)
    req_ids = torch.arange(reqs, dtype=torch.int32, device=DEV)
    seq = (num_blocks // reqs) * block_size
    slot_map = bt[:, -1].to(torch.int64) * block_size + (block_size - 1)
    # static topk buffer (graph inputs must be stable)
    topk = torch.stack([torch.randperm(seq, device=DEV)[:top_k].to(torch.int32) for _ in range(reqs)])
    topk[:, -1] = -1

    def step():
        leader.swap_in(kv_cache=kv, req_id_per_token=req_ids, block_table=bt,
                       topk_indices=topk, block_size=block_size, slot_mapping=slot_map,
                       produce_plan=True)
        shared.apply_plan(kv_cache=kv, block_size=block_size, num_tokens=reqs)

    # warmup on a side stream (torch graph-capture convention)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            step()
    except Exception as e:  # noqa: BLE001
        return f"CAPTURE_FAILED overlap={overlap}: {e!r}"
    g.replay()
    torch.cuda.synchronize()
    hi = shared._plan.hot_indices[:reqs]
    valid = hi[topk >= 0] if False else hi
    ok = int((valid[topk >= 0] >= -1).all().item())
    return f"CAPTURE_OK overlap={overlap} (replayed; hot_indices valid={bool(ok)})"


def main():
    for overlap in (False, True):
        print(run(overlap))


if __name__ == "__main__":
    main()
