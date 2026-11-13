# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 specialized decode on MI355X, whole steps against vLLM's stock decode
of the same step: TP8 at DCP1 and DCP8, regular decode and DSpark k = 7 verify
(DCP1 verify runs M1c), fp8 KV.

Each rank runs `vllm bench forward` in-process on the 8-layer dummy Kimi-K3. The
bench compares the step's CUDA graph replay with the same step run eagerly; the
test plans the specialized decode in captures only (`_patch_plan`: by itself it
plans eager steps too), so the replay is the specialized decode and the eager run
vLLM's. Test-only changes to the bench:

- the capture without its in-graph timing events and timing
  (`_capture_and_check`): the bench hangs on ROCm, likely on those events;
- the eager runs before `graph_capture`, not inside it: there vLLM's custom
  all-reduce takes an uncaptured call for a warmup and returns
  `torch.empty_like` (no reduction), so the stock reference came out all zeros;
- seeded random KV blocks and recurrent states instead of zeros
  (`_random_kv_blocks`): over a zero context a read of the wrong page or state
  slot returns the right zeros;
- every MoE layer's routed experts zeroed (`_zero_routed_experts`), as NVIDIA's
  dummy weights leave them: vLLM's AITER fused MoE is not deterministic (two
  runs of one input differed by 0.45%), and its fp4 activations turn that into
  ~10% a layer later. test_numerics' K2 TP8 test covers the routed experts.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from tests.models.kimi_k3.amd.specialized_decode.test_numerics import needs_gfx950
from tests.utils import multi_gpu_test
from vllm.utils.network_utils import get_open_port

MODEL = os.environ.get("KIMI_K3_STEP_MODEL", "vllm-project/Kimi-K3-8L-dummy")
DRAFT = os.environ.get("KIMI_K3_STEP_DRAFT", "vllm-project/Kimi-K3-DSpark-for-8L-dummy")
WORLD = 8
# (DCP, speculative tokens) -> the steps: `NAME=N*Q@KV[:D]`, every served width
# of max_num_seqs 8 (regular decode) or 2 (verify: 8 or 16 rows)
CONFIGS = {
    (1, 0): ("d1=1*1@1900", "d3=3*1@1500", "d8=8*1@1100"),
    (8, 0): ("d1=1*1@1900", "d3=3*1@1500", "d8=8*1@1100"),
    (1, 7): ("v1=1*8@1900:7", "v2=2*8@1500:7"),
    (8, 7): ("v1=1*8@1900:7", "v2=2*8@1500:7"),
}


def _patch_plan(plans: list) -> None:
    """Plan in captures only, recording every captured step's plan."""
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.decode import (
        SpecializedDecode,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import m1c_at

    plan = SpecializedDecode.plan

    def captured_only(self, positions, intermediate_tensors):
        if not torch.cuda.is_current_stream_capturing():
            return None
        step = plan(self, positions, intermediate_tensors)
        plans.append(
            dict(
                rows=int(positions.size(-1)),
                planned=step is not None,
                m1=self.runner.m1,
                m1c=step is not None and m1c_at(self.runner, step.width),
            )
        )
        return step

    SpecializedDecode.plan = captured_only  # type: ignore[method-assign]


def _capture_and_check(self):
    """`ForwardBench._capture_and_time` without the in-graph rendezvous, timing
    events and timing, the eager runs before `graph_capture`: the same capture
    and checks."""
    from vllm.benchmarks.forward.capture import _compare, _first_tensor
    from vllm.distributed.parallel_state import graph_capture

    snapshots = [(view, view.clone()) for view in self.state_views]

    def restore() -> None:
        for view, saved in snapshots:
            view.copy_(saved)

    eager = _first_tensor(self._run_model()).clone()
    restore()
    reference = _first_tensor(self._run_model()).clone()
    with graph_capture(self.device) as capture_context:
        stream = capture_context.stream
        # per-stream setup on the capture stream (its all-reduces don't reduce)
        self._run_model()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            restore()
            output = self._run_model()
    hidden = _first_tensor(output)
    graph.replay()
    torch.accelerator.synchronize()
    first = hidden.clone()
    graph.replay()
    torch.accelerator.synchronize()
    self.result.update(
        eager_repeat=_compare(reference, eager),
        graph_vs_eager=_compare(first, reference),
        graph_deterministic=bool(torch.equal(first, hidden)),
        finite=bool(hidden.isfinite().all()),
        latency_us={"median": 0.0, "sample_spread_pct": 0.0},
    )
    del graph
    return output


@torch.inference_mode()
def _random_kv_blocks(kv_caches, attn_groups, batch, num_blocks) -> None:
    """`kv_init.zero_kv_blocks` with seeded random contents (the same on every
    rank): fp8 (E4M3) codes, bf16 conv states and fp32 recurrent states."""
    from vllm.benchmarks.forward.kv_init import _block_dim, _group_tensors

    seed = 0
    for group in attn_groups:
        ranges = batch.group_block_ranges[group.kv_cache_group_id]
        for tensor in _group_tensors(kv_caches, group, num_blocks):
            dim, factor = _block_dim(tensor, num_blocks)
            for begin, end in ranges:
                view = tensor.narrow(dim, begin * factor, (end - begin) * factor)
                g = torch.Generator(device=view.device).manual_seed(seed)
                seed += 1
                x = torch.randn(view.shape, generator=g, device=view.device)
                if view.dtype == torch.uint8:  # an fp8 (E4M3) cache's codes
                    view.copy_((x * 0.5).to(torch.float8_e4m3fn).view(torch.uint8))
                elif view.dtype == torch.float32:
                    view.copy_(x * 0.05)
                else:
                    view.copy_(x * 0.5)


def _zero_routed_experts() -> None:
    """Every MoE layer's routed experts' down projection zeroed (0 under any
    MXFP4 scale; K2 reads the same weights), as the bench installs a case's
    routing (`--routing real`)."""
    import vllm.benchmarks.forward.routing as routing

    install = routing.install_routing

    def install_routing(runner, *args, **kwargs):
        for module in runner.model.modules():
            w2 = getattr(module, "w2_weight", None)
            if isinstance(w2, torch.Tensor):
                w2.data.view(torch.uint8).zero_()
        return install(runner, *args, **kwargs)

    routing.install_routing = install_routing


def _worker(rank: int, port: int, out_dir: str, dcp: int, k: int) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(WORLD),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    plans: list = []
    _patch_plan(plans)
    import vllm.benchmarks.forward.capture as capture
    import vllm.benchmarks.forward.kv_init as kv_init

    capture.ForwardBench._capture_and_time = _capture_and_check
    kv_init.zero_kv_blocks = _random_kv_blocks
    _zero_routed_experts()

    import vllm.benchmarks.forward.main as fm
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    argv = [
        "--model", MODEL, "--load-format", "dummy", "--trust-remote-code",
        "--language-model-only", "-tp", str(WORLD), "-dcp", str(dcp),
        "--kv-cache-dtype", "fp8", "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.8", "--max-num-seqs", "2" if k else "8",
        "--attention-config", json.dumps({"mla_prefill_backend": "ROCM_AITER_FA"}),
        "--additional-config", json.dumps({"decode_backend": "latency_specialized"}),
        "--routing", "real", "--kv-len-jitter", "0", "--exclude-sampler",
        "--iters", "1", "--samples", "1",
        "--output-json", os.path.join(out_dir, "result.json"),
    ]  # fmt: skip
    if dcp > 1:
        argv += ["--dcp-comm-backend", "a2a"]
    if k:
        spec = {
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": k,
            "attention_backend": "ROCM_AITER_MLA",
        }
        argv += ["--speculative-config", json.dumps(spec), "--exclude-drafter"]
    for case in CONFIGS[(dcp, k)]:
        argv += ["--case", case]
    parser = FlexibleArgumentParser()
    fm.add_cli_args(parser)
    fm.main(parser.parse_args(argv))
    with open(os.path.join(out_dir, f"plans{rank}.json"), "w") as f:
        json.dump(plans, f)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "dcp, k", list(CONFIGS), ids=[f"dcp{d}-k{k}" for d, k in CONFIGS]
)
def test_specialized_decode_matches_stock(
    monkeypatch: pytest.MonkeyPatch, tmp_path, dcp: int, k: int
) -> None:
    # set before the workers start: unpickling `_worker` imports vLLM, which
    # reads them on import
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", "asm")
    # M1 runs W_UK / W_UV as AITER's MXFP4 bmms, off by default
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FP4BMM", "1")
    port = get_open_port()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(r, port, str(tmp_path), dcp, k))
        for r in range(WORLD)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 1500
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    assert [p.exitcode for p in procs] == [0] * WORLD
    _check(tmp_path, CONFIGS[(dcp, k)], q=k + 1, m1c=dcp == 1 and k > 0)


def _check(out: Path, cases, q: int, m1c: bool) -> None:
    results = json.loads((out / "result.json").read_text())
    assert [r["case"] for r in results] == [c.split("=")[0] for c in cases]
    rows = {int(c.split("=")[1].split("*")[0]) * q for c in cases}
    for rank in range(WORLD):
        plans = json.loads((out / f"plans{rank}.json").read_text())
        # every captured step ran the kernels, the MLA layers M1 (at DCP1
        # verify M1c in K2)
        assert {p["rows"] for p in plans} == rows, plans
        for plan in plans:
            assert plan["planned"] and plan["m1"], plan
            assert plan["m1c"] == m1c, plan
    for result in results:
        for rank, r in enumerate(result["ranks"]):
            err = r["graph_vs_eager"]["relative_rms"]
            noise = r["eager_repeat"]["relative_rms"]
            print(
                f"{result['case']} rank {rank}: vs stock rel rms {err:.2e} "
                f"(stock vs stock {noise:.2e})"
            )
            assert r["finite"] and r["graph_deterministic"], (result["case"], rank)
            # Order and rounding only (measured 7.7e-4 - 3.2e-3, the most at DCP1
            # verify; at 8+ rows the stock step alone varies by up to 1.4e-3).
            assert err < 5e-3, (result["case"], rank, err)
