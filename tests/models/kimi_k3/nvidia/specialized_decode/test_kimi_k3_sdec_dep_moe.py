# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode at DEP: every MegaMoE layer's ``DepMoe`` (the
EpMoeLayer schedule: split-K [shared gate_up | latent down] front, side-stream
router and shared expert, MegaMoE routed experts, latent RMSNorm + up-projection)
against the stock ``KimiMoE.forward`` on identical inputs, on real ranks (DEP2 by
default; ``KIMI_K31_DEP_MOE_WORLD=4`` runs DEP4 on one 4-GPU node).

Live routed experts: under ``--load-format dummy`` MegaMoE's packed int32 UE8M0
weight scales stay 0, so the routed experts output exactly 0 and a step test
cannot see the routed path (router, top-k weights, dispatch, combine, latent norm
and up-projection). Here every expert scale is set to 2^-2, the router weight is
scaled so the sigmoid top-16 is decisive, and the shared expert is scaled so both
branches carry comparable magnitude; the test asserts the routed experts' output is
live (a sizeable share of the layer output).

Each rank runs ``vllm bench forward`` in-process on a layer-truncated K3.1
(``KIMI_K31_DEP_MODEL``, its DSpark draft config ``KIMI_K31_DEP_DRAFT``) with real
routing. During the eager (stock) step, every MoE layer whose batch a DepMoe
serves takes the layer's input plus per-row noise (distinct rows) through both
``KimiMoE.forward`` and ``DepMoe.run`` (every DP rank at the same layer and batch,
as the MegaMoE collective requires) and compares the outputs row by row; a
row-shifted copy of the output is the negative control.

Bound: both paths quantize the latent to MXFP8 for MegaMoE's FP4 x FP8 experts;
the schedule's split-K front reduces the BF16 latent in another order, and one
BF16 ulp flips some FP8 activation codes (and the split-K FP32 router order can
flip a near-tie expert), so rows differ by up to ~2e-2. A row placed or routed
wrongly differs by O(1).
"""

import json
import os
import time

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

MODEL = os.environ.get("KIMI_K31_DEP_MODEL")
DRAFT = os.environ.get("KIMI_K31_DEP_DRAFT")
WORLD = int(os.environ.get("KIMI_K31_DEP_MOE_WORLD", "2"))
# Per DP rank: DSpark k = 4 verify (q = 5) steps of 5 / 20 / 80 rows.
CASES = ("v1=1*5@1900:4", "v4=4*5@1500:4", "v16=16*5@1100:4")
CAPTURE = (5, 20, 80)
# E8M0 2^-2 in every byte of the packed int32 UE8M0 scales: the routed latent's
# RMS then sits far above the latent RMSNorm's sqrt(eps) (at 2^-7 SiTU's quadratic
# small-input regime leaves it ~1e-5 and the norm flattens it).
LIVE_SCALE = 0x7D7D7D7D
GATE_GAIN, SHARED_GAIN = 100.0, 6.0
# Per-layer relative error of the whole batch and of any single row (measured at
# DEP2 / DEP4, 5 / 20 / 80 rows: batch <= 9.9e-3, row <= 2.1e-2; a row-shifted
# output gives >= 1.2 on every row).
BATCH_BOUND, ROW_BOUND = 2e-2, 5e-2

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_device_capability_family(100),
        reason="the specialized decode kernels need SM100/SM103",
    ),
    pytest.mark.skipif(
        not all(os.path.isdir(p or "") for p in (MODEL, DRAFT)),
        reason="needs a layer-truncated K3.1 config (KIMI_K31_DEP_MODEL) and its "
        "DSpark draft config (KIMI_K31_DEP_DRAFT)",
    ),
    pytest.mark.skipif(
        torch.accelerator.device_count() < WORLD, reason=f"Need {WORLD} GPUs."
    ),
]


def admit(capture_sizes):
    """Admit the specialized decode under ``vllm bench forward`` (one-shot graph
    captures of each exact batch; the deployment's capture sizes are
    ``capture_sizes``)."""
    import dataclasses

    import vllm.models.kimi_k3.nvidia.model as model
    import vllm.models.kimi_k3.nvidia.specialized_decode.runtime as runtime
    from vllm.config import CUDAGraphMode

    check, build = runtime.unsupported, runtime.specialized_decode_config

    def unsupported(vllm_config):
        compilation = vllm_config.compilation_config
        mode = compilation.cudagraph_mode
        compilation.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        try:
            return check(vllm_config)
        finally:
            compilation.cudagraph_mode = mode

    def specialized_decode_config(vllm_config):
        config = build(vllm_config)
        assert config is not None, "the specialized decode was not admitted"
        return dataclasses.replace(config, capture_sizes=capture_sizes)

    runtime.unsupported = unsupported
    runtime.specialized_decode_config = specialized_decode_config
    model.specialized_decode_config = specialized_decode_config


def live_moe_weights():
    """Dummy weights with live routed experts (see the module docstring)."""
    from vllm.model_executor.model_loader import base_loader

    load = base_loader.BaseModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = load(self, *args, **kwargs)
        with torch.no_grad():
            tensors = [*model.named_parameters(), *model.named_buffers()]
            for name, t in tensors:
                if ".experts." in name and "scale" in name and t.dtype == torch.int32:
                    t.fill_(LIVE_SCALE)
                elif name.endswith(".gate.weight") and t.dtype == torch.bfloat16:
                    t.mul_(GATE_GAIN)
                elif "shared_experts." in name and t.dtype == torch.bfloat16:
                    t.mul_(SHARED_GAIN)
        return model

    base_loader.BaseModelLoader.load_model = load_model  # type: ignore[method-assign]


def compare_layers(records):
    """After each eager KimiMoE.forward whose batch a DepMoe serves, run the DepMoe
    on the same rows and record the per-row error and the routed share."""
    import vllm.models.kimi_k3.nvidia.model as km
    import vllm.models.kimi_k3.nvidia.specialized_decode.runtime.dep as dep

    moes = {}
    init = dep.DepMoe.__init__

    def register(self, module, *args, **kwargs):
        init(self, module, *args, **kwargs)
        moes[id(module)] = self

    dep.DepMoe.__init__ = register  # type: ignore[method-assign]
    stock = km.KimiMoE.forward

    def forward(self, hidden_states):
        moe = moes.get(id(self))
        rows = hidden_states.shape[0]
        if (
            moe is None
            or torch.cuda.is_current_stream_capturing()
            or not moe.serves(rows)
        ):
            return stock(self, hidden_states)
        out = stock(self, hidden_states)
        # The layer's input at unit RMS plus unit per-row noise: no two rows are
        # equal (the bench's dummy batch repeats rows), and both branches keep
        # their share at every layer (the dummy residual stream grows per layer).
        gen = torch.Generator(device=hidden_states.device).manual_seed(len(records))
        noise = torch.randn(
            hidden_states.shape, generator=gen, device=hidden_states.device
        )
        h = hidden_states.float()
        rms = h.pow(2).mean().sqrt().clamp(min=1e-20)
        x = (h / rms + noise).to(hidden_states.dtype)
        routed = []
        hook = self.experts.register_forward_hook(
            lambda m, i, o: routed.append(o.float().pow(2).mean().sqrt().item())
        )
        try:
            ref = stock(self, x).float()
        finally:
            hook.remove()
        got = moe.run(x).float()
        norms = ref.norm(dim=1).clamp(min=1e-20)
        err = (got - ref).norm(dim=1) / norms
        # Negative control: the same output with its rows shifted by one.
        shifted = (got.roll(1, dims=0) - ref).norm(dim=1) / norms
        # The routed branch's share: RMSNorm(routed latent) up-projected, i.e. the
        # layer output minus the stock shared expert.
        shared = self.shared_experts(x).float()
        records.append(
            dict(
                rows=rows,
                rel=((got - ref).norm() / ref.norm()).item(),
                row_max=err.max().item(),
                shifted_min=shifted.min().item() if rows > 1 else float("inf"),
                routed_rms=routed[0] if routed else 0.0,
                routed_share=((ref - shared).norm() / ref.norm()).item(),
            )
        )
        return out

    km.KimiMoE.forward = forward  # type: ignore[method-assign]


def worker(rank, port, out_dir):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(WORLD),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
    )
    records: list = []
    admit(CAPTURE)
    live_moe_weights()
    compare_layers(records)
    import vllm.benchmarks.forward.main as fm
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    spec = {"method": "dspark", "num_speculative_tokens": 4, "model": DRAFT}
    argv = [
        "--model", MODEL, "--load-format", "dummy", "--trust-remote-code",
        "-tp", "1", "-dp", str(WORLD), "--enable-expert-parallel",
        "--routing", "real", "--moe-backend", "deep_gemm_mega_moe",
        "--kv-cache-dtype", "fp8", "--mamba-cache-dtype", "float32",
        "--mamba-ssm-cache-dtype", "float32", "--use-replayssm",
        "--max-model-len", "8192", "--language-model-only",
        "--gpu-memory-utilization", "0.8", "--max-num-seqs", "16",
        "--speculative-config", json.dumps(spec), "--exclude-drafter",
        "--additional-config", json.dumps({"decode_backend": "latency_specialized"}),
        "--kernel-config", json.dumps({"enable_flashinfer_autotune": False}),
        "--kv-len-jitter", "0", "--iters", "1", "--samples", "1",
        "--output-json", os.path.join(out_dir, "result.json"),
    ]  # fmt: skip
    for case in CASES:
        argv += ["--case", case]
    parser = FlexibleArgumentParser()
    fm.add_cli_args(parser)
    fm.main(parser.parse_args(argv))
    with open(os.path.join(out_dir, f"moe{rank}.json"), "w") as f:
        json.dump(records, f)


def test_dep_moe_matches_stock_with_live_experts(tmp_path):
    port = get_open_port()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=worker, args=(r, port, str(tmp_path))) for r in range(WORLD)
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
    for rank in range(WORLD):
        records = json.loads((tmp_path / f"moe{rank}.json").read_text())
        assert {r["rows"] for r in records} >= set(CAPTURE), records
        for rows in sorted({r["rows"] for r in records}):
            at = [r for r in records if r["rows"] == rows]
            print(
                f"rank {rank} rows {rows} ({len(at)} layer calls): rel max "
                f"{max(r['rel'] for r in at):.2e}, row max "
                f"{max(r['row_max'] for r in at):.2e}, shifted min "
                f"{min(r['shifted_min'] for r in at):.2e}, routed share "
                f"{min(r['routed_share'] for r in at):.2f}-"
                f"{max(r['routed_share'] for r in at):.2f}"
            )
        for r in records:
            # The routed experts are live: they carry a sizeable share of the output.
            # Both branches are live: the routed experts carry a sizeable share of
            # the output, and so does the shared expert.
            assert r["routed_rms"] > 0 and 0.2 < r["routed_share"] < 0.98, r
            assert r["rel"] < BATCH_BOUND, (rank, r)
            assert r["row_max"] < ROW_BOUND, (rank, r)
            # ... and the bound rejects a misplaced row.
            assert r["shifted_min"] > ROW_BOUND, (rank, r)
