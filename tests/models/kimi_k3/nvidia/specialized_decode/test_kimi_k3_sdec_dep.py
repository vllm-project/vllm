# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode at DEP2 (TP1 attention per DP rank + EP2 MoE) on 2
real ranks: admission (TP1 + EP, FP32 KDA conv and recurrent state, RecoverSSM),
the DEP plan (`SpecializedDecode.build_dep` / `plan`: q = 5 verify steps take
`run_wide`), and the step's output against the stock decode
of the same step.

Each rank runs `vllm bench forward` in-process on a layer-truncated K3.1
(`KIMI_K31_DEP_MODEL`: a config-only K3.1 directory with a few layers, e.g. 3
KDA + 1 DSA + MegaMoE layers; dummy weights, balanced EP routing). The bench
compares the step's CUDA graph replay (captured: the specialized decode) with the
same step run eagerly (outside a capture the specialized decode never plans: the
stock forward), per rank. Contexts stay below the DSA top-k (2048), so both paths
attend over every token and the routing is the bench's fixture on both: the
outputs differ only by accumulation order and FP8 / BF16 rounding.
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
# Its DSpark draft config (target layers inside the truncated model).
DRAFT = os.environ.get("KIMI_K31_DEP_DRAFT")
WORLD = 2
# Per DP rank: DSpark k = 4 verify (q = 5) steps. (Under DSpark the KDA metadata
# counts draft-less decodes as spec decodes, so a q = 1 step is not uniform and
# runs stock, at TP8 as at DEP.)
CASES = ("v1=1*5@1900:4", "v4=4*5@1500:4", "v16=16*5@1100:4")
# The deployment's capture sizes (rows).
CAPTURE = (5, 20, 80)

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


def admit(capture_sizes, plans):
    """Admit the specialized decode under `vllm bench forward` (its graphs are
    one-shot captures of each exact batch; the deployment's capture sizes are
    `capture_sizes`) and record every captured step's plan."""
    import dataclasses

    import vllm.models.kimi_k3.nvidia.model as model
    import vllm.models.kimi_k3.nvidia.specialized_decode.runtime as runtime
    from vllm.config import CUDAGraphMode
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (
        SpecializedDecode,
    )

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
    plan = SpecializedDecode.plan

    def recording(self, positions, intermediate_tensors):
        step = plan(self, positions, intermediate_tensors)
        if torch.cuda.is_current_stream_capturing():
            kinds = sorted({type(a).__name__ for a in self.attention.values()})
            moes = sorted({type(m).__name__ for m in self.wide_moe.values()})
            plans.append(
                dict(
                    rows=positions.shape[0],
                    planned=step is not None,
                    wide=bool(step is not None and step.wide),
                    q=step.q if step is not None else None,
                    dep=self.dep,
                    attention=kinds,
                    moe=moes,
                )
            )
        return step

    SpecializedDecode.plan = recording  # type: ignore[method-assign]


def worker(rank, port, out_dir):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(WORLD),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
    )
    plans: list = []
    admit(CAPTURE, plans)
    import vllm.benchmarks.forward.main as fm
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    spec = {"method": "dspark", "num_speculative_tokens": 4, "model": DRAFT}
    argv = [
        "--model", MODEL, "--load-format", "dummy", "--trust-remote-code",
        "-tp", "1", "-dp", str(WORLD), "--enable-expert-parallel",
        "--routing", "balanced", "--moe-backend", "deep_gemm_mega_moe",
        "--kv-cache-dtype", "fp8", "--mamba-cache-dtype", "float32",
        "--mamba-ssm-cache-dtype", "float32", "--use-replayssm",
        "--max-model-len", "8192", "--language-model-only",
        "--gpu-memory-utilization", "0.8", "--max-num-seqs", "16",
        "--speculative-config", json.dumps(spec), "--exclude-drafter",
        "--additional-config", json.dumps({"decode_backend": "latency_specialized"}),
        "--kernel-config", json.dumps({"enable_flashinfer_autotune": False}),
        "--kv-len-jitter", "0", "--iters", "2", "--samples", "1",
        "--output-json", os.path.join(out_dir, "result.json"),
    ]  # fmt: skip
    for case in CASES:
        argv += ["--case", case]
    parser = FlexibleArgumentParser()
    fm.add_cli_args(parser)
    fm.main(parser.parse_args(argv))
    with open(os.path.join(out_dir, f"plans{rank}.json"), "w") as f:
        json.dump(plans, f)


def test_dep2_specialized_decode_matches_stock(tmp_path):
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
    results = json.loads((tmp_path / "result.json").read_text())
    assert [r["case"] for r in results] == [c.split("=")[0] for c in CASES]
    for rank in range(WORLD):
        plans = json.loads((tmp_path / f"plans{rank}.json").read_text())
        # Every captured step took the DEP wide path, with the TP1 DSA and KDA
        # providers and the EP MoE schedule.
        assert plans, rank
        for plan in plans:
            assert plan["dep"] and plan["planned"] and plan["wide"], plan
            assert plan["attention"] == ["DepAttention", "DepKda"], plan
            assert plan["moe"] == ["DepMoe"], plan
        assert {p["rows"] for p in plans} >= {5, 20, 80}, plans
        assert {p["q"] for p in plans} == {5}, plans
    for result in results:
        for rank, r in enumerate(result["ranks"]):
            err = r["graph_vs_eager"]["relative_rms"]
            print(f"{result['case']} rank {rank}: spec vs stock rel rms {err:.2e}")
            # Order and BF16 rounding only (measured 1.4e-4 - 2.6e-4; a shared
            # expert row read at the wrong pitch gave 1.1e-2 - 1.2e-2). Dummy
            # weights leave the routed experts at 0: test_kimi_k3_sdec_dep_moe
            # covers them.
            assert err < 5e-3, (result["case"], rank, err)
