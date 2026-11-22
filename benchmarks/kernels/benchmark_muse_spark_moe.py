# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Muse Spark NVFP4 MoE experts on Hopper (SM90).

Times one expert-parallel rank of a Muse Spark NVFP4 MoE layer (EP8: 32 of 256
experts, hidden and intermediate size 4096, top-8) from routed hidden states to
the finalized output: both expert GEMMs, SwiGLU, the per-route post-expert
RMSNorm and the FP32 top-k weighted sum. Arms:

- `hopper_nvfp4`: `MuseMoERunner` with `moe_backend="auto"`, which selects
  `MuseSparkHopperNvFp4Experts` on SM90.
- `hopper_nvfp4_twin`: a second, independent `hopper_nvfp4` instance (A/A null
  twin).
- `marlin`: `MuseMoERunner` with `moe_backend="marlin"` (`MarlinExperts`).
- `humming_grouped`, `humming_indexed`: Humming NVFP4 GEMMs in the grouped and
  indexed layouts, with Marlin's fold of the split W13 global scales, followed
  by the same `MusePostExpertFinalize`. Skipped without the `humming` package.

All arms share the raw NVFP4 weights, the routings and the post-expert norm.

Protocol for every `(pool, T)`:

- Draw `N` routings with seeds `pool + 37 * T + i`, from captured traces
  (`--routing-dir`, relabelled so the `--ep-rank` rank is local) or synthetic.
- Capture one CUDA graph per arm that runs the `N` calls back to back
  (`--no-graph` runs them eagerly). With `--l2-flush call`, every call is
  preceded by an L2 flush and a flush-only pseudo-arm, timed in the same rounds,
  is subtracted; the default `none` leaves the cache warm between calls.
- Time whole cycles of a cyclic Latin square of arm orders and its reverse.
- Report 5%-trimmed latency, effective weight bandwidth, speedups of
  `hopper_nvfp4` over every other arm with 95% bootstrap confidence intervals,
  the null twin, bitwise twin equality, relative L2 error against an FP32
  reference and peak memory; `--json` also stores every per-round sample.

Captured routing archives hold global top-8 expert IDs for layers `L0` to
`L61`. `verify_B{1,4,16,32}_real.npz` and `extend_real.npz` store `offsets`
(`[num_forwards + 1]` token offsets) and one `[tokens, 8]` array per layer;
`prefill/prefill_<tag>_c<C>burst_k<NN>.npz` stores `ids` (`[62, tokens, 8]`)
for consecutive chunks of one prompt stream.

Usage:

```console
python benchmarks/kernels/benchmark_muse_spark_moe.py --json decode.json
python benchmarks/kernels/benchmark_muse_spark_moe.py --mode prefill --no-graph
python benchmarks/kernels/benchmark_muse_spark_moe.py --routing-dir DIR --json r.json
```

Without `--routing-dir` the routing is synthetic (`--synthetic-dist`). With
`--checkpoint-dir` and `--layer`, the arms use that layer's real experts of EP
rank `--ep-rank` instead of synthetic weights.
"""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import regex as re
import torch
import torch.nn.functional as F

import vllm
from vllm.config import (
    KernelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.kernel import MoEBackend
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    UnfinalizedMoEOutput,
    route_major_moe_output,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
    get_moe_permute_scratch,
    moe_permute_unpermute_supported,
    moe_prepare_scatter,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.utils.humming.activation import (
    get_humming_activation,
)
from vllm.model_executor.layers.quantization.utils.humming.schema import (
    check_and_fallback_input_schema,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.models.muse_spark.common.layers import MuseSparkRMSNormWithInputScale
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MuseMoERunner,
    MusePostExpertFinalize,
    MuseSparkRoutedExperts,
    MuseSparkRouter,
)
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.import_utils import has_humming
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

# Muse Spark EP8 geometry.
E_GLOBAL, EP_SIZE, E_LOCAL = 256, 8, 32
HIDDEN_DIM, INTERMEDIATE_DIM, TOP_K = 4096, 4096, 8
NVFP4_BLOCK, E4M3_MAX, E4M3_MIN_SUBNORMAL = 16, 448.0, 2.0**-9
E2M1_MEAN_SQUARE = 8.5625  # mean of v**2 over the 8 E2M1 magnitudes
POST_NORM_EPS = 1e-8
# Bytes one local expert streams: FP4 codes plus E4M3 block scales of W13 + W2.
WEIGHT_BYTES_PER_EXPERT = 3 * HIDDEN_DIM * INTERMEDIATE_DIM * 9 // 16
FLOPS_PER_ROUTE = 2 * 3 * HIDDEN_DIM * INTERMEDIATE_DIM

# Synthetic weights follow the Muse Spark NVFP4 checkpoint (layers 1..60): block
# scales lie in [4, 448] (gate/up) and [3, 448] (down), no global scale is
# exactly representable in BF16, and 12,800 of 15,360 experts have different
# gate and up global scales, with up/gate in [0.2936, 7.082] (p90 2x, p99 4x;
# log2 of the ratio is fitted as a truncated normal). The `subnormal` profile
# centres the block scales in the E4M3 subnormal range instead.
SCALE_LOG2_MEAN = {"split": 7.0, "equal": 7.0, "subnormal": -6.0}
SCALE_LOG2_STD, GLOBAL_SCALE_LOG2_STD = 0.4, 0.25
GATE_UP_SCALE_MIN, DOWN_SCALE_MIN = 4.0, 3.0
SPLIT_W13_FRACTION = 12800 / 15360
UP_GATE_RATIO_RANGE = (0.2936, 7.082)
UP_GATE_LOG2_MEAN, UP_GATE_LOG2_STD = -0.315, 0.997

# Synthetic routing.
NORMAL_POPULARITY, ZIPF_ALPHA = 0.03, 1.05
OUTLIER_FRAC, OUTLIER_SCALE = 0.01, 30.0

# Captured routing archives: layers 0 and 61 are BF16, 1..60 NVFP4.
NUM_ROUTING_LAYERS = 62
DEFAULT_ROUTING_LAYERS = tuple(range(1, NUM_ROUTING_LAYERS - 1))
VERIFY_BATCH_SIZES = (1, 4, 16, 32)
VERIFY_WINDOW = 8
PREFILL_MIN_TOKENS = 1024
EP_RANK_MODES = ("busiest", "median")
_PREFILL_NAME = re.compile(
    r"^prefill_(?P<tag>.+)_c(?P<conc>\d+)burst_k(?P<chunk>\d+)\.npz$"
)
_PREFILL_CACHE_CHUNKS = 8

DECODE_SHAPES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1023]
DECODE_SHAPES += [1024, 1536, 2048]
PREFILL_SHAPES = [1024, 2048, 4096, 8192, 16384]
CANDIDATE, TWIN, MARLIN = "hopper_nvfp4", "hopper_nvfp4_twin", "marlin"
HummingGemmMode = Literal["grouped", "indexed"]
HUMMING_LAYOUTS: dict[str, HummingGemmMode] = {
    "humming_grouped": "grouped",
    "humming_indexed": "indexed",
}
HUMMING_ARMS = tuple(HUMMING_LAYOUTS)
ALL_ARMS = (CANDIDATE, TWIN, MARLIN, *HUMMING_ARMS)
# `MuseMoERunner` arms: the `moe_backend` each uses and the experts it must get.
RUNNER_ARMS: dict[str, tuple[MoEBackend, str]] = {
    CANDIDATE: ("auto", "MuseSparkHopperNvFp4Experts"),
    TWIN: ("auto", "MuseSparkHopperNvFp4Experts"),
    MARLIN: ("marlin", "MarlinExperts"),
}
FLUSH_ARM = "l2_flush"
NULL_TWIN_BAND = (0.99, 1.01)
TRIM = 0.05
_MB = 1024.0 * 1024.0


@dataclass
class StepInput:
    """One routing and activation sample for a single MoE call."""

    hidden_states: torch.Tensor  # [T, HIDDEN_DIM] bf16
    topk_ids: torch.Tensor  # [T, TOP_K] int32 global expert IDs
    topk_weights: torch.Tensor  # [T, TOP_K] float32, rows sum to 1
    local_routes: int
    active_local_experts: int
    routing: dict[str, Any]


@dataclass
class BenchmarkArm:
    """One benchmarked MoE backend."""

    name: str
    backend_class: str
    weight_mb: float
    forward_fn: Callable[[StepInput], torch.Tensor]
    workspace_mb_fn: Callable[[int], float]
    vllm_config: VllmConfig | None = None
    owner: Any = None


# ---------------------------------------------------------------------------
# Weights and FP32 reference
# ---------------------------------------------------------------------------


def _nvfp4_quant_config() -> ModelOptNvFp4Config:
    return ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=NVFP4_BLOCK,
    )


def _avoid_bf16_exact(x: torch.Tensor) -> torch.Tensor:
    """Nudge values that BF16 represents exactly up by one FP32 ULP.

    No global scale of the real checkpoint is BF16-exact, and the arms differ in
    whether they round global scales to BF16.
    """
    exact = x == x.to(torch.bfloat16).float()
    return torch.where(exact, torch.nextafter(x, torch.full_like(x, math.inf)), x)


def _sample_up_gate_ratio(
    num: int, gen: torch.Generator, device: torch.device
) -> torch.Tensor:
    """Draw up/gate global-scale ratios from the truncated log-normal fit."""
    lo, hi = (
        0.5
        * (
            1.0
            + math.erf((math.log2(r) - UP_GATE_LOG2_MEAN) / UP_GATE_LOG2_STD / 2**0.5)
        )
        for r in UP_GATE_RATIO_RANGE
    )
    u = torch.rand((num,), dtype=torch.float64, generator=gen, device=device)
    z = torch.special.ndtri(lo + (hi - lo) * u)
    ratio = torch.exp2(UP_GATE_LOG2_MEAN + UP_GATE_LOG2_STD * z)
    return ratio.clamp(*UP_GATE_RATIO_RANGE).float()


def _random_nvfp4_tensor(
    rows: int,
    k: int,
    log2_mean: float,
    scale_min: float,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random FP4 codes, E4M3 block scales and per-expert global scales.

    Block scales are log2-normal around `2**log2_mean`, clamped to
    `[scale_min, 448]`. The global scales keep `x @ w.T` near unit variance for
    unit-variance `x`, up to a per-expert log2-normal spread.
    """
    packed = torch.randint(
        0, 256, (E_LOCAL, rows, k // 2), dtype=torch.uint8, generator=gen, device=device
    )
    z = torch.randn((E_LOCAL, rows, k // NVFP4_BLOCK), generator=gen, device=device)
    sf = (
        torch.exp2(log2_mean + SCALE_LOG2_STD * z)
        .clamp(scale_min, E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    del z
    mean_sf_sq = 4.0**log2_mean * math.exp(2.0 * (math.log(2.0) * SCALE_LOG2_STD) ** 2)
    base = 1.0 / math.sqrt(E2M1_MEAN_SQUARE * mean_sf_sq * k)
    spread = torch.randn((E_LOCAL,), generator=gen, device=device)
    return (
        packed,
        sf,
        _avoid_bf16_exact(base * torch.exp2(GLOBAL_SCALE_LOG2_STD * spread)),
    )


def generate_checkpoint(
    seed: int, device: torch.device, scale_profile: str = "split"
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Generate one EP rank's synthetic NVFP4 experts and post-norm weight.

    The tensors have the names, shapes and dtypes that the Muse Spark experts
    load for ModelOpt NVFP4: `w13` holds the gate rows `[0, I)` then the up rows
    `[I, 2I)`, block scales are linear, and `w13_weight_scale_2[:, 0]` and
    `[:, 1]` are the gate and up global scales.

    Args:
        seed: RNG seed.
        device: Device for the tensors.
        scale_profile: `split` matches the checkpoint's scale statistics,
            `equal` gives every expert equal gate and up global scales, and
            `subnormal` is `split` with block scales centred in the E4M3
            subnormal range.

    Returns:
        The checkpoint tensors keyed by parameter name, and the zero-centred
        `[HIDDEN_DIM]` BF16 post-expert norm weight.

    """
    gen = torch.Generator(device=device).manual_seed(seed)
    log2_mean = SCALE_LOG2_MEAN[scale_profile]
    subnormal = scale_profile == "subnormal"
    w13_w, w13_sf, gate_s2 = _random_nvfp4_tensor(
        2 * INTERMEDIATE_DIM,
        HIDDEN_DIM,
        log2_mean,
        E4M3_MIN_SUBNORMAL if subnormal else GATE_UP_SCALE_MIN,
        gen,
        device,
    )
    w2_w, w2_sf, w2_s2 = _random_nvfp4_tensor(
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
        log2_mean,
        E4M3_MIN_SUBNORMAL if subnormal else DOWN_SCALE_MIN,
        gen,
        device,
    )
    up_s2 = gate_s2
    if scale_profile != "equal":
        split = torch.ones((E_LOCAL,), dtype=torch.bool, device=device)
        num_equal = round(E_LOCAL * (1.0 - SPLIT_W13_FRACTION))
        split[torch.randperm(E_LOCAL, generator=gen, device=device)[:num_equal]] = False
        ratio = _sample_up_gate_ratio(E_LOCAL, gen, device)
        up_s2 = torch.where(split, _avoid_bf16_exact(gate_s2 * ratio), gate_s2)
    post_norm_w = (0.1 * torch.randn((HIDDEN_DIM,), generator=gen, device=device)).to(
        torch.bfloat16
    )
    checkpoint = {
        "w13_weight": w13_w,
        "w13_weight_scale": w13_sf,
        "w13_weight_scale_2": torch.stack((gate_s2, up_s2), dim=1).contiguous(),
        "w13_input_scale": torch.ones((E_LOCAL, 2), device=device),
        "w2_weight": w2_w,
        "w2_weight_scale": w2_sf,
        "w2_weight_scale_2": w2_s2,
        "w2_input_scale": torch.ones((E_LOCAL,), device=device),
    }
    return checkpoint, post_norm_w


def scale_summary(checkpoint: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Block-scale ranges and gate/up global-scale split of one rank's experts."""
    s2 = checkpoint["w13_weight_scale_2"].float()
    split = s2[:, 1] != s2[:, 0]
    ratio = s2[:, 1] / s2[:, 0]
    summary: dict[str, Any] = {
        "split_w13_experts": int(split.sum()),
        "subnormal_block_scale_fraction": float(
            (checkpoint["w13_weight_scale"].view(torch.uint8) & 0x78)
            .eq(0)
            .float()
            .mean()
        ),
    }
    for name in ("w13_weight_scale", "w2_weight_scale"):
        sf = checkpoint[name].float()
        summary[f"{name}_range"] = [float(sf.min()), float(sf.max())]
    if bool(split.any()):
        summary["up_gate_ratio_range"] = [
            float(ratio[split].min()),
            float(ratio[split].max()),
        ]
    return summary


def _make_post_expert_norm(
    post_norm_w: torch.Tensor, device: torch.device, eps: float
) -> MuseSparkRMSNormWithInputScale:
    """The post-expert norm exactly as the Muse Spark MoE block builds it."""
    norm = MuseSparkRMSNormWithInputScale(
        hidden_size=HIDDEN_DIM,
        eps=eps,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
        output_norm_gain_center_type="zero",
    ).to(device=device, dtype=torch.bfloat16)
    norm.weight.copy_(post_norm_w)
    if (
        norm.fused_gain_center != 1.0
        or not norm.applies_weight_before_norm
        or norm.variance_size_override is not None
    ):
        raise RuntimeError("unexpected post-expert norm configuration")
    return norm


def fp32_reference_batch(
    inputs: Sequence[StepInput],
    checkpoint: dict[str, torch.Tensor],
    post_norm_w: torch.Tensor,
    eps: float,
) -> list[torch.Tensor]:
    """Unrounded FP32 MoE outputs of the local experts.

    For every route of token `t` to a local expert: `silu(x @ gate.T) * (x @
    up.T)`, then `@ down.T`, then the weight-first RMSNorm `v * rsqrt(mean(v**2)
    + eps)` with `v = y * (1 + w)`, accumulated with the FP32 top-k weight.
    Matmuls run at full FP32 precision (no TF32).
    """
    gain = 1.0 + post_norm_w.float()
    outs = [
        torch.zeros(
            (d.hidden_states.shape[0], HIDDEN_DIM),
            dtype=torch.float32,
            device=post_norm_w.device,
        )
        for d in inputs
    ]
    touched: set[int] = set()
    for d in inputs:
        touched.update(d.topk_ids[d.topk_ids < E_LOCAL].unique().tolist())

    def dequant(
        prefix: str, e: int, rows: slice, scale_2: torch.Tensor
    ) -> torch.Tensor:
        return dequantize_to_dtype(
            checkpoint[f"{prefix}_weight"][e, rows],
            checkpoint[f"{prefix}_weight_scale"][e, rows],
            scale_2,
            torch.float32,
            block_size=NVFP4_BLOCK,
            swizzle=False,
        )

    gate_rows = slice(0, INTERMEDIATE_DIM)
    up_rows = slice(INTERMEDIATE_DIM, 2 * INTERMEDIATE_DIM)
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        for e in sorted(touched):
            s13 = checkpoint["w13_weight_scale_2"][e]
            w_gate = dequant("w13", e, gate_rows, s13[0])
            w_up = dequant("w13", e, up_rows, s13[1])
            w_down = dequant("w2", e, slice(None), checkpoint["w2_weight_scale_2"][e])
            for out, d in zip(outs, inputs):
                tok, slot = (d.topk_ids == e).nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                x = d.hidden_states[tok].float()
                v = (F.silu(x @ w_gate.t()) * (x @ w_up.t())) @ w_down.t() * gain
                v = v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + eps)
                out.index_add_(0, tok, v * d.topk_weights[tok, slot, None])
    finally:
        torch.set_float32_matmul_precision(precision)
    return outs


# Expert tensors of one MoE layer in the Muse Spark ModelOpt NVFP4 checkpoint,
# keyed by the parameter they load into (see `OMNI_WEIGHTS_MAPPER`).
_HF_EXPERT_TENSORS = {
    "w13_weight": "gate_up_proj",
    "w13_weight_scale": "gate_up_proj_weight_scale",
    "w13_weight_scale_2": "gate_up_proj_weight_scale_2",
    "w13_input_scale": "gate_up_proj_input_scale",
    "w2_weight": "down_proj",
    "w2_weight_scale": "down_proj_weight_scale",
    "w2_weight_scale_2": "down_proj_weight_scale_2",
    "w2_input_scale": "down_proj_input_scale",
}
_HF_EXPERT_PREFIXES = (
    "model.language_model.layers.{layer}.mlp.experts.",
    "model.layers.{layer}.mlp.experts.",
)


def load_checkpoint_experts(
    checkpoint_dir: str, layer: int, ep_rank: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor, float, dict[str, Any]]:
    """Load one EP8 rank's experts of one NVFP4 MoE layer from a checkpoint.

    Reads global experts `[32 * ep_rank, 32 * ep_rank + 32)` of every expert
    tensor through `safetensors` slices; they become the benchmarked local
    experts, so the routing must be relabelled to the same rank.

    Args:
        checkpoint_dir: Directory with `config.json`,
            `model.safetensors.index.json` and the shards.
        layer: Decoder layer with NVFP4 experts (1..60 for Muse Spark).
        ep_rank: EP8 rank whose experts to load.
        device: Device for the tensors.

    Returns:
        The expert tensors keyed like `generate_checkpoint`, the post-expert
        norm weight, its `eps`, and metadata for the results file.

    Raises:
        ValueError: If the layer has no NVFP4 experts or a tensor has an
            unexpected shape.

    """
    from safetensors import safe_open

    with open(os.path.join(checkpoint_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    config = config.get("text_config", config)
    eps = float(config.get("post_norm_eps", POST_NORM_EPS))
    index = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index, encoding="utf-8") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    prefix = next(
        (
            p.format(layer=layer)
            for p in _HF_EXPERT_PREFIXES
            if p.format(layer=layer) + "gate_up_proj" in weight_map
        ),
        None,
    )
    if prefix is None:
        raise ValueError(f"layer {layer} has no NVFP4 experts in {index}")

    def read(name: str, rows: slice | None) -> torch.Tensor:
        path = os.path.join(checkpoint_dir, weight_map[name])
        with safe_open(path, framework="pt", device="cpu") as f:
            if rows is None:
                return f.get_tensor(name)
            return f.get_slice(name)[rows]

    lo = E_LOCAL * ep_rank
    experts = slice(lo, lo + E_LOCAL)
    two_i, d, i = 2 * INTERMEDIATE_DIM, HIDDEN_DIM, INTERMEDIATE_DIM
    specs = {
        "w13_weight": (torch.uint8, (E_LOCAL, two_i, d // 2)),
        "w13_weight_scale": (torch.float8_e4m3fn, (E_LOCAL, two_i, d // NVFP4_BLOCK)),
        "w13_weight_scale_2": (torch.float32, (E_LOCAL, 2)),
        "w2_weight": (torch.uint8, (E_LOCAL, d, i // 2)),
        "w2_weight_scale": (torch.float8_e4m3fn, (E_LOCAL, d, i // NVFP4_BLOCK)),
        "w2_weight_scale_2": (torch.float32, (E_LOCAL,)),
    }
    checkpoint: dict[str, torch.Tensor] = {}
    for param, suffix in _HF_EXPERT_TENSORS.items():
        tensor = read(prefix + suffix, experts)
        if param in specs:
            dtype, shape = specs[param]
            if tensor.dtype == torch.uint8 and dtype == torch.float8_e4m3fn:
                tensor = tensor.view(dtype)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{prefix + suffix}: {tensor.dtype} {list(tensor.shape)}, "
                    f"expected {dtype} {list(shape)}"
                )
        checkpoint[param] = tensor.to(device)
    post_norm_w = read(prefix + "post_expert_norm.weight", None)
    metadata = {
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "layer": layer,
        "ep_rank": ep_rank,
        "post_norm_eps": eps,
    }
    return checkpoint, post_norm_w.to(device, torch.bfloat16), eps, metadata


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _check_ep8_rank0(routed: MuseSparkRoutedExperts) -> None:
    """Raise unless `routed` holds EP rank 0's experts of an EP8 deployment."""
    pc = routed.moe_config.moe_parallel_config
    expected = torch.full((E_GLOBAL,), -1, dtype=torch.int32)
    expected[:E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32)
    expert_map = routed.expert_map
    if (
        (pc.tp_size, pc.ep_size, pc.ep_rank, pc.dp_size, pc.pcp_size)
        != (1, EP_SIZE, 0, 1, 1)
        or pc.use_all2all_kernels
        or pc.enable_eplb
        or routed.global_num_experts != E_GLOBAL
        or routed.local_num_experts != E_LOCAL
        or expert_map is None
        or not torch.equal(expert_map.cpu().to(torch.int32), expected)
    ):
        raise RuntimeError(f"{routed.layer_name}: not EP{EP_SIZE} rank 0: {pc}")


def _build_runner_arm(
    name: str,
    checkpoint: dict[str, torch.Tensor],
    post_norm_w: torch.Tensor,
    post_norm_eps: float,
    device: torch.device,
    allow_fallback: bool,
) -> BenchmarkArm:
    """Build a `MuseMoERunner` as EP rank 0 of an EP8 deployment on one GPU.

    `tp_size=EP_SIZE` with expert parallelism gives the per-rank configuration
    of a TP8 + EP8 server: MoE `tp_size=1`, `ep_size=8`, 32 local experts and an
    `expert_map` with `-1` for the other ranks' experts.
    """
    moe_backend, expected = RUNNER_ARMS[name]
    gc.collect()
    torch.accelerator.synchronize()
    mem_before = torch.accelerator.memory_allocated()
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            enable_expert_parallel=True, all2all_backend="allgather_reducescatter"
        ),
        kernel_config=KernelConfig(moe_backend=moe_backend),
    )
    expert_bias = torch.zeros((E_GLOBAL,), dtype=torch.float32, device=device)
    with set_current_vllm_config(vllm_config):
        router = MuseSparkRouter(
            top_k=TOP_K,
            num_experts=E_GLOBAL,
            expert_bias=expert_bias,
            route_scale=1.0,
            router_score_normalization="sigmoid",
            router_score_division_eps=1e-15,
        )
        runner = FusedMoEFactory(
            num_experts=E_GLOBAL,
            top_k=TOP_K,
            hidden_size=HIDDEN_DIM,
            intermediate_size=INTERMEDIATE_DIM,
            renormalize=True,
            router=router,
            quant_config=_nvfp4_quant_config(),
            tp_size=EP_SIZE,
            dp_size=1,
            pcp_size=1,
            prefix=f"bench.muse_spark.{name}",
            params_dtype=torch.bfloat16,
            scoring_func="sigmoid",
            e_score_correction_bias=expert_bias,
            routed_scaling_factor=1.0,
            routed_experts_cls=MuseSparkRoutedExperts,
            runner_cls=MuseMoERunner,
            runner_args={
                "post_expert_norm": _make_post_expert_norm(
                    post_norm_w, device, post_norm_eps
                ),
                "fuse_all_reduce": False,
            },
        )
        runner.to(device)
        routed = runner.routed_experts
        _check_ep8_rank0(routed)
        loaded = list(
            routed.load_weights((k, v.clone()) for k, v in checkpoint.items())
        )
        if len(loaded) != len(checkpoint):
            raise RuntimeError(f"{name}: loaded {len(loaded)} of {len(checkpoint)}")
        routed.quant_method.process_weights_after_loading(routed)
        runner.process_weights_after_loading()
    gc.collect()
    torch.accelerator.synchronize()
    weight_mb = (torch.accelerator.memory_allocated() - mem_before) / _MB

    moe_kernel = routed.quant_method.moe_kernel
    if moe_kernel is None:
        raise RuntimeError(f"{name}: no MoE kernel after weight loading")
    experts = moe_kernel.fused_experts
    finalize = runner.post_expert_finalize
    if experts.post_gemm2_hook is not finalize or finalize.pre_norm_all_reduce:
        raise RuntimeError(f"{name}: the post-expert finalize is not attached")
    backend_class = type(experts).__name__
    if backend_class != expected:
        message = f"{name}: moe_backend={moe_backend!r} resolved to {backend_class}"
        if not allow_fallback:
            raise RuntimeError(f"{message}, expected {expected}")
        print(f"warning: {message}", flush=True)

    _, _, size_n, size_k, top_k = experts.moe_problem_size(
        torch.empty((1, HIDDEN_DIM), dtype=torch.bfloat16, device="meta"),
        routed.w13_weight,
        routed.w2_weight,
        torch.empty((1, TOP_K), dtype=torch.int32, device="meta"),
    )
    itemsize = experts.workspace_dtype(torch.bfloat16).itemsize

    def forward(d: StepInput) -> torch.Tensor:
        out = routed.forward_modular(
            x=d.hidden_states, topk_weights=d.topk_weights, topk_ids=d.topk_ids
        )
        assert isinstance(out, torch.Tensor)
        return out

    def workspace_mb(num_tokens: int) -> float:
        # As reserved by the modular kernel: workspace13 shares a buffer with
        # the fused output, plus workspace2.
        ws13, ws2, out_shape = experts.workspace_shapes(
            num_tokens,
            size_n,
            size_k,
            top_k,
            E_GLOBAL,
            routed.w13_weight.shape[0],
            None,
            MoEActivation.SILU,
        )
        elems = max(1, math.prod(ws13), math.prod(out_shape)) + math.prod(ws2)
        return elems * itemsize / _MB

    return BenchmarkArm(
        name, backend_class, weight_mb, forward, workspace_mb, vllm_config, runner
    )


def _make_ep8_moe_config(device: torch.device) -> FusedMoEConfig:
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=EP_SIZE,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=E_GLOBAL,
        experts_per_token=TOP_K,
        hidden_dim=HIDDEN_DIM,
        intermediate_size=INTERMEDIATE_DIM,
        num_local_experts=E_LOCAL,
        num_logical_experts=E_GLOBAL,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        moe_backend="humming",
        max_num_tokens=max(PREFILL_SHAPES),
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.MiniMax2,
    )


# Floor for the stream-K lock scratch. `create_humming_locks` allocates
# `8 * num_sms` int32 entries, only 1056 on a 132-SM H200.
_MIN_HUMMING_LOCKS = 2048


@dataclass
class HummingMoEWorkspace:
    """Buffers preallocated for steps of up to `max_tokens` tokens.

    `gemm_in` holds the inputs of both Humming GEMMs (the scattered W13 input,
    then the SwiGLU output) and `gemm_out` their outputs (the W13 output, then
    the W2 output), so no GEMM reads and writes the same buffer, as with the
    workspace13 and workspace2 split of `HummingExpertsBase.prepare_buffers`.
    Arms that share a workspace must run one at a time on one stream.

    Attributes:
        max_tokens: Largest token count the buffers hold.
        gemm_in: GEMM inputs, `max_tokens * TOP_K * max(HIDDEN_DIM,
            INTERMEDIATE_DIM)` bf16 elements.
        gemm_out: GEMM outputs, `max_tokens * TOP_K * max(2 *
            INTERMEDIATE_DIM, HIDDEN_DIM)` bf16 elements.
        local_route: Whether each route is local, `[max_tokens * TOP_K]` bool.
        permuted_idx: Sorted GEMM2 row of each route, `-1` for remote routes,
            `[max_tokens * TOP_K]` int32.
        neg_one: The `-1` sentinel, a 0-dim int32 tensor.

    """

    max_tokens: int
    gemm_in: torch.Tensor
    gemm_out: torch.Tensor
    local_route: torch.Tensor
    permuted_idx: torch.Tensor
    neg_one: torch.Tensor

    @classmethod
    def allocate(cls, max_tokens: int, device: torch.device) -> HummingMoEWorkspace:
        """Allocate the buffers for up to `max_tokens` tokens on `device`.

        Raises:
            ValueError: If `max_tokens` is not positive.

        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        rows = max_tokens * TOP_K
        in_elems = rows * max(HIDDEN_DIM, INTERMEDIATE_DIM)
        out_elems = rows * max(2 * INTERMEDIATE_DIM, HIDDEN_DIM)
        return cls(
            max_tokens=max_tokens,
            gemm_in=torch.empty(in_elems, dtype=torch.bfloat16, device=device),
            gemm_out=torch.empty(out_elems, dtype=torch.bfloat16, device=device),
            local_route=torch.empty(rows, dtype=torch.bool, device=device),
            permuted_idx=torch.empty(rows, dtype=torch.int32, device=device),
            neg_one=torch.full((), -1, dtype=torch.int32, device=device),
        )


def fold_split_w13_scale_2(
    w13_weight_scale_2: torch.Tensor, w2_weight_scale_2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold split w1/w3 global scales into one W13 scale, in FP32.

    The fold of `MuseSparkModelOptNvFp4FusedMoE._process_hopper_marlin_weights`:
    W13 keeps the gate (w1) scale on both halves, so the SiLU input stays
    exact. That scales the up projection, and with it the SwiGLU output, by
    `gate / up`; W2 is linear in that output, so multiplying its global scale
    by `up / gate` restores the result.

    Args:
        w13_weight_scale_2: Per-expert `[gate, up]` global scales, `[E, 2]`.
        w2_weight_scale_2: Per-expert W2 global scales, `[E]`.

    Returns:
        The W13 and W2 global scales, each `[E]` float32.

    Raises:
        ValueError: If the shapes disagree or a gate scale is not positive.

    """
    if w13_weight_scale_2.dim() != 2 or w13_weight_scale_2.size(1) != 2:
        raise ValueError(
            f"w13_weight_scale_2 must be [E, 2], got {tuple(w13_weight_scale_2.shape)}"
        )
    gate = w13_weight_scale_2[:, 0].float()
    up = w13_weight_scale_2[:, 1].float()
    w2 = w2_weight_scale_2.float().reshape(-1)
    if w2.numel() != gate.numel():
        raise ValueError(
            f"w2_weight_scale_2 has {w2.numel()} scales for {gate.numel()} experts"
        )
    if not bool((gate > 0).all()):
        raise ValueError("w13 gate global scales must be positive")
    return gate.contiguous(), (w2 * (up / gate)).contiguous()


class StandaloneHummingMuseMoE:
    """Humming NVFP4 W13 + SwiGLU + W2, then `MusePostExpertFinalize`.

    Issues the calls of vLLM's `HummingGroupedExperts.apply` (`"grouped"`) or
    `HummingIndexedExperts.apply` (`"indexed"`) for the Muse Spark EP8 rank-0
    shard, then hands the unweighted GEMM2 rows to `MusePostExpertFinalize`
    in place of the experts' own top-k reduction. Weights go through vLLM's
    Humming conversion after the split-scale fold of the Muse Spark Marlin
    path, so both GEMMs use FP32 per-expert global scales.

    Args:
        gemm_mode: `"grouped"` for `GemmType.GROUPED_CONTIGUOUS`, or
            `"indexed"` for `GemmType.INDEXED`.
        checkpoint: Local-shard ModelOpt NVFP4 tensors from
            `generate_checkpoint`.
        post_norm_w: Post-expert RMSNorm weight, `[HIDDEN_DIM]`.
        post_norm_eps: Post-expert RMSNorm epsilon.
        device: CUDA device for the weights.
        workspace: Preallocated buffers; they bound the token count.

    """

    def __init__(
        self,
        gemm_mode: HummingGemmMode,
        checkpoint: dict[str, torch.Tensor],
        post_norm_w: torch.Tensor,
        post_norm_eps: float,
        device: torch.device,
        workspace: HummingMoEWorkspace,
    ) -> None:
        from vllm.utils.humming import (
            BaseWeightSchema,
            GemmType,
            HummingInputSchema,
            WeightScale2Type,
            get_heuristics_config,
            humming_forward,
            may_process_input,
            prepare_and_transform_humming_tensors,
        )

        if gemm_mode not in ("grouped", "indexed"):
            raise ValueError(f"Unknown Humming gemm mode: {gemm_mode!r}")
        self.gemm_mode = gemm_mode
        self.gemm_type = (
            GemmType.GROUPED_CONTIGUOUS if gemm_mode == "grouped" else GemmType.INDEXED
        )
        self.workspace = workspace
        self._humming_forward = humming_forward
        self._may_process_input = may_process_input

        self.moe_config = _make_ep8_moe_config(device)
        self.norm = _make_post_expert_norm(post_norm_w, device, post_norm_eps)
        self.post_expert_finalize = MusePostExpertFinalize(
            self.norm, self.moe_config, fuse_all_reduce=False
        )
        # EP rank 0 owns global experts [0, E_LOCAL).
        self.expert_map = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=device)
        self.expert_map[:E_LOCAL] = torch.arange(
            E_LOCAL, dtype=torch.int32, device=device
        )
        # Muse Spark has no SwiGLU clamp, so the default config is what
        # `ApplyMoEActivationConfig.from_configs` yields for it.
        self.activation_kwargs = get_humming_activation(MoEActivation.SILU)

        w13_scale_2, w2_scale_2 = fold_split_w13_scale_2(
            checkpoint["w13_weight_scale_2"], checkpoint["w2_weight_scale_2"]
        )
        # (shape_n, shape_k, N stacks, tensors) per sublayer; a gated W13 is
        # two N stacks, as in `_convert_sublayer_to_humming`.
        sublayers = {
            "w13": (
                2 * INTERMEDIATE_DIM,
                HIDDEN_DIM,
                [INTERMEDIATE_DIM, INTERMEDIATE_DIM],
                {
                    "weight": checkpoint["w13_weight"].clone(),
                    "weight_scale": checkpoint["w13_weight_scale"].clone(),
                    "weight_scale_2": w13_scale_2,
                },
            ),
            "w2": (
                HIDDEN_DIM,
                INTERMEDIATE_DIM,
                [HIDDEN_DIM],
                {
                    "weight": checkpoint["w2_weight"].clone(),
                    "weight_scale": checkpoint["w2_weight_scale"].clone(),
                    "weight_scale_2": w2_scale_2,
                },
            ),
        }
        checkpoint_schema = BaseWeightSchema.from_config(
            {"quant_method": "modelopt", "quant_algo": "nvfp4"}
        )
        self.configs: dict[str, Any] = {}
        self.tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.tunings: dict[str, list[Any]] = {}
        self.tuning_strs: dict[str, str] = {}
        self.hadamard_block_sizes: dict[str, int | None] = {}
        for name, (shape_n, shape_k, n_stacks, tensors) in sublayers.items():
            # `_process_single_sublayer` with the default (BF16) input schema.
            weight_schema, converted = checkpoint_schema.convert_humming(
                tensors=tensors,
                shape_n_stacks=n_stacks,
                shape_k_stacks=[shape_k],
                param_dtype=torch.bfloat16,
                num_experts=E_LOCAL,
            )
            if weight_schema.weight_scale_2_type != WeightScale2Type.TENSOR:
                raise RuntimeError(
                    f"{name}: expected one FP32 global scale per expert, got "
                    f"{weight_schema.weight_scale_2_type}"
                )
            input_schema, input_tensors = HummingInputSchema().convert_humming(
                tensors=tensors,
                shape_n_stacks=n_stacks,
                shape_k_stacks=[shape_k],
                param_dtype=torch.bfloat16,
                num_experts=E_LOCAL,
            )
            converted.update(input_tensors)
            input_schema = check_and_fallback_input_schema(
                weight_schema=weight_schema,
                input_schema=input_schema,
                param_dtype=torch.bfloat16,
                allow_fp8_sm100=False,
            )
            config, transformed = prepare_and_transform_humming_tensors(
                tensors=converted,
                shape_n=shape_n,
                shape_k=shape_k,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema,
                weight_schema=weight_schema,
                has_bias=False,
                num_experts=E_LOCAL,
                torch_dtype=torch.bfloat16,
                device=device,
            )
            if config.input_quant_mode.should_quantize:
                raise RuntimeError(f"{name}: expected BF16 activations (W4A16)")
            tuning = get_heuristics_config(
                layer_config=config,
                use_f16_accum=False,
                use_batch_invariant=False,
                gemm_type=self.gemm_type,
            )
            self.configs[name] = config
            self.tensors[name] = transformed
            self.tunings[name] = tuning
            self.tuning_strs[name] = json.dumps(tuning)
            self.hadamard_block_sizes[name] = weight_schema.hadamard_block_size

        self.compute_config_str = json.dumps(
            {
                "use_batch_invariant": False,
                "use_f16_accum": False,
                "gemm_type": self.gemm_type.value,
            }
        )
        device_index = (
            device.index
            if device.index is not None
            else torch.accelerator.current_device_index()
        )
        self.num_sms = num_compute_units(device_index)
        self.locks = torch.zeros(
            max(_MIN_HUMMING_LOCKS, self.num_sms * 8),
            dtype=torch.int32,
            device=device,
        )
        self.scratch: MoEPermuteScratch | None = None
        if gemm_mode == "grouped":
            self.scratch = get_moe_permute_scratch(
                num_experts=E_GLOBAL, num_local_experts=E_LOCAL, device=device
            )
            self.scratch.reserve(workspace.max_tokens * TOP_K)

    @property
    def backend_label(self) -> str:
        return f"StandaloneHumming{self.gemm_mode.capitalize()}NvFp4"

    def workspace_mb(self, num_tokens: int) -> float:
        """Workspace vLLM's Humming experts reserve, counted like Marlin's.

        Args:
            num_tokens: Tokens in the step.

        Returns:
            workspace13 + workspace2 + output, in MiB.

        """
        rows = num_tokens * TOP_K
        elems = (
            rows * max(HIDDEN_DIM, INTERMEDIATE_DIM)
            + rows * max(2 * INTERMEDIATE_DIM, HIDDEN_DIM)
            + num_tokens * HIDDEN_DIM
        )
        return elems * torch.bfloat16.itemsize / _MB

    def forward(self, d: StepInput) -> torch.Tensor:
        """Run one MoE step for `d`.

        Enqueues device work only, without a host sync. Apart from the returned
        tensor, it allocates nothing of its own: the small buffers that
        `moe_align_block_size`, `route_major_moe_output` and Humming's launcher
        allocate are allocated the same way in vLLM.

        Args:
            d: Routing and activations; `T` must not exceed `max_tokens`.

        Returns:
            The finalized MoE output, `[T, HIDDEN_DIM]` bf16.

        Raises:
            ValueError: If `T` exceeds the workspace's `max_tokens`.

        """
        x, topk_ids, topk_weights = d.hidden_states, d.topk_ids, d.topk_weights
        num_tokens = x.size(0)
        if num_tokens > self.workspace.max_tokens:
            raise ValueError(
                f"{num_tokens} tokens exceed the Humming workspace "
                f"({self.workspace.max_tokens})"
            )
        # A new tensor per call: the harness keeps the output of every routing
        # captured in one CUDA graph.
        output = torch.empty((num_tokens, HIDDEN_DIM), dtype=x.dtype, device=x.device)
        if num_tokens == 0:
            return output
        # `HummingExpertsBase.estimate_local_valid_shape_m` without DP.
        valid_shape_m = math.ceil(num_tokens * TOP_K * E_LOCAL / E_GLOBAL)
        if self.gemm_mode == "grouped":
            unfinalized = self._forward_grouped(
                x, topk_ids, topk_weights, valid_shape_m
            )
        else:
            unfinalized = self._forward_indexed(
                x, topk_ids, topk_weights, valid_shape_m
            )
        self.post_expert_finalize(unfinalized, output)
        return output

    def _forward_grouped(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        valid_shape_m: int,
    ) -> UnfinalizedMoEOutput:
        """The calls of `HummingGroupedExperts.apply` up to GEMM2."""
        assert self.scratch is not None
        num_tokens = x.size(0)
        rows = num_tokens * TOP_K
        ws = self.workspace
        w13_in = ws.gemm_in[: rows * HIDDEN_DIM].view(rows, HIDDEN_DIM)
        gate_up = ws.gemm_out[: rows * 2 * INTERMEDIATE_DIM].view(
            rows, 2 * INTERMEDIATE_DIM
        )
        down_in = ws.gemm_in[: rows * INTERMEDIATE_DIM].view(rows, INTERMEDIATE_DIM)
        down_out = ws.gemm_out[: rows * HIDDEN_DIM].view(rows, HIDDEN_DIM)

        expert_offsets, scatter_idx = moe_prepare_scatter(
            topk_ids, self.expert_map, self.scratch
        )
        # Sorted rows from expert_offsets[-1] on hold remote or invalid routes.
        num_valid_tokens = expert_offsets[-1:]
        inputs = self._process(
            "w13",
            x,
            w13_in,
            scatter_idx=scatter_idx,
            num_valid_tokens=num_valid_tokens,
        )
        self._gemm(
            "w13",
            inputs,
            gate_up,
            valid_shape_m=valid_shape_m,
            expert_layout=expert_offsets,
        )
        inputs = self._process(
            "w2",
            gate_up,
            down_in,
            num_valid_tokens=expert_offsets[-1:],
            activation=True,
        )
        self._gemm(
            "w2",
            inputs,
            down_out,
            valid_shape_m=valid_shape_m,
            expert_layout=expert_offsets,
        )

        # `moe_prepare_scatter` sends remote and invalid routes to the sorted
        # rows at or past `expert_offsets[-1]`, never to `-1`, but
        # `MusePostExpertFinalize` skips only `-1`: mask in place on the GPU.
        local_route = ws.local_route[:rows].view(num_tokens, TOP_K)
        permuted_idx = ws.permuted_idx[:rows].view(num_tokens, TOP_K)
        torch.lt(scatter_idx, num_valid_tokens, out=local_route)
        torch.where(local_route, scatter_idx, ws.neg_one, out=permuted_idx)
        return UnfinalizedMoEOutput(
            gemm2_permuted=down_out,
            expert_weights=topk_weights,
            expanded_idx_to_permuted_idx=permuted_idx,
        )

    def _forward_indexed(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        valid_shape_m: int,
    ) -> UnfinalizedMoEOutput:
        """The calls of `HummingIndexedExperts.apply` up to GEMM2."""
        num_tokens = x.size(0)
        rows = num_tokens * TOP_K
        ws = self.workspace
        gate_up = ws.gemm_out[: rows * 2 * INTERMEDIATE_DIM].view(
            rows, 2 * INTERMEDIATE_DIM
        )
        down_in = ws.gemm_in[: rows * INTERMEDIATE_DIM].view(rows, INTERMEDIATE_DIM)
        down_out = ws.gemm_out[: rows * HIDDEN_DIM].view(rows, HIDDEN_DIM)

        # `HummingIndexedExperts.prepare_humming_moe_kwargs`; `expert_map` is
        # always set here, so the scatter heuristic always applies.
        block_m = self._block_m("w13", valid_shape_m)
        use_scatter = topk_ids.numel() > 2 * self.num_sms
        alignment = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_m,
            num_experts=E_GLOBAL,
            expert_map=self.expert_map,
            ignore_invalid_experts=True,
            return_scatter_idx=use_scatter,
        )
        sorted_ids, expert_ids, num_tokens_padded = alignment[:3]
        scatter_idx = alignment[3] if len(alignment) == 4 else None
        common = {
            "sorted_ids": sorted_ids,
            "expert_ids": expert_ids,
            "num_tokens_padded": num_tokens_padded,
            "valid_shape_m": valid_shape_m,
        }
        w13_kwargs: dict[str, Any] = {**common, "top_k": TOP_K}
        w2_kwargs: dict[str, Any] = {**common, "top_k": 1}
        w2_block_m = self._block_m("w2", valid_shape_m)
        if w2_block_m != block_m:
            sorted_ids2, expert_ids2, num_tokens_padded2 = moe_align_block_size(
                topk_ids=topk_ids,
                block_size=w2_block_m,
                num_experts=E_GLOBAL,
                expert_map=self.expert_map,
                ignore_invalid_experts=True,
            )
            w2_kwargs.update(
                sorted_ids=sorted_ids2,
                expert_ids=expert_ids2,
                num_tokens_padded=num_tokens_padded2,
            )

        # BF16 activations need no W13 input processing, so this returns `x`.
        inputs = self._process("w13", x, None)
        self._gemm("w13", inputs, gate_up, **w13_kwargs)
        inputs = self._process(
            "w2", gate_up, down_in, scatter_idx=scatter_idx, activation=True
        )
        self._gemm("w2", inputs, down_out, **w2_kwargs)
        # One GEMM2 row per route, handed over as `MarlinExperts` hands its rows
        # to `post_gemm2_hook`.
        return route_major_moe_output(
            down_out.view(num_tokens, TOP_K, HIDDEN_DIM),
            topk_weights,
            topk_ids,
            self.expert_map,
        )

    def _block_m(self, name: str, valid_shape_m: int) -> int:
        """Block M of the tuning entry vLLM picks for `valid_shape_m`."""
        for lower, upper, config in self.tunings[name]:
            if lower < valid_shape_m <= upper:
                return int(config["block_shape"][0])
        return 64

    def _process(
        self,
        name: str,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None,
        *,
        scatter_idx: torch.Tensor | None = None,
        num_valid_tokens: torch.Tensor | None = None,
        activation: bool = False,
    ) -> torch.Tensor:
        """`HummingExpertsBase.process_input` for unquantized activations."""
        config = self.configs[name]
        hadamard_block_size = self.hadamard_block_sizes[name]
        has_input_processing = (
            config.input_quant_mode.should_quantize
            or (hadamard_block_size is not None and hadamard_block_size > 1)
            or activation
            or scatter_idx is not None
            or num_valid_tokens is not None
        )
        processed, _, _ = self._may_process_input(
            config,
            inputs=inputs,
            outputs=outputs if has_input_processing else None,
            token_scales=None,
            hadamard_block_size=hadamard_block_size,
            layout="scatter" if scatter_idx is not None else "normal",
            expert_tokens=None,
            scatter_idx=scatter_idx,
            num_valid_tokens=num_valid_tokens,
            **(self.activation_kwargs if activation else {}),
        )
        return processed

    def _gemm(
        self, name: str, inputs: torch.Tensor, outputs: torch.Tensor, **kwargs: Any
    ) -> None:
        """`HummingExpertsBase.humming_forward` on this arm's tensors."""
        tensors = self.tensors[name]
        self._humming_forward(
            self.configs[name],
            inputs=inputs,
            weight=tensors["weight"],
            weight_scale=tensors.get("weight_scale"),
            zero_point=tensors.get("zero_point"),
            bias=tensors.get("bias"),
            weight_scale_2=tensors.get("weight_scale_2"),
            input_scale=None,
            input_scale_2=None,
            outputs=outputs,
            locks=self.locks,
            compute_config=self.compute_config_str,
            tuning_config=self.tuning_strs[name],
            **kwargs,
        )


def _build_humming_arm(
    arm_name: str,
    checkpoint: dict[str, torch.Tensor],
    post_norm_w: torch.Tensor,
    post_norm_eps: float,
    device: torch.device,
    workspace: HummingMoEWorkspace,
) -> tuple[BenchmarkArm | None, str | None]:
    """Build one Humming arm, or return why it cannot run here.

    Only a missing `humming` package or missing permute ops (grouped layout)
    skip the arm; any other setup error raises, so a broken baseline cannot
    silently drop out of the comparison.
    """
    gemm_mode = HUMMING_LAYOUTS[arm_name]
    if not has_humming():
        return None, "the humming package is not installed"
    if gemm_mode == "grouped" and not moe_permute_unpermute_supported():
        return None, "grouped Humming needs the moe_permute_unpermute ops"
    gc.collect()
    torch.accelerator.synchronize()
    mem_before = torch.accelerator.memory_allocated()
    arm = StandaloneHummingMuseMoE(
        gemm_mode, checkpoint, post_norm_w, post_norm_eps, device, workspace
    )
    gc.collect()
    torch.accelerator.synchronize()
    weight_mb = (torch.accelerator.memory_allocated() - mem_before) / _MB
    return (
        BenchmarkArm(
            name=arm_name,
            backend_class=arm.backend_label,
            weight_mb=weight_mb,
            forward_fn=arm.forward,
            workspace_mb_fn=arm.workspace_mb,
            owner=arm,
        ),
        None,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def select_ep_rank(route_counts: np.ndarray, ep_rank: str | int) -> int:
    """Pick the EP rank to emulate: `busiest`, `median` (4th of 8) or fixed."""
    if not isinstance(ep_rank, str):
        return int(ep_rank)
    order = np.argsort(-np.asarray(route_counts, dtype=np.int64), kind="stable")
    return int(order[0] if ep_rank == "busiest" else order[(EP_SIZE - 1) // 2])


def rotate_ids_to_rank(ids: Any, rank: int) -> Any:
    """Relabel global expert IDs so rank `rank`'s experts become `[0, E_LOCAL)`."""
    return (ids - E_LOCAL * rank) % E_GLOBAL


def _check_ids(ids: np.ndarray, path: str) -> None:
    if ids.ndim != 3 or ids.shape[0] != NUM_ROUTING_LAYERS or ids.shape[2] != TOP_K:
        raise ValueError(f"{path}: expected ids [62, T, 8], got {list(ids.shape)}")
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= E_GLOBAL):
        raise ValueError(f"{path}: expert ids outside [0, {E_GLOBAL})")


def _meta_sha16(path: str) -> str | None:
    with suppress(OSError, ValueError), open(path + ".meta.json") as f:
        return json.load(f).get("sha16")
    return None


@dataclass(frozen=True)
class _OffsetArchive:
    """A verify or extend archive: `[62, tokens, 8]` IDs plus forward offsets."""

    path: str
    offsets: np.ndarray
    ids: np.ndarray

    @property
    def num_forwards(self) -> int:
        return len(self.offsets) - 1

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def rows(self, layer: int, forward: int) -> np.ndarray:
        return self.ids[layer, self.offsets[forward] : self.offsets[forward + 1]]


def _load_offset_archive(path: str) -> _OffsetArchive:
    with np.load(path, allow_pickle=False) as z:
        offsets = z["offsets"].astype(np.int64)
        ids = np.stack([z[f"L{layer}"] for layer in range(NUM_ROUTING_LAYERS)])
    _check_ids(ids, path)
    if offsets[0] != 0 or offsets[-1] != ids.shape[1] or (np.diff(offsets) < 0).any():
        raise ValueError(f"{path}: offsets do not partition {ids.shape[1]} tokens")
    return _OffsetArchive(path, offsets, ids)


@dataclass(frozen=True)
class _PrefillChunk:
    """One prefill chunk archive and its place in its prompt stream."""

    path: str
    num_tokens: int
    tokens_to_stream_end: int


def _npz_shape(path: str, key: str) -> tuple[int, ...]:
    """Read an array's shape from an `.npz` header without loading it."""
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as fp:
        version = np.lib.format.read_magic(fp)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(fp)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(fp)
    return tuple(shape)


def _index_prefill_chunks(prefill_dir: str) -> list[_PrefillChunk]:
    """Index `prefill_<tag>_c<C>burst_k<NN>.npz`; consecutive `NN` form a stream."""
    if not os.path.isdir(prefill_dir):
        return []
    found = []
    for name in os.listdir(prefill_dir):
        m = _PREFILL_NAME.match(name)
        if m is not None:
            key = (m["tag"], int(m["conc"]), int(m["chunk"]))
            found.append((key, os.path.join(prefill_dir, name)))
    found.sort()
    streams: list[list[tuple[str, int]]] = []
    prev = None
    for (tag, conc, chunk), path in found:
        if prev != (tag, conc, chunk - 1):
            streams.append([])
        shape = _npz_shape(path, "ids")
        if len(shape) != 3 or shape[0] != NUM_ROUTING_LAYERS or shape[2] != TOP_K:
            raise ValueError(f"{path}: unexpected ids shape {list(shape)}")
        streams[-1].append((path, int(shape[1])))
        prev = (tag, conc, chunk)
    chunks = []
    for stream in streams:
        remaining = sum(n for _, n in stream)
        for path, n in stream:
            chunks.append(_PrefillChunk(path, n, remaining))
            remaining -= n
    return chunks


class RoutingProvider:
    """Captured routing traces relabelled to one EP rank, or synthetic routing.

    A captured sample draws its forward (or first prefill chunk) and then its
    layer from `np.random.RandomState(seed)`, takes `T` tokens, counts routes on
    each contiguous 32-expert rank, picks a rank with `ep_rank` and relabels the
    IDs so that rank is local. Traces hold no routing weights, so samples get
    random positive normalized weights. The `T` tokens are:

    - verify (`T <= 256` with `auto`): the first `T` tokens of a forward of the
      smallest `verify_B{B}` with `8 * B >= T`; for `T < 8`, the first token of
      `T` requests (one decode step of `T` requests).
    - extend (`T < 1024`): the first `T` tokens of a forward that long, or
      distinct forwards concatenated.
    - prefill: the first `T` tokens of a prompt stream with at least `T` left.
    """

    def __init__(
        self,
        routing_dir: str | None,
        routing_kind: str,
        synthetic_dist: str,
        device: torch.device,
        *,
        ep_rank: str | int = "busiest",
        layers: Sequence[int] | None = None,
        prefill_dir: str | None = None,
    ) -> None:
        self.routing_dir = routing_dir
        self.routing_kind = routing_kind
        self.synthetic_dist = synthetic_dist
        self.device = device
        self.ep_rank = ep_rank
        self.layers = tuple(layers or DEFAULT_ROUTING_LAYERS)
        if any(not 0 <= x < NUM_ROUTING_LAYERS for x in self.layers):
            raise ValueError(f"routing layers must be in [0, 62): {self.layers}")
        self._verify_paths: dict[int, str] = {}
        self._extend_path: str | None = None
        self.prefill_dir = prefill_dir
        if routing_dir is not None:
            for b in VERIFY_BATCH_SIZES:
                path = os.path.join(routing_dir, f"verify_B{b}_real.npz")
                if os.path.isfile(path):
                    self._verify_paths[b] = path
            path = os.path.join(routing_dir, "extend_real.npz")
            self._extend_path = path if os.path.isfile(path) else None
            self.prefill_dir = prefill_dir or os.path.join(routing_dir, "prefill")
        self._archives: dict[str, _OffsetArchive] = {}
        self._prefill_chunks: list[_PrefillChunk] | None = None
        self._prefill_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._used_paths: set[str] = set()

    def _archive(self, path: str) -> _OffsetArchive:
        if path not in self._archives:
            self._archives[path] = _load_offset_archive(path)
            self._used_paths.add(path)
        return self._archives[path]

    def _chunks(self) -> list[_PrefillChunk]:
        if self._prefill_chunks is None:
            self._prefill_chunks = (
                _index_prefill_chunks(self.prefill_dir) if self.prefill_dir else []
            )
        return self._prefill_chunks

    def _prefill_ids(self, path: str) -> np.ndarray:
        ids = self._prefill_cache.pop(path, None)
        if ids is None:
            with np.load(path, allow_pickle=False) as z:
                ids = z["ids"]
            _check_ids(ids, path)
            self._used_paths.add(path)
        self._prefill_cache[path] = ids
        while len(self._prefill_cache) > _PREFILL_CACHE_CHUNKS:
            self._prefill_cache.popitem(last=False)
        return ids

    def _draw_layer(self, rs: np.random.RandomState) -> int:
        return self.layers[int(rs.randint(len(self.layers)))]

    def resolve_kind(self, num_tokens: int) -> str:
        """Archive kind used for `num_tokens` tokens."""
        if self.routing_kind != "auto":
            return self.routing_kind
        if num_tokens <= VERIFY_WINDOW * max(self._verify_paths, default=0):
            return "verify"
        available = {
            "verify": bool(self._verify_paths),
            "extend": self._extend_path is not None,
            "prefill": bool(self._chunks()),
        }
        preference = ("extend", "prefill", "verify")
        if num_tokens >= PREFILL_MIN_TOKENS:
            preference = ("prefill", "extend", "verify")
        for kind in preference:
            if available[kind]:
                return kind
        raise FileNotFoundError(f"No routing archives in {self.routing_dir}")

    def _sample_verify(
        self, num_tokens: int, rs: np.random.RandomState
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._verify_paths:
            raise FileNotFoundError(f"No verify_B*_real.npz in {self.routing_dir}")
        one_per_request = num_tokens < VERIFY_WINDOW
        requests = num_tokens if one_per_request else -(-num_tokens // VERIFY_WINDOW)
        batch = min((b for b in self._verify_paths if b >= requests), default=None)
        if batch is not None:
            arch = self._archive(self._verify_paths[batch])
            forwards = [int(rs.randint(arch.num_forwards))]
            layer = self._draw_layer(rs)
            rows = arch.rows(layer, forwards[0])
            step = VERIFY_WINDOW if one_per_request else 1
            ids = rows[: num_tokens * step : step]
        else:
            arch = self._archive(self._verify_paths[max(self._verify_paths)])
            count = -(-num_tokens // int(arch.lengths.min()))
            if count > arch.num_forwards:
                raise ValueError(f"{arch.path}: too few forwards for T={num_tokens}")
            forwards = [int(f) for f in rs.choice(arch.num_forwards, count, False)]
            layer = self._draw_layer(rs)
            ids = np.concatenate([arch.rows(layer, f) for f in forwards])
            ids = ids[:num_tokens]
        if ids.shape[0] != num_tokens:
            raise ValueError(f"{arch.path}: forward shorter than T={num_tokens}")
        source = os.path.basename(arch.path)
        return ids, {"archive": source, "forwards": forwards, "layer": layer}

    def _sample_extend(
        self, num_tokens: int, rs: np.random.RandomState
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._extend_path is None:
            raise FileNotFoundError(f"No extend_real.npz in {self.routing_dir}")
        arch = self._archive(self._extend_path)
        eligible = np.flatnonzero(arch.lengths >= num_tokens)
        if len(eligible) > 0:
            forwards = [int(eligible[rs.randint(len(eligible))])]
        else:
            order = rs.permutation(arch.num_forwards)
            total = np.cumsum(arch.lengths[order])
            if total[-1] < num_tokens:
                raise ValueError(f"{arch.path} holds fewer than T={num_tokens} tokens")
            forwards = [int(f) for f in order[: np.searchsorted(total, num_tokens) + 1]]
        layer = self._draw_layer(rs)
        ids = np.concatenate([arch.rows(layer, f) for f in forwards])[:num_tokens]
        source = os.path.basename(arch.path)
        return ids, {"archive": source, "forwards": forwards, "layer": layer}

    def _sample_prefill(
        self, num_tokens: int, rs: np.random.RandomState
    ) -> tuple[np.ndarray, dict[str, Any]]:
        chunks = self._chunks()
        left = np.array([c.tokens_to_stream_end for c in chunks], dtype=np.int64)
        eligible = np.flatnonzero(left >= num_tokens)
        if len(eligible) == 0:
            raise FileNotFoundError(
                f"No prefill stream with T={num_tokens} tokens in {self.prefill_dir}"
            )
        first = int(eligible[rs.randint(len(eligible))])
        layer = self._draw_layer(rs)
        parts: list[np.ndarray] = []
        collected, idx = 0, first
        while collected < num_tokens:
            parts.append(
                self._prefill_ids(chunks[idx].path)[layer][: num_tokens - collected]
            )
            collected += parts[-1].shape[0]
            idx += 1
        source = os.path.basename(chunks[first].path)
        return np.concatenate(parts), {
            "archive": source,
            "chunks": idx - first,
            "layer": layer,
        }

    def _real_ids(self, num_tokens: int, seed: int) -> tuple[torch.Tensor, dict]:
        rs = np.random.RandomState(seed)
        kind = self.resolve_kind(num_tokens)
        sampler = {
            "verify": self._sample_verify,
            "extend": self._sample_extend,
            "prefill": self._sample_prefill,
        }[kind]
        ids_np, routing = sampler(num_tokens, rs)
        ids_np = ids_np.astype(np.int64)
        counts = np.bincount(ids_np.ravel() // E_LOCAL, minlength=EP_SIZE)
        rank = select_ep_rank(counts, self.ep_rank)
        routing.update(
            kind=kind,
            ep_rank=rank,
            rank_over_mean=float(counts[rank] / counts.mean()),
            busiest_over_mean=float(counts.max() / counts.mean()),
        )
        ids = rotate_ids_to_rank(ids_np, rank).astype(np.int32)
        return torch.from_numpy(ids).to(self.device).contiguous(), routing

    def _synthetic_routing(
        self, num_tokens: int, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=self.device).manual_seed(seed)
        dev = self.device
        x = torch.randn((num_tokens, HIDDEN_DIM), generator=gen, device=dev).to(
            torch.bfloat16
        )
        if self.synthetic_dist == "zipf":
            perm = torch.randperm(E_GLOBAL, generator=gen, device=dev)
            probs = torch.arange(1, E_GLOBAL + 1, dtype=torch.float32, device=dev)
            probs = probs.pow(-ZIPF_ALPHA)
            expert_probs = torch.empty_like(probs)
            expert_probs[perm] = probs / probs.sum()
            u = torch.rand((num_tokens, E_GLOBAL), generator=gen, device=dev)
            gumbel = -(-u.clamp(1e-6, 1.0 - 1e-6).log()).log()
            ids = (expert_probs.log().unsqueeze(0) + gumbel).topk(TOP_K, dim=-1).indices
            raw_w = torch.rand((num_tokens, TOP_K), generator=gen, device=dev) + 0.05
            weights = raw_w / raw_w.sum(dim=-1, keepdim=True)
        else:
            scores = torch.rand((num_tokens, E_GLOBAL), generator=gen, device=dev)
            if self.synthetic_dist == "normal":
                z = torch.randn((E_GLOBAL,), generator=gen, device=dev)
                u = torch.rand((E_GLOBAL,), generator=gen, device=dev).clamp(min=1e-12)
                scores = scores + NORMAL_POPULARITY * z / (-u.log()).sqrt()
            elif self.synthetic_dist == "skewed":
                heat = torch.rand((E_GLOBAL,), generator=gen, device=dev).pow(6)
                scores = scores + 4.0 * heat
                n_out = max(1, int(HIDDEN_DIM * OUTLIER_FRAC))
                idx = torch.randperm(HIDDEN_DIM, generator=gen, device=dev)[:n_out]
                x[:, idx] = (x[:, idx].float() * OUTLIER_SCALE).to(x.dtype)
            else:
                raise ValueError(f"Unknown synthetic_dist: {self.synthetic_dist!r}")
            routed = scores.topk(TOP_K, dim=-1)
            ids = routed.indices
            weights = routed.values / routed.values.sum(dim=-1, keepdim=True)
        counts = torch.bincount((ids // E_LOCAL).flatten(), minlength=EP_SIZE)
        rank = select_ep_rank(counts.cpu().numpy(), self.ep_rank)
        ids = rotate_ids_to_rank(ids, rank).to(torch.int32).contiguous()
        return x, ids, weights.float().contiguous()

    def sample(self, num_tokens: int, seed: int) -> StepInput:
        """Draw one routing and its activations."""
        if self.routing_dir is not None:
            ids, routing = self._real_ids(num_tokens, seed)
            gen = torch.Generator(device=self.device).manual_seed(seed)
            x = torch.randn(
                (num_tokens, HIDDEN_DIM), generator=gen, device=self.device
            ).to(torch.bfloat16)
            w = torch.rand((num_tokens, TOP_K), generator=gen, device=self.device)
            w = w + 0.05
            weights = (w / w.sum(dim=-1, keepdim=True)).float().contiguous()
        else:
            x, ids, weights = self._synthetic_routing(num_tokens, seed)
            routing = {"kind": f"synthetic_{self.synthetic_dist}"}
        local = ids[ids < E_LOCAL]
        return StepInput(
            x, ids, weights, int(local.numel()), int(local.unique().numel()), routing
        )

    def describe(self) -> dict[str, Any]:
        """Routing options and the archives read so far."""
        return {
            "routing_dir": self.routing_dir,
            "prefill_dir": self.prefill_dir,
            "routing_kind": self.routing_kind,
            "synthetic_dist": None if self.routing_dir else self.synthetic_dist,
            "ep_rank": self.ep_rank,
            "layers": list(self.layers),
            "archives": {
                os.path.basename(p): _meta_sha16(p) for p in sorted(self._used_paths)
            },
        }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def latin_square_orders(num_arms: int) -> list[list[int]]:
    """A cyclic Latin square of arm indices followed by its row-reversed copy.

    Over `2 * num_arms` rounds every arm takes every slot twice and runs before
    each other arm in exactly half of the rounds.
    """
    base = [[(i + j) % num_arms for j in range(num_arms)] for i in range(num_arms)]
    return base + [order[::-1] for order in base]


def trimmed_mean(values: Sequence[float] | np.ndarray, trim: float = TRIM) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    k = int(len(arr) * trim)
    return float(arr[k : len(arr) - k].mean() if len(arr) > 2 * k else arr.mean())


def _trimmed_row_means(samples: np.ndarray, trim: float = TRIM) -> np.ndarray:
    n = samples.shape[1]
    k = int(n * trim)
    ordered = np.sort(samples, axis=1)
    return ordered[:, k : n - k].mean(axis=1) if n > 2 * k else ordered.mean(axis=1)


def _percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def paired_comparison(
    t_ctrl: Sequence[float],
    t_cand: Sequence[float],
    orders: Sequence[Sequence[str]],
    ctrl: str,
    cand: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired speedup of `cand` over `ctrl` (above 1 means `cand` is faster).

    Args:
        t_ctrl: Per-round control latencies (us per call).
        t_cand: Per-round candidate latencies, paired with `t_ctrl` by round.
        orders: The arm order of every round.
        ctrl: Control arm name in `orders`.
        cand: Candidate arm name in `orders`.
        n_boot: Bootstrap resamples over rounds.
        seed: Bootstrap seed.

    Returns:
        `ratio` = `tm(ctrl) / (tm(ctrl) - tm(ctrl - cand))` with `tm` the
        5%-trimmed mean over rounds, and `ratio_ci_95`, its bootstrap 95% CI;
        `paired_ratio_tm`, the trimmed mean of per-round `ctrl / cand`, with
        `paired_ratio_ci_95`; `tm_delta_us`; and `before` / `after`, the paired
        ratio over the rounds where the control ran before (after) `cand`.

    """
    tc = np.asarray(t_ctrl, dtype=np.float64)
    tx = np.asarray(t_cand, dtype=np.float64)
    if len(tc) == 0 or len(tc) != len(tx):
        raise ValueError("t_ctrl and t_cand must be non-empty and paired")
    delta = tc - tx
    rr = tc / tx
    mc = trimmed_mean(tc)
    tm_delta = trimmed_mean(delta)
    idx = np.random.RandomState(seed).randint(0, len(tc), size=(n_boot, len(tc)))
    mc_b = _trimmed_row_means(tc[idx])
    denom_b = mc_b - _trimmed_row_means(delta[idx])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_b = np.where(denom_b > 0, mc_b / denom_b, np.inf)
    before = [i for i, o in enumerate(orders) if o.index(ctrl) < o.index(cand)]
    after = [i for i, o in enumerate(orders) if o.index(ctrl) > o.index(cand)]
    return {
        "ratio": float(mc / (mc - tm_delta)) if mc > tm_delta else float("inf"),
        "ratio_ci_95": _percentile_ci(ratio_b),
        "paired_ratio_tm": trimmed_mean(rr),
        "paired_ratio_ci_95": _percentile_ci(_trimmed_row_means(rr[idx])),
        "tm_delta_us": tm_delta,
        "before": trimmed_mean(rr[before]) if before else None,
        "after": trimmed_mean(rr[after]) if after else None,
    }


def _geomean(values: Sequence[float]) -> float | None:
    valid = [v for v in values if v > 0 and math.isfinite(v)]
    return math.exp(sum(map(math.log, valid)) / len(valid)) if valid else None


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class L2Flusher:
    """Evicts the L2 cache by reading a buffer of at least twice its size."""

    def __init__(self, size_mb: int | None, device: torch.device) -> None:
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        num_bytes = max(2 * l2_bytes, (size_mb or 0) * 1024 * 1024)
        self.buffer = torch.ones(num_bytes // 4, dtype=torch.float32, device=device)
        self.sink = torch.empty((), dtype=torch.float32, device=device)
        self.size_mb = num_bytes / _MB

    def __call__(self) -> None:
        torch.sum(self.buffer, dim=0, out=self.sink)


def _in_arm_context(arm: BenchmarkArm | None, fn: Callable[[], Any]) -> Any:
    if arm is None or arm.vllm_config is None:
        return fn()
    with (
        set_current_vllm_config(arm.vllm_config),
        set_forward_context(None, arm.vllm_config),
    ):
        return fn()


def _run_calls(
    arm: BenchmarkArm | None,
    inputs: Sequence[StepInput],
    flush: L2Flusher | None,
) -> list[torch.Tensor]:
    """Run one MoE call per input (`arm=None` runs only the flushes)."""
    outs = []
    for d in inputs:
        if flush is not None:
            flush()
        if arm is not None:
            outs.append(arm.forward_fn(d))
    return outs


def _warmup(arm: BenchmarkArm, inputs: Sequence[StepInput]) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _in_arm_context(arm, lambda: _run_calls(arm, inputs[:2], None))
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()


def _peak_delta_mb(arm: BenchmarkArm, sample: StepInput) -> float:
    torch.accelerator.synchronize()
    base = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    out = _in_arm_context(arm, lambda: arm.forward_fn(sample))
    torch.accelerator.synchronize()
    del out
    return max(0.0, (torch.accelerator.max_memory_allocated() - base) / _MB)


def time_rounds(
    names: Sequence[str],
    launch: Callable[[str], None],
    rounds: int,
    calls_per_launch: int,
) -> tuple[dict[str, list[float]], list[list[str]]]:
    """Time every unit once per round, in Latin-square order, back to back.

    The host synchronizes once per round. `rounds` is rounded up to whole
    cycles of the `2 * len(names)` orders.
    """
    latin = latin_square_orders(len(names))
    rounds = -(-rounds // len(latin)) * len(latin)
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in names
    ]
    timings: dict[str, list[float]] = {name: [] for name in names}
    orders: list[list[str]] = []
    for r in range(rounds):
        order = [names[i] for i in latin[r % len(latin)]]
        orders.append(order)
        for (start, end), name in zip(events, order):
            start.record()
            launch(name)
            end.record()
        torch.accelerator.synchronize()
        for (start, end), name in zip(events, order):
            timings[name].append(start.elapsed_time(end) * 1e3 / calls_per_launch)
    return timings, orders


def run_shape(
    arms: Sequence[BenchmarkArm],
    provider: RoutingProvider,
    pool: int,
    num_tokens: int,
    args: Any,
    checkpoint: dict[str, torch.Tensor],
    post_norm_w: torch.Tensor,
    post_norm_eps: float,
    flush: L2Flusher | None,
) -> dict[str, Any]:
    """Benchmark every arm on `N` routings of `num_tokens` tokens."""
    n = args.prefill_n if args.prefill_n and num_tokens >= 1024 else args.n
    ds = [provider.sample(num_tokens, pool + 37 * num_tokens + i) for i in range(n)]
    arm_map = {a.name: a for a in arms}
    names = [a.name for a in arms]

    # Grow every workspace to its high-water mark before any capture.
    for arm in arms:
        _warmup(arm, ds)
    peak_mb = {a.name: _peak_delta_mb(a, ds[0]) for a in arms}
    eager_first = (
        {
            CANDIDATE: _in_arm_context(
                arm_map[CANDIDATE], lambda: arm_map[CANDIDATE].forward_fn(ds[0])
            )
        }
        if CANDIDATE in arm_map
        else {}
    )

    units = [*names, FLUSH_ARM] if flush is not None else names
    outputs: dict[str, list[torch.Tensor]] = {}
    if args.no_graph:
        for arm in arms:
            outputs[arm.name] = _in_arm_context(
                arm, lambda a=arm: _run_calls(a, ds, None)
            )

        def launch(name: str) -> None:
            arm = arm_map.get(name)
            _in_arm_context(arm, lambda: _run_calls(arm, ds, flush))

    else:
        graphs: dict[str, torch.cuda.CUDAGraph] = {}
        for name in units:
            arm = arm_map.get(name)
            graph = torch.cuda.CUDAGraph()

            def capture(
                arm: BenchmarkArm | None = arm, graph: torch.cuda.CUDAGraph = graph
            ) -> list[torch.Tensor]:
                with torch.cuda.graph(graph):
                    return _run_calls(arm, ds, flush)

            outs = _in_arm_context(arm, capture)
            graphs[name] = graph
            if arm is not None:
                outputs[name] = outs
        for name in units:
            graphs[name].replay()

        def launch(name: str) -> None:
            graphs[name].replay()

    torch.accelerator.synchronize()

    # Correctness on exactly the outputs that are timed.
    row_checks: dict[str, Any] = {
        "all_finite": {
            name: all(bool(torch.isfinite(t).all()) for t in outputs[name])
            for name in names
        }
    }
    if CANDIDATE in outputs:
        ref_out = outputs[CANDIDATE]
        row_checks["rel_l2_vs_hopper_nvfp4"] = {
            name: max(_relative_l2(x, y) for x, y in zip(outputs[name], ref_out))
            for name in names
            if name != CANDIDATE
        }
        row_checks["graph_eager_bitwise_equal"] = bool(
            torch.equal(eager_first[CANDIDATE], ref_out[0])
        )
        if TWIN in outputs:
            row_checks["twin_bitwise_equal"] = all(
                torch.equal(x, y) for x, y in zip(ref_out, outputs[TWIN])
            )
    if not args.skip_ref_check:
        k = n if args.ref_samples <= 0 else min(n, args.ref_samples)
        refs = fp32_reference_batch(ds[:k], checkpoint, post_norm_w, post_norm_eps)
        row_checks["rel_l2_fp32"] = {
            name: max(_relative_l2(outputs[name][i], refs[i]) for i in range(k))
            for name in names
        }
        del refs
    del eager_first
    if args.no_graph:
        outputs.clear()
        torch.accelerator.empty_cache()

    for _ in range(args.warmup_rounds):
        for name in units:
            launch(name)
    torch.accelerator.synchronize()
    raw, orders = time_rounds(units, launch, args.rounds, n)
    flush_us = raw.pop(FLUSH_ARM, None)
    timings = {
        name: [t - f for t, f in zip(raw[name], flush_us)] if flush_us else raw[name]
        for name in names
    }

    us = {name: trimmed_mean(timings[name]) for name in names}
    mean_active = float(np.mean([d.active_local_experts for d in ds]))
    mean_routes = float(np.mean([d.local_routes for d in ds]))
    compare_orders = [[x for x in o if x != FLUSH_ARM] for o in orders]

    def compare(ctrl: str) -> dict[str, Any]:
        return paired_comparison(
            timings[ctrl],
            timings[CANDIDATE],
            compare_orders,
            ctrl,
            CANDIDATE,
            args.bootstrap_resamples,
        )

    comparisons: dict[str, Any] = {}
    if CANDIDATE in timings:
        for name in names:
            if name not in (CANDIDATE, TWIN):
                comparisons[f"{name}_vs_{CANDIDATE}"] = compare(name)
        if TWIN in timings:
            comparisons["null_twin"] = compare(TWIN)
            ratio = comparisons["null_twin"]["ratio"]
            comparisons["null_twin"]["in_band"] = (
                NULL_TWIN_BAND[0] <= ratio <= NULL_TWIN_BAND[1]
            )
        for key, pool_names in (
            ("best_humming", [h for h in HUMMING_ARMS if h in us]),
            ("best_baseline", [b for b in (MARLIN, *HUMMING_ARMS) if b in us]),
        ):
            if pool_names:
                best = min(pool_names, key=us.__getitem__)
                comparisons[f"{key}_vs_{CANDIDATE}"] = {
                    "baseline": best,
                    **comparisons[f"{best}_vs_{CANDIDATE}"],
                }

    row = {
        "pool": pool,
        "T": num_tokens,
        "num_routings": n,
        "rounds": len(orders),
        "use_graph": not args.no_graph,
        "l2_flush": flush is not None,
        "mean_local_routes": mean_routes,
        "mean_active_local_experts": mean_active,
        "mean_rank_over_mean": _mean_field(ds, "rank_over_mean"),
        "us": us,
        "weight_gbps": {
            name: mean_active * WEIGHT_BYTES_PER_EXPERT / (t * 1e3)
            for name, t in us.items()
        },
        "tflops": {
            name: mean_routes * FLOPS_PER_ROUTE / (t * 1e6) for name, t in us.items()
        },
        "comparisons": comparisons,
        **row_checks,
        "peak_mem_delta_mb": peak_mb,
        "declared_workspace_mb": {a.name: a.workspace_mb_fn(num_tokens) for a in arms},
        "routing_sources": [d.routing for d in ds],
        "timings_us": timings,
        "flush_us": flush_us,
        "orders": orders,
    }
    del outputs
    if not args.no_graph:
        del graphs
    gc.collect()
    torch.accelerator.empty_cache()
    return row


def _mean_field(ds: Sequence[StepInput], key: str) -> float | None:
    vals = [d.routing[key] for d in ds if key in d.routing]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_ratio(c: dict[str, Any] | None) -> str:
    if c is None:
        return "n/a"
    lo, hi = c["ratio_ci_95"]
    return f"{c['ratio']:.3f}x [{lo:.3f},{hi:.3f}]"


def format_row(row: dict[str, Any]) -> str:
    us = " ".join(f"{k}={v:.1f}" for k, v in row["us"].items())
    comps = row["comparisons"]
    parts = [f"p{row['pool']} T={row['T']:<5d} us({us})"]
    for key, label in (
        (f"{MARLIN}_vs_{CANDIDATE}", "marlin/hopper"),
        (f"best_humming_vs_{CANDIDATE}", "best_humming/hopper"),
    ):
        if key in comps:
            parts.append(f"{label}={_fmt_ratio(comps[key])}")
    if "null_twin" in comps:
        eq = int(bool(row.get("twin_bitwise_equal")))
        parts.append(f"null={comps['null_twin']['ratio']:.4f} eq={eq}")
    if row.get("rel_l2_fp32"):
        l2 = " ".join(f"{k}={v:.5f}" for k, v in row["rel_l2_fp32"].items())
        parts.append(f"fp32_l2({l2})")
    return " | ".join(parts)


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rows into buckets by `T` plus the null-twin and accuracy checks."""

    def bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"rows": len(rows)}
        keys = sorted({k for r in rows for k in r["comparisons"] if k != "null_twin"})
        for key in keys:
            comps = [r["comparisons"][key] for r in rows if key in r["comparisons"]]
            ratios = [c["ratio"] for c in comps]
            out[key] = {
                "geomean": _geomean(ratios),
                "min": min(ratios),
                "max": max(ratios),
                "min_ci_lo": min(c["ratio_ci_95"][0] for c in comps),
            }
        return out

    buckets = {
        "verify_8_256": [r for r in results if 8 <= r["T"] <= 256],
        "decode_1_2048": [r for r in results if r["T"] <= 2048],
        "prefill_1024_16384": [r for r in results if 1024 <= r["T"] <= 16384],
    }
    twins = [
        r["comparisons"]["null_twin"]["ratio"]
        for r in results
        if "null_twin" in r["comparisons"]
    ]
    l2 = [r["rel_l2_fp32"] for r in results if r.get("rel_l2_fp32")]

    def all_true(key: str) -> bool | None:
        vals = [r[key] for r in results if key in r]
        return all(vals) if vals else None

    return {
        "buckets": {k: bucket(v) for k, v in buckets.items() if v},
        "null_twin_range": [min(twins), max(twins)] if twins else None,
        "checks": {
            "null_twin_in_band": (
                all(NULL_TWIN_BAND[0] <= t <= NULL_TWIN_BAND[1] for t in twins)
                if twins
                else None
            ),
            "twin_bitwise_equal": all_true("twin_bitwise_equal"),
            "graph_eager_bitwise_equal": all_true("graph_eager_bitwise_equal"),
            "hopper_nvfp4_rel_l2_le_marlin": (
                all(x[CANDIDATE] <= x[MARLIN] for x in l2)
                if l2 and {CANDIDATE, MARLIN} <= set(l2[0])
                else None
            ),
        },
        "max_rel_l2_fp32": {
            arm: max(x[arm] for x in l2 if arm in x) for arm in (l2[0] if l2 else {})
        },
        "max_peak_mem_delta_mb": {
            arm: max(r["peak_mem_delta_mb"][arm] for r in results)
            for arm in (results[0]["peak_mem_delta_mb"] if results else {})
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Summary ===")
    for name, stats in summary["buckets"].items():
        bits = [f"{name} (rows={stats['rows']})"]
        for key, s in stats.items():
            if key != "rows" and s["geomean"] is not None:
                bits.append(
                    f"{key.removesuffix('_vs_' + CANDIDATE)}: geo={s['geomean']:.3f}x "
                    f"min={s['min']:.3f}x min_ci_lo={s['min_ci_lo']:.3f}"
                )
        print("  " + " | ".join(bits))
    print(f"  null twin range: {summary['null_twin_range']}")
    print(f"  checks: {summary['checks']}")
    print(f"  max rel L2 vs FP32: {summary['max_rel_l2_fp32']}")
    print(f"  max peak memory delta (MiB): {summary['max_peak_mem_delta_mb']}")


def _git_revision() -> str | None:
    with suppress(OSError, subprocess.CalledProcessError):
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    return None


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    for dist in (name, f"{name}-kernels"):
        with suppress(PackageNotFoundError):
            return version(dist)
    return None


def _environment() -> dict[str, str]:
    secret = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("VLLM_", "CUDA_", "TORCH_CUDA_ARCH")) and not secret.search(k)
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> Any:
    parser = FlexibleArgumentParser(
        description="Benchmark the Muse Spark NVFP4 MoE experts on Hopper (SM90)."
    )
    parser.add_argument(
        "--mode",
        choices=("decode", "prefill", "all"),
        default="decode",
        help="Token counts to run when `--shapes` is not given.",
    )
    parser.add_argument("--shapes", type=int, nargs="+", help="Token counts T.")
    parser.add_argument(
        "--pools",
        type=int,
        nargs="+",
        default=[61000, 67000, 73000],
        help="Routing seed pools; each pool draws its own routings.",
    )
    parser.add_argument("--n", type=int, default=24, help="Routings per graph.")
    parser.add_argument(
        "--prefill-n", type=int, default=None, help="Routings per graph for T >= 1024."
    )
    parser.add_argument("--rounds", type=int, default=40, help="Timed rounds.")
    parser.add_argument(
        "--warmup-rounds", type=int, default=3, help="Untimed rounds before timing."
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=2000, help="Bootstrap resamples."
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ALL_ARMS,
        default=list(ALL_ARMS),
        help="Arms to benchmark.",
    )
    parser.add_argument(
        "--routing-dir",
        default=None,
        help="Directory of captured routing archives; synthetic routing if unset.",
    )
    parser.add_argument(
        "--prefill-dir",
        default=None,
        help="Prefill chunk archives (default: `<routing-dir>/prefill`).",
    )
    parser.add_argument(
        "--routing-kind",
        choices=("auto", "verify", "extend", "prefill"),
        default="auto",
        help="`auto`: verify up to T=256, extend below 1024, prefill above.",
    )
    parser.add_argument(
        "--ep-rank",
        type=lambda s: s if s in EP_RANK_MODES else int(s),
        default="busiest",
        help="EP rank to emulate per routing: busiest, median or 0..7.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Routing layers to sample (default: 1..60).",
    )
    parser.add_argument(
        "--synthetic-dist",
        choices=("normal", "zipf", "skewed"),
        default="normal",
        help="Synthetic routing distribution.",
    )
    parser.add_argument(
        "--weights-seed", type=int, default=5252, help="Synthetic weight seed."
    )
    parser.add_argument(
        "--scale-profile",
        choices=tuple(SCALE_LOG2_MEAN),
        default="split",
        help="Global and block scales of the synthetic weights.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Muse Spark NVFP4 checkpoint to take real experts from.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Decoder layer of `--checkpoint-dir` (also the routing layer).",
    )
    parser.add_argument(
        "--allow-marlin-fallback",
        action="store_true",
        help="Keep going if a hopper_nvfp4 arm does not get the Hopper kernels.",
    )
    parser.add_argument(
        "--l2-flush",
        choices=("none", "call"),
        default="none",
        help="`call` flushes L2 before every call and subtracts the flush time.",
    )
    parser.add_argument(
        "--l2-flush-mb",
        type=int,
        default=None,
        help="Flush buffer size (default and minimum: twice the L2 size).",
    )
    parser.add_argument(
        "--no-graph", action="store_true", help="Time eager calls instead of graphs."
    )
    parser.add_argument(
        "--ref-samples",
        type=int,
        default=4,
        help="Routings per (pool, T) checked against FP32 (0 = all).",
    )
    parser.add_argument(
        "--skip-ref-check", action="store_true", help="Skip the FP32 reference."
    )
    parser.add_argument("--json", default=None, help="Write the results to this file.")
    args = parser.parse_args()
    if isinstance(args.ep_rank, int) and not 0 <= args.ep_rank < EP_SIZE:
        parser.error(f"--ep-rank must be busiest, median or 0..{EP_SIZE - 1}")
    if args.shapes is not None and min(args.shapes) <= 0:
        parser.error("--shapes must be positive")
    if args.checkpoint_dir is not None:
        if args.layer is None or not isinstance(args.ep_rank, int):
            parser.error("--checkpoint-dir needs --layer and a numeric --ep-rank")
        args.layers = args.layers or [args.layer]
    return args


def _resolve_shapes(args: Any) -> list[int]:
    if args.shapes is not None:
        return list(args.shapes)
    if args.mode == "decode":
        return list(DECODE_SHAPES)
    if args.mode == "prefill":
        return list(PREFILL_SHAPES)
    return sorted(set(DECODE_SHAPES + PREFILL_SHAPES))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_muse_spark_moe.py needs a CUDA GPU.")
    shapes = _resolve_shapes(args)
    requested = list(dict.fromkeys(args.arms))
    device = torch.device("cuda", torch.accelerator.current_device_index())
    props = torch.cuda.get_device_properties(device)
    print(
        f"device={props.name} sm={props.major}{props.minor} "
        f"sms={props.multi_processor_count} E={E_GLOBAL} E_local={E_LOCAL} "
        f"D={HIDDEN_DIM} I={INTERMEDIATE_DIM} top_k={TOP_K}",
        flush=True,
    )
    if props.major != 9:
        print(
            f"warning: the Hopper NVFP4 MoE kernels need SM90; on SM{props.major}"
            f"{props.minor} the hopper_nvfp4 arms may fall back to Marlin.",
            flush=True,
        )

    fd, init_file = tempfile.mkstemp(prefix="vllm_bench_muse_moe_")
    os.close(fd)
    t_start = time.time()
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{init_file}",
                local_rank=0,
                backend=current_platform.dist_backend,
            )
            initialize_model_parallel(1, 1)
        init_workspace_manager(device)
        if args.checkpoint_dir is None:
            checkpoint, post_norm_w = generate_checkpoint(
                args.weights_seed, device, args.scale_profile
            )
            post_norm_eps = POST_NORM_EPS
            weights = {"kind": "synthetic", "seed": args.weights_seed}
            weights["scale_profile"] = args.scale_profile
        else:
            checkpoint, post_norm_w, post_norm_eps, weights = load_checkpoint_experts(
                args.checkpoint_dir, args.layer, args.ep_rank, device
            )
            weights["kind"] = "checkpoint"
        weights["scales"] = scale_summary(checkpoint)
        print(f"weights: {json.dumps(weights)}", flush=True)

        arms: list[BenchmarkArm] = []
        skipped: dict[str, str] = {}
        humming_ws = None
        for name in requested:
            if name in HUMMING_ARMS:
                if humming_ws is None:
                    humming_ws = HummingMoEWorkspace.allocate(max(shapes), device)
                arm, reason = _build_humming_arm(
                    name, checkpoint, post_norm_w, post_norm_eps, device, humming_ws
                )
                if arm is None:
                    assert reason is not None
                    skipped[name] = reason
                    print(f"skipping {name}: {reason}", flush=True)
                    continue
            else:
                arm = _build_runner_arm(
                    name,
                    checkpoint,
                    post_norm_w,
                    post_norm_eps,
                    device,
                    args.allow_marlin_fallback,
                )
            arms.append(arm)
        if not any(a.name == CANDIDATE for a in arms):
            raise RuntimeError(f"the {CANDIDATE} arm is required")

        flush = L2Flusher(args.l2_flush_mb, device) if args.l2_flush == "call" else None
        provider = RoutingProvider(
            args.routing_dir,
            args.routing_kind,
            args.synthetic_dist,
            device,
            ep_rank=args.ep_rank,
            layers=args.layers,
            prefill_dir=args.prefill_dir,
        )
        print(
            "arms: "
            + ", ".join(
                f"{a.name}={a.backend_class} ({a.weight_mb:.0f} MiB)" for a in arms
            )
            + f" | graph={not args.no_graph} | l2_flush={args.l2_flush}",
            flush=True,
        )

        results = []
        for pool in args.pools:
            for num_tokens in shapes:
                row = run_shape(
                    arms,
                    provider,
                    pool,
                    num_tokens,
                    args,
                    checkpoint,
                    post_norm_w,
                    post_norm_eps,
                    flush,
                )
                results.append(row)
                print(format_row(row), flush=True)

        summary = summarize(results)
        print_summary(summary)
        elapsed = time.time() - t_start
        print(f"done in {elapsed:.1f}s", flush=True)
        if args.json:
            report = {
                "metadata": {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "command": sys.argv,
                    "vllm_version": vllm.__version__,
                    "git_revision": _git_revision(),
                    "humming_version": _package_version("humming"),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "device_name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "num_sms": props.multi_processor_count,
                    "l2_cache_bytes": props.L2_cache_size,
                    "l2_flush_mb": flush.size_mb if flush is not None else None,
                    "environment": _environment(),
                    "geometry": {
                        "num_experts": E_GLOBAL,
                        "ep_size": EP_SIZE,
                        "num_local_experts": E_LOCAL,
                        "hidden_dim": HIDDEN_DIM,
                        "intermediate_dim": INTERMEDIATE_DIM,
                        "top_k": TOP_K,
                    },
                    "arms": {a.name: a.backend_class for a in arms},
                    "arm_weight_mb": {a.name: a.weight_mb for a in arms},
                    "null_twin": "independent runner with its own weight copy",
                    "peak_mem_delta_mb": "per-call transient allocations after "
                    "warm-up; workspaces are in declared_workspace_mb",
                    "weights": weights,
                    "skipped_arms": skipped,
                    "routing": provider.describe(),
                    "args": vars(args),
                    "elapsed_s": elapsed,
                },
                "results": results,
                "summary": summary,
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"wrote results to {args.json}", flush=True)
    finally:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()
        with suppress(OSError):
            os.unlink(init_file)


if __name__ == "__main__":
    main()
