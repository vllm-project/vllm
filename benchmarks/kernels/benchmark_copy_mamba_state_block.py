# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the `_copy_mamba_state_block` device function.

Isolates the per-(request, state, tile) mamba state copy body from the surrounding
postprocess decision logic. The copy metadata comes from the production
`MambaSpecDecodeGPUContext`, and every config checks the copy before timing
it. `--model` takes any HF hybrid model with a conv +
temporal state per layer; its layout (tp=1) comes from vLLM's own state
shape/dtype logic. The presets listed by `--list-models` work offline.

Grid mirrors the fused-kernel launch shape
(num_reqs, total_states, temporal_tiles).

Usage:
    # nsys (whole-run trace)
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
      --force-overwrite=true -o bench_copy_mamba \
      python benchmarks/kernels/benchmark_copy_mamba_state_block.py

    nsys stats --report cuda_gpu_kern_sum bench_copy_mamba.nsys-rep \
      | grep -i bench_copy_mamba_state_block

    # ncu (kernel metrics; use --nvtx-include to slice per-config, and -k to
    # skip the L2-flush kernels that also run inside each range).
    # The trailing "/" is Nsight's push/pop match marker, not part of the name;
    # the range names below intentionally contain no "/" so they parse.
    ncu --set full --nvtx --nvtx-include "reqs=64_conv=SD_tiles=16/" \
      -k regex:bench_copy_mamba_state_block -o bench_copy_mamba \
      python benchmarks/kernels/benchmark_copy_mamba_state_block.py

    # CUPTI kernel timing only (cold L2, median per config; needs
    # cupti-python >= 13, otherwise FlashInfer falls back to CUDA events):
    python benchmarks/kernels/benchmark_copy_mamba_state_block.py

    # Any HF hybrid model (fetches its config):
    python benchmarks/kernels/benchmark_copy_mamba_state_block.py \
      --model ibm-granite/granite-4.0-h-tiny
"""

import argparse
import math
import os
import statistics
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
import torch.cuda.nvtx as nvtx
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_conv_state_layout,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.model_executor.models import ModelRegistry
from vllm.transformers_utils.config import get_config, get_hf_text_config
from vllm.triton_utils import tl, triton
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroup, MambaSpec
from vllm.v1.worker.mamba_utils import (
    _TEMPORAL_TILES,
    MambaSpecDecodeGPUContext,
    _copy_mamba_state_block,
)

DEVICE = "cuda"


@dataclass(frozen=True)
class ModelStateConfig:
    """State layout parameters for one hybrid-mamba model at tp=1.

    All values are derived from the model's HF config plus a chosen num_spec.
    Conv/temporal shapes here are per-block (num_blocks is added at runtime).
    """

    name: str
    num_state_layers: int
    conv_dim: int
    conv_state_len: int
    temporal_shape: tuple[int, ...]
    conv_dtype: torch.dtype
    temporal_dtype: torch.dtype
    num_spec: int
    layer_kind: str  # "gated_delta_net", "mamba2", or the HF architecture
    notes: str = ""

    @property
    def total_states(self) -> int:
        return self.num_state_layers * 2  # conv + temporal


# Qwen/Qwen3.5-9B: gated delta net, 24 linear + 8 full-attn out of 32.
# See huggingface.co/Qwen/Qwen3.5-9B/config.json and
# MambaStateShapeCalculator.gated_delta_net_state_shape.
#   conv_dim = head_k*num_k_heads*2 + head_v*num_v_heads
#            = 128*16*2 + 128*32 = 8192
#   temporal = (num_v_heads, head_v_dim, head_k_dim) = (32, 128, 128)
_QWEN3_5_9B = ModelStateConfig(
    name="Qwen/Qwen3.5-9B",
    num_state_layers=24,
    conv_dim=8192,
    conv_state_len=4 - 1 + 2,  # conv_kernel - 1 + num_spec
    temporal_shape=(32, 128, 128),
    conv_dtype=torch.bfloat16,
    temporal_dtype=torch.float32,  # mamba_ssm_dtype in the HF config
    num_spec=2,
    layer_kind="gated_delta_net",
    notes="24 linear + 8 full_attention (interval 4) out of 32",
)

# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4: mamba2, 40 M + 8 * + 40 E
# out of 88 layers (hybrid_override_pattern in HF config).
# See huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/config.json
# and MambaStateShapeCalculator.mamba2_state_shape.
#   intermediate = mamba_num_heads * mamba_head_dim = 128*64 = 8192
#   conv_dim = intermediate + 2*n_groups*ssm_state_size
#            = 8192 + 2*8*128 = 10240
#   temporal = (mamba_num_heads, mamba_head_dim, ssm_state_size)
#            = (128, 64, 128)
# num_spec=1 matches num_nextn_predict_layers in the HF config.
_NEMOTRON3_SUPER_120B = ModelStateConfig(
    name="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    num_state_layers=40,
    conv_dim=10240,
    conv_state_len=4 - 1 + 1,  # conv_kernel - 1 + num_spec
    temporal_shape=(128, 64, 128),
    conv_dtype=torch.bfloat16,
    temporal_dtype=torch.float32,  # mamba_ssm_cache_dtype in the HF config
    num_spec=1,
    layer_kind="mamba2",
    notes="40 mamba + 8 attention + 40 moe out of 88",
)

# Qwen/Qwen3.5-0.8B: gated delta net, 18 linear + 6 full-attn out of 24.
# See huggingface.co/Qwen/Qwen3.5-0.8B/config.json and
# MambaStateShapeCalculator.gated_delta_net_state_shape.
#   conv_dim = head_k*num_k_heads*2 + head_v*num_v_heads
#            = 128*16*2 + 128*16 = 6144
#   temporal = (num_v_heads, head_v_dim, head_k_dim) = (16, 128, 128)
_QWEN3_5_0_8B = ModelStateConfig(
    name="Qwen/Qwen3.5-0.8B",
    num_state_layers=18,
    conv_dim=6144,
    conv_state_len=4 - 1 + 2,  # conv_kernel - 1 + num_spec
    temporal_shape=(16, 128, 128),
    conv_dtype=torch.bfloat16,
    temporal_dtype=torch.float32,  # mamba_ssm_dtype in the HF config
    num_spec=2,
    layer_kind="gated_delta_net",
    notes="18 linear + 6 full_attention (interval 4) out of 24",
)

MODEL_CONFIGS: dict[str, ModelStateConfig] = {
    _QWEN3_5_9B.name: _QWEN3_5_9B,
    _QWEN3_5_0_8B.name: _QWEN3_5_0_8B,
    _NEMOTRON3_SUPER_120B.name: _NEMOTRON3_SUPER_120B,
}


def model_state_config_from_hf(
    model: str, num_spec: int | None, trust_remote_code: bool
) -> ModelStateConfig:
    """Derive the state layout of `model` from vLLM's own shape/dtype logic.

    `num_spec=None` uses the model's MTP layer count from its HF config.
    """
    if num_spec is None:
        hf_text_config = get_hf_text_config(get_config(model, trust_remote_code))
        num_spec = (
            getattr(hf_text_config, "mtp_num_hidden_layers", None)
            or getattr(hf_text_config, "num_nextn_predict_layers", None)
            or 0
        )
    # ngram loads no draft model; only num_speculative_tokens affects shapes.
    speculative_config = (
        {"method": "ngram", "num_speculative_tokens": num_spec} if num_spec else None
    )
    vllm_config = EngineArgs(
        model=model,
        trust_remote_code=trust_remote_code,
        skip_tokenizer_init=True,
        speculative_config=speculative_config,
    ).create_engine_config()
    model_config = vllm_config.model_config
    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture, model_config=model_config
    )
    if (
        not hasattr(model_cls, "get_mamba_state_copy_func")
        # Models with several state layouts (e.g. Qwen4Exp) aren't modeled here.
        or hasattr(model_cls, "get_mamba_specs_from_config")
        or model_cls.get_mamba_state_copy_func()
        != (get_conv_copy_spec, get_temporal_copy_spec)
    ):
        raise ValueError(
            f"{model} ({model_config.architecture}) has no single conv + "
            "temporal mamba state layout"
        )

    conv_shape, temporal_shape = model_cls.get_mamba_state_shape_from_config(
        vllm_config
    )
    conv_dtype, temporal_dtype = model_cls.get_mamba_state_dtype_from_config(
        vllm_config
    )
    if is_conv_state_dim_first():
        conv_dim, conv_state_len = conv_shape
    else:
        conv_state_len, conv_dim = conv_shape

    # Older configs tag mamba layers "mamba"; Transformers 5 remaps them to
    # "linear_attention", the tag GDN layers already use.
    parallel_config = vllm_config.parallel_config
    num_state_layers = model_config.get_num_layers_by_block_type(
        parallel_config, "mamba"
    ) or model_config.get_num_layers_by_block_type(parallel_config, "linear_attention")
    if num_state_layers == 0:
        raise ValueError(f"{model}: found no mamba or linear_attention layers")
    total_layers = model_config.get_total_num_hidden_layers()
    return ModelStateConfig(
        name=model,
        num_state_layers=num_state_layers,
        conv_dim=conv_dim,
        conv_state_len=conv_state_len,
        temporal_shape=tuple(temporal_shape),
        conv_dtype=conv_dtype,
        temporal_dtype=temporal_dtype,
        num_spec=num_spec,
        layer_kind=model_config.architecture,
        notes=f"{num_state_layers} state layers out of {total_layers}",
    )


# Each request gets its own src/dst block ids inside every state tensor so
# concurrent copies never alias; the state tensors are sized to 2*num_reqs
# blocks in bench_one. Cold L2 comes from the flush before each timed launch.
SRC_COL = 0
DST_COL = 1
TOKEN_BIAS = 0

COPY_BLOCK_SIZE = 1024
DEFAULT_ITERS_WARMUP = 20
DEFAULT_ITERS_TIMED = 200


@triton.jit(do_not_specialize=["src_col", "dst_col", "token_bias", "num_reqs"])
def bench_copy_mamba_state_block(
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req: tl.int64,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    num_reqs,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    """Minimal wrapper: run the copy body for every (batch, state, tile) program.

    Grid: (num_reqs, total_states, TEMPORAL_TILES).
    """
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)
    if batch_idx >= num_reqs:
        return
    _copy_mamba_state_block(
        state_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        state_dim_row_count_ptr,
        state_dim_row_stride_ptr,
        tile_idx,
        COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST,
        TEMPORAL_TILES,
    )


def build_state_tensors(
    model: ModelStateConfig, conv_state_dim_first: bool, num_blocks: int
) -> dict[str, list[torch.Tensor]]:
    """Allocate one conv and one temporal state tensor per state-carrying layer.

    Returns each layer's `kv_cache` list ([conv, temporal]) keyed by layer
    name. Random contents let `check_copy` tell copied blocks from stale ones.
    """
    conv_shape_per_block = (
        (model.conv_dim, model.conv_state_len)
        if conv_state_dim_first
        else (model.conv_state_len, model.conv_dim)
    )

    layer_states: dict[str, list[torch.Tensor]] = {}
    for i in range(model.num_state_layers):
        conv = torch.randn(
            (num_blocks, *conv_shape_per_block),
            dtype=model.conv_dtype,
            device=DEVICE,
        )
        temporal = torch.randn(
            (num_blocks, *model.temporal_shape),
            dtype=model.temporal_dtype,
            device=DEVICE,
        )
        layer_states[f"layers.{i}.mixer"] = [conv, temporal]
    return layer_states


def build_gpu_context(
    layer_states: dict[str, list[torch.Tensor]], block_table: torch.Tensor
) -> MambaSpecDecodeGPUContext:
    """Populate the copy metadata with the production MambaSpecDecodeGPUContext.

    The context reads only each layer's `kv_cache`, so a namespace holding the
    state tensors stands in for the attention layer in the forward context.
    """
    conv, temporal = next(iter(layer_states.values()))
    spec = MambaSpec(
        # Only the postprocess decision logic reads block_size.
        block_size=16,
        shapes=(tuple(conv.shape[1:]), tuple(temporal.shape[1:])),
        dtypes=(conv.dtype, temporal.dtype),
    )
    layer_config = MambaManager.layer_config(spec, mamba_cache_mode="align")
    kv_cache_config = KVCacheConfig(
        num_blocks=conv.shape[0],
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroup.from_layers(dict.fromkeys(layer_states, layer_config))
        ],
    )
    copy_funcs = {spec.mamba_type: (get_conv_copy_spec, get_temporal_copy_spec)}
    forward_context = {
        name: SimpleNamespace(kv_cache=states) for name, states in layer_states.items()
    }
    ctx = MambaSpecDecodeGPUContext.create(
        max_num_reqs=block_table.shape[0],
        kv_cache_config=kv_cache_config,
        copy_funcs=copy_funcs,
        forward_context=forward_context,
        device=block_table.device,
    )
    ctx.initialize_from_forward_context(
        kv_cache_config, forward_context, copy_funcs, [block_table]
    )
    return ctx


def check_copy(
    layer_states: dict[str, list[torch.Tensor]], block_table: torch.Tensor
) -> None:
    """Assert every state's dst blocks match its src blocks.

    With TOKEN_BIAS=0 both conv and temporal copies move whole blocks.
    """
    src = block_table[:, SRC_COL].long()
    dst = block_table[:, DST_COL].long()
    for name, states in layer_states.items():
        for state_type_idx, state in enumerate(states):
            assert torch.equal(state[dst], state[src]), (
                f"{name} state {state_type_idx}: dst blocks differ from src"
            )


def bytes_moved_per_launch(
    layer_states: dict[str, list[torch.Tensor]], conv_state_dim_first: bool
) -> int:
    """Bytes copied (read+write) across all (num_reqs, state, tile) programs,
    per request. Multiply by num_reqs to get per-launch total."""
    per_req = 0
    for conv, temporal in layer_states.values():
        elem = conv.element_size()
        if conv_state_dim_first:
            per_req += conv.size(1) * (conv.size(2) - TOKEN_BIAS) * elem
        else:
            per_req += (conv.size(1) - TOKEN_BIAS) * conv.stride(1) * elem
        per_req += temporal[0].numel() * temporal.element_size()
    return per_req


def bench_one(
    model: ModelStateConfig,
    num_reqs: int,
    temporal_tiles: int,
    iters_warmup: int,
    iters_timed: int,
) -> None:
    # The production metadata reads the conv layout from the env, so the
    # allocation and the CONV_STATE_DIM_FIRST constexpr follow it too.
    conv_state_dim_first = is_conv_state_dim_first()
    # Give every request distinct src/dst block ids so concurrent copies don't
    # share cache lines. State tensors hold at least 2*num_reqs blocks.
    num_blocks = 2 * num_reqs
    layer_states = build_state_tensors(model, conv_state_dim_first, num_blocks)

    # Single mamba group. Per-request rows point to distinct src/dst blocks:
    # req i reads block 2*i and writes block 2*i+1. Two columns are enough
    # since token_bias=0 (no extra columns read by the temporal path).
    block_table = torch.zeros(num_reqs, 2, dtype=torch.int32, device=DEVICE)
    row_ids = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    block_table[:, SRC_COL] = 2 * row_ids
    block_table[:, DST_COL] = 2 * row_ids + 1
    ctx = build_gpu_context(layer_states, block_table)

    grid = (num_reqs, ctx.num_states, temporal_tiles)
    args = (
        SRC_COL,
        DST_COL,
        TOKEN_BIAS,
        ctx.block_table_ptrs,
        ctx.block_table_stride_req,
        ctx.state_base_addrs,
        ctx.state_block_strides,
        ctx.state_elem_sizes,
        ctx.state_inner_sizes,
        ctx.state_conv_widths,
        ctx.state_group_indices,
        ctx.state_dim_row_count,
        ctx.state_dim_row_stride,
        num_reqs,
    )

    def launch():
        bench_copy_mamba_state_block[grid](
            *args,
            COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
            CONV_STATE_DIM_FIRST=conv_state_dim_first,
            TEMPORAL_TILES=temporal_tiles,
        )

    launch()
    torch.accelerator.synchronize()
    check_copy(layer_states, block_table)

    layout = "DS" if conv_state_dim_first else "SD"
    # Underscore separator: '/' is Nsight's NVTX level marker and breaks
    # --nvtx-include filters when it appears inside a range name.
    label = f"reqs={num_reqs}_conv={layout}_tiles={temporal_tiles}"
    nvtx.range_push(label)
    # Not use_cuda_graph: its no-CUPTI fallback keeps L2 cold by cloning tensor
    # args, which can't relocate states reached through raw metadata pointers.
    times_ms = bench_gpu_time_with_cupti(
        launch,
        dry_run_iters=iters_warmup,
        repeat_iters=iters_timed,
        cold_l2_cache=True,
        use_cuda_graph=False,
    )
    nvtx.range_pop()

    median_us = statistics.median(times_ms) * 1000.0
    bytes_per_req = bytes_moved_per_launch(layer_states, conv_state_dim_first)
    total_bytes = bytes_per_req * num_reqs
    # Read + write: memory traffic is 2x copy size.
    gbps = (2 * total_bytes) / (median_us * 1e-6) / 1e9
    print(
        f"{label:30s} grid={grid!s:16s} "
        f"copy/req={bytes_per_req / 1024 / 1024:6.2f} MiB  "
        f"median={median_us:8.2f} us  BW~{gbps:6.1f} GB/s"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=_QWEN3_5_9B.name,
        help=(
            "HF model whose state layout drives the benchmark. Preset names "
            "(see --list-models) work offline; others fetch the HF config."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List offline preset names for --model and exit.",
    )
    parser.add_argument(
        "--num-spec",
        type=int,
        default=None,
        help=(
            "Speculative tokens, which lengthen the conv state. Default: the "
            "preset's value, or the MTP layer count in the HF config."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the HF config.",
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128],
        help="num_reqs values to sweep (default matches serving benchmark)",
    )
    parser.add_argument(
        "--conv-layout",
        choices=["SD", "DS", "both"],
        default="SD",
        help="Conv state layout to benchmark. Real Qwen3.5-9B is SD.",
    )
    parser.add_argument(
        "--temporal-tiles",
        type=int,
        nargs="+",
        default=[_TEMPORAL_TILES],
        help="TEMPORAL_TILES values to sweep (default: production value).",
    )
    parser.add_argument(
        "--iters-warmup",
        type=int,
        default=DEFAULT_ITERS_WARMUP,
        help="Untimed launches before CUPTI timing.",
    )
    parser.add_argument(
        "--iters-timed",
        type=int,
        default=DEFAULT_ITERS_TIMED,
        help=(
            "CUPTI-timed launches per config, each after an L2 flush; the "
            "median is reported."
        ),
    )
    args = parser.parse_args()

    if args.list_models:
        for name in sorted(MODEL_CONFIGS.keys()):
            print(name)
        return

    assert torch.cuda.is_available(), "CUDA required"
    model = MODEL_CONFIGS.get(args.model)
    if model is None:
        model = model_state_config_from_hf(
            args.model, args.num_spec, args.trust_remote_code
        )
    elif args.num_spec is not None:
        model = replace(
            model,
            num_spec=args.num_spec,
            conv_state_len=model.conv_state_len - model.num_spec + args.num_spec,
        )
    conv_elem = torch.tensor([], dtype=model.conv_dtype).element_size()
    temp_elem = torch.tensor([], dtype=model.temporal_dtype).element_size()
    conv_bytes_per_block = model.conv_dim * model.conv_state_len * conv_elem
    temp_inner = math.prod(model.temporal_shape)
    temp_bytes_per_block = temp_inner * temp_elem

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"model: {model.name} ({model.layer_kind}) tp=1 num_spec={model.num_spec}\n"
        f"state_layers={model.num_state_layers} state_types=2 "
        f"total_states={model.total_states}  # {model.notes}\n"
        f"conv: dim={model.conv_dim} state_len={model.conv_state_len} "
        f"dtype={model.conv_dtype} "
        f"(={conv_bytes_per_block / 1024:.1f} KiB/block)\n"
        f"temporal: shape={model.temporal_shape} inner={temp_inner} "
        f"dtype={model.temporal_dtype} "
        f"(={temp_bytes_per_block / 1024 / 1024:.2f} MiB/block)\n"
        f"iters warmup={args.iters_warmup} timed={args.iters_timed}"
    )
    print()

    layouts = ["SD", "DS"] if args.conv_layout == "both" else [args.conv_layout]

    torch.cuda.cudart().cudaProfilerStart()
    t0 = time.perf_counter()
    for layout in layouts:
        os.environ["VLLM_SSM_CONV_STATE_LAYOUT"] = layout
        get_conv_state_layout.cache_clear()
        for tiles in args.temporal_tiles:
            for nr in args.concurrencies:
                bench_one(model, nr, tiles, args.iters_warmup, args.iters_timed)
    torch.cuda.cudart().cudaProfilerStop()
    print(f"\ntotal wall: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
