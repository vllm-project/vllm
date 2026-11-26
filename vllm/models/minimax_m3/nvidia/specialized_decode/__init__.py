# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 specialized decode on NVIDIA GPUs: a decode executor over the
stock backbone, enabled by
`additional_config={"decode_backend": "latency_specialized"}`.

The executor runs FULL-graph decode steps of up to `MAX_ROWS` rows; every
other step runs the stock forward. It issues the same cross-rank collectives as
the stock layers, so the deployment check below is collective over the TP
group and every per-step decision reads host state that all TP ranks share.

The executor runs one precision setup: an fp8 (E4M3) KV cache and an fp8
(E4M3) index-K cache, MXFP8 attention and dense linears, NVFP4 routed experts.
NVFP4 (and BF16) KV and index caches are refused. The stock decode it is
checked against runs the same setup with the switch off. The resolved cache
dtypes are part of `DecodeConfig`, and each stage implementation declares
the ones it runs, so that more setups can be added stage by stage.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Tensor parallel sizes the executor runs at. TP8 and TP16 change the per-rank head
# counts (and the MoE intermediate size) every stage is written for.
SUPPORTED_TP_SIZES = (4,)
# Most DSpark draft tokens per request: a verify step has K + 1 rows a request.
MAX_SPECULATIVE_TOKENS = 7
# Most requests of an executor step.
MAX_REQUESTS = 8
# Rows of the largest executor step: MAX_REQUESTS x (MAX_SPECULATIVE_TOKENS + 1). Larger
# FULL decode graphs run the stock layers.
MAX_ROWS = MAX_REQUESTS * (MAX_SPECULATIVE_TOKENS + 1)
# The KV / index page size the kernels are written for, and the most pages a request may
# span (the page top-k's column limit, kernels/page_topk.MAX_COLS).
PAGE_SIZE = 128
MAX_PAGES = 8192
# --kv-cache-dtype values the executor accepts, by the dtype it reads.
KV_CACHE_DTYPES: dict[str, "CacheDtype"] = {"fp8": "fp8", "fp8_e4m3": "fp8"}
# attention_config.indexer_kv_dtype values the executor accepts (the MiniMax-M3 default,
# "auto", is BF16).
INDEXER_KV_DTYPES: dict[str, "CacheDtype"] = {"fp8": "fp8", "fp8_e4m3": "fp8"}

# KV connectors the executor runs with. A connector's per-layer hooks
# (`wait_for_layer_load`, `save_kv_layer`) wrap the stock attention layers,
# which the executor's steps skip; these connectors' per-layer hooks are no-ops
# and they move KV around the forward instead (`start_load_kv`, `wait_for_save`).
STEP_LEVEL_KV_CONNECTORS = ("SimpleCPUOffloadConnector",)

# The dtypes a cache can have in a DecodeConfig. Only fp8 is accepted today; BF16 is
# named so that stages can already declare it.
CacheDtype = Literal["bf16", "fp8"]

# The stage slots of ``stages.StepStages``, in order (checked there).
STAGE_NAMES = (
    "front",
    "index_score",
    "page_topk",
    "sparse_decode",
    "dense_decode",
    "o_proj",
    "allreduce_norm",
    "moe",
    "dense_mlp",
)


class CacheDtypes(NamedTuple):
    """The dtypes of the caches the attention stages read and write."""

    # The main K/V cache.
    kv: CacheDtype
    # The indexer's index-K cache.
    index: CacheDtype  # type: ignore[assignment]


def fused_finalize_max_tokens(vllm_config: VllmConfig, stock_cap: int) -> int:
    """The most tokens whose MoE top-k finalize the stock model leaves to the
    next layer's fused finalize + all-reduce + norm + MXFP8 (`stock_cap`
    without the switch).

    With the specialized decode requested, every executor step defers, up to
    `MAX_ROWS`: from 33 rows on that replaces the MoE's finalize, the
    shared-expert add, FlashInfer's all-reduce + norm and the MXFP8 quantize of
    the next QKV input (five launches) by the two
    PDL-chained launches it already runs up to 32, and the wide front's GEMV
    reads their MXFP8 rows. Host state only (every rank decides alike); the
    deployment check itself runs in `specialized_decode_config`.
    """
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend != "latency_specialized":
        return stock_cap
    return max(stock_cap, MAX_ROWS)


def cache_dtypes(vllm_config: VllmConfig) -> CacheDtypes | None:
    """The deployment's cache dtypes, or None if the executor refuses them
    (NVFP4, BF16, ...)."""
    kv = KV_CACHE_DTYPES.get(str(vllm_config.cache_config.cache_dtype))
    index = INDEXER_KV_DTYPES.get(str(vllm_config.attention_config.indexer_kv_dtype))
    if kv is None or index is None:
        return None
    return CacheDtypes(kv=kv, index=index)


@dataclass(frozen=True)
class DecodeConfig:
    # The KV and index-K cache dtypes; stage implementations declare which they run
    # (``Stage.cache_dtypes``).
    cache_dtypes: CacheDtypes
    # Rows a request occupies in a decode step: 1 + the DSpark draft tokens.
    query_len: int
    # Tune the executor's FlashInfer GEMMs at build
    # (kernel_config.enable_flashinfer_autotune is not False).
    autotune: bool
    # Compile the executor's JIT kernels for every row count at build.
    jit_warmup: bool
    # The FULL decode CUDA graph sizes the executor runs (every step's rows).
    step_rows: tuple[int, ...] = ()


def max_rows(query_len: int) -> int:
    """Rows of the largest executor step at `query_len` rows a request: whole
    requests of up to `MAX_REQUESTS`, at most `MAX_ROWS`."""
    if query_len < 1:
        raise ValueError(f"query_len must be positive, got {query_len}")
    return min(MAX_ROWS, MAX_REQUESTS * query_len)


def _num_speculative_tokens(vllm_config: VllmConfig) -> int:
    spec = vllm_config.speculative_config
    return 0 if spec is None else int(spec.num_speculative_tokens or 0)


def unsupported(vllm_config: VllmConfig) -> list[str]:
    """Every requirement of the specialized decode this deployment misses."""
    parallel = vllm_config.parallel_config
    attention = vllm_config.attention_config
    cache = vllm_config.cache_config
    spec = vllm_config.speculative_config
    compilation = vllm_config.compilation_config
    cudagraph_mode = compilation.cudagraph_mode
    quant = vllm_config.quant_config
    capture_sizes = compilation.cudagraph_capture_sizes or []
    query_len = 1 + _num_speculative_tokens(vllm_config)
    requirements = {
        "a B200/GB200 (SM100) or B300/GB300 (SM103) GPU": (
            current_platform.is_cuda()
            and (
                current_platform.is_device_capability(100)
                or current_platform.is_device_capability(103)
            )
        ),
        "async scheduling": bool(vllm_config.scheduler_config.async_scheduling),
        "tensor parallel size "
        + " or ".join(str(tp) for tp in SUPPORTED_TP_SIZES)
        + " over the whole world, without pipeline, data (DP attention), expert "
        "or context parallelism or micro-batching": (
            parallel.tensor_parallel_size in SUPPORTED_TP_SIZES
            and parallel.world_size == parallel.tensor_parallel_size
            and parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1
            and not parallel.enable_expert_parallel
            and parallel.decode_context_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and not parallel.use_ubatching
        ),
        f"no speculative decoding, or DSpark with at most "
        f"{MAX_SPECULATIVE_TOKENS} draft tokens and without adaptive "
        "verification": (
            spec is None
            or (
                spec.method == "dspark"
                and _num_speculative_tokens(vllm_config) <= MAX_SPECULATIVE_TOKENS
                and not spec.enable_adaptive_verification
            )
        ),
        # The executor reads and writes fp8 KV only: NVFP4 KV is refused.
        "an fp8 (E4M3) KV cache (--kv-cache-dtype fp8; not nvfp4 or bf16)": (
            cache.cache_dtype in KV_CACHE_DTYPES
        ),
        # Likewise for the index-K cache: NVFP4 and MXFP4 are refused.
        "an fp8 (E4M3) indexer cache (--attention-config "
        '\'{"indexer_kv_dtype": "fp8"}\'; not nvfp4 or bf16), without '
        "indexer context parallelism": (
            attention.indexer_kv_dtype in INDEXER_KV_DTYPES
            and not attention.indexer_decode_cp
            and not attention.minimax_m3_indexer_cp
        ),
        "the MiniMax-M3.1 checkpoint format: MXFP8 attention and dense linears "
        "with NVFP4 routed experts (modelopt_mxfp8, moe_quant_algo NVFP4), on "
        "the default (TRTLLM-gen) MoE backend": (
            quant is not None
            and quant.get_name() == "modelopt_mxfp8"
            and getattr(quant, "moe_quant_algo", None) == "NVFP4"
            and vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe"
        ),
        "FULL CUDA graphs for decode batches only (FULL_AND_PIECEWISE or "
        "FULL_DECODE_ONLY)": (
            cudagraph_mode is not None
            and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL
        ),
        f"a CUDA graph capture size of at most {max_rows(query_len)} rows "
        f"(up to {MAX_REQUESTS} requests of {query_len} rows)": (
            any(size <= max_rows(query_len) for size in capture_sizes)
        ),
        # The decode loop runs in forward; a compiled forward would trace it.
        "no torch.compile of the model": (
            compilation.mode != CompilationMode.STOCK_TORCH_COMPILE
        ),
        # Kernel stages may bind weight addresses at build.
        "no sleep mode": not vllm_config.model_config.enable_sleep_mode,
        "no LoRA or batch invariance, and no KV transfer other than "
        + " or ".join(STEP_LEVEL_KV_CONNECTORS)
        + " (CPU KV offload)": (
            vllm_config.lora_config is None
            and (
                vllm_config.kv_transfer_config is None
                or vllm_config.kv_transfer_config.kv_connector
                in STEP_LEVEL_KV_CONNECTORS
            )
            and not envs.VLLM_BATCH_INVARIANT
        ),
        f"a KV block size of {PAGE_SIZE} (--block-size {PAGE_SIZE})": (
            cache.block_size == PAGE_SIZE
        ),
        f"a max model length of at most {MAX_PAGES * PAGE_SIZE} tokens "
        f"({MAX_PAGES} pages of {PAGE_SIZE})": (
            -(-int(vllm_config.model_config.max_model_len) // PAGE_SIZE) <= MAX_PAGES
        ),
    }
    return [need for need, ok in requirements.items() if not ok]


def specialized_decode_config(vllm_config: VllmConfig) -> DecodeConfig | None:
    """The specialized decode settings, or None for the stock decode.

    Collective over the TP group when requested: every rank raises if any rank
    cannot run the specialized decode.
    """
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "auto":
        return None
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    missing = unsupported(vllm_config)
    supported = torch.tensor([0 if missing else 1], dtype=torch.int32)
    # The GPU check is rank-local, and ranks that disagree would run different
    # collectives and hang: every rank raises if any rank misses a requirement.
    dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    if not supported.item():
        reason = "; ".join(missing) or "another tensor parallel rank to support it"
        raise ValueError(f"decode_backend=latency_specialized needs {reason}.")
    query_len = 1 + _num_speculative_tokens(vllm_config)
    dtypes = cache_dtypes(vllm_config)
    assert dtypes is not None
    cap = max_rows(query_len)
    step_rows = tuple(
        sorted(
            {
                size
                for size in vllm_config.compilation_config.cudagraph_capture_sizes or []
                if 0 < size <= cap
            }
        )
    )
    logger.info_once(
        "MiniMax-M3.1 specialized decode on, for FULL-graph decode steps of up "
        "to %d rows (%d rows a request; %s KV cache, %s index cache).",
        cap,
        query_len,
        dtypes.kv,
        dtypes.index,
    )
    kernel = vllm_config.kernel_config
    return DecodeConfig(
        cache_dtypes=dtypes,
        query_len=query_len,
        autotune=kernel.enable_flashinfer_autotune is not False,
        jit_warmup=bool(kernel.enable_jit_warmup),
        step_rows=step_rows,
    )
