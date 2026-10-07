# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for FlashInfer API changes.

Users of vLLM should always import **only** these wrappers.
"""

import contextlib
import functools
import importlib
import importlib.util
import os
import shutil
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NoReturn

import requests
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import PIN_MEMORY

if TYPE_CHECKING:
    from flashinfer.fused_moe.core import ActivationType
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

logger = init_logger(__name__)


_bf16_autotune_buckets: ContextVar[tuple[int, ...] | None] = ContextVar(
    "flashinfer_bf16_autotune_buckets", default=None
)


@contextlib.contextmanager
def autotune_bf16_only(
    tuning_buckets: tuple[int, ...], *, skip_ops: set[str] | None = None
) -> Iterator[None]:
    """Tune BF16 calls with bounded buckets, outside full-model autotuning."""
    token = _bf16_autotune_buckets.set(tuning_buckets)
    try:
        with autotune(tune_mode=False, skip_ops=skip_ops):
            yield
    finally:
        _bf16_autotune_buckets.reset(token)


def flashinfer_bf16_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    pdl: bool,
    backend: str,
) -> torch.Tensor:
    from flashinfer import mm_bf16

    tuning_buckets = _bf16_autotune_buckets.get()
    tuning = (
        autotune(tune_mode=True, tuning_buckets=tuning_buckets)
        if tuning_buckets is not None
        else contextlib.nullcontext()
    )
    with tuning:
        return mm_bf16(
            a,
            b,
            bias=bias,
            pdl=pdl,
            out_dtype=torch.bfloat16,
            backend=backend,
        )


# This is the storage path for the cubins, it can be replaced
# with a local path for testing.
# Referenced from https://github.com/flashinfer-ai/flashinfer/blob/0c9a92c3d9a7e043ab6f3f7b2273269caf6ab044/flashinfer/jit/cubin_loader.py#L35  # noqa: E501
FLASHINFER_CUBINS_REPOSITORY = os.environ.get(
    "FLASHINFER_CUBINS_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",  # noqa: E501
)

_DEFAULT_CUDA_HOME = "/usr/local/cuda"


def _flashinfer_nvcc_path() -> str | None:
    """Return the nvcc FlashInfer's JIT would run, or None if it is missing.

    Mirrors ``flashinfer.jit.cpp_ext.get_cuda_path()`` without importing
    FlashInfer, whose import initializes CUDA.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        cuda_home = (
            os.path.dirname(os.path.dirname(nvcc)) if nvcc else _DEFAULT_CUDA_HOME
        )
    return shutil.which(os.path.join(cuda_home, "bin", "nvcc"))


@functools.cache
def has_flashinfer_cubin() -> bool:
    """Return `True` if flashinfer-cubin package is available."""
    if envs.VLLM_HAS_FLASHINFER_CUBIN:
        return True
    if importlib.util.find_spec("flashinfer_cubin") is not None:
        return True
    logger.debug_once("flashinfer-cubin package was not found")
    return False


@functools.cache
def has_flashinfer() -> bool:
    """Return `True` if flashinfer-python package is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    if importlib.util.find_spec("flashinfer") is None:
        logger.debug_once("FlashInfer unavailable since package was not found")
        return False
    # FlashInfer's JIT runs nvcc and `ninja` (from PATH).
    if not has_flashinfer_cubin() and (
        _flashinfer_nvcc_path() is None or shutil.which("ninja") is None
    ):
        logger.warning_once(
            "FlashInfer kernels are disabled: flashinfer-cubin is not installed "
            "and nvcc (CUDA_HOME, CUDA_PATH, PATH or /usr/local/cuda) or ninja "
            "(PATH) is missing. Set CUDA_HOME to a CUDA toolkit and put ninja on "
            "PATH, or run `flashinfer download-kernels`."
        )
        return False
    return True


@functools.cache
def has_flashinfer_bf16_gemm() -> bool:
    """Return whether FlashInfer exposes the BF16 dense GEMM API."""
    if not has_flashinfer():
        return False
    mod = _get_submodule("flashinfer")
    return mod is not None and callable(getattr(mod, "mm_bf16", None))


@functools.cache
def is_flashinfer_bf16_gemm_supported(
    backend: str,
    compute_capability: int | None = None,
) -> bool:
    """Return whether an exact FlashInfer BF16 backend is available."""
    if not current_platform.is_cuda() or not has_flashinfer_bf16_gemm():
        return False

    mod = _get_submodule("flashinfer")
    mm_bf16 = getattr(mod, "mm_bf16", None) if mod is not None else None
    backend_supported = getattr(mm_bf16, "is_backend_supported", None)
    if not callable(backend_supported):
        return False

    if compute_capability is None:
        device_capability = current_platform.get_device_capability()
        if device_capability is None:
            return False
        compute_capability = device_capability.to_int()

    try:
        return bool(backend_supported(backend, compute_capability))
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return False


@functools.cache
def is_flashinfer_cutedsl_bf16_gemm_supported() -> bool:
    """Return whether the CuTeDSL BF16 dense GEMM backend is available."""
    if not is_flashinfer_bf16_gemm_supported("cute-dsl"):
        return False
    try:
        from flashinfer.cute_dsl.utils import is_cute_dsl_available
        from flashinfer.utils import is_sm100a_supported
    except (ImportError, ModuleNotFoundError):
        return False
    try:
        return is_cute_dsl_available() and is_sm100a_supported(torch.device("cuda"))
    except (RuntimeError, TypeError, ValueError):
        return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable FlashInfer backend."""
    raise RuntimeError(
        "FlashInfer backend is not available. Please install the package "
        "to enable FlashInfer kernels: "
        "https://github.com/flashinfer-ai/flashinfer"
    )


def _missing_sparse_mla(*_: Any, **__: Any) -> NoReturn:
    raise RuntimeError(
        "FlashInfer sparse MLA decode APIs are not available. "
        "Install a FlashInfer build that includes sparse MLA decode support."
    )


def _get_submodule(module_name: str) -> Any | None:
    """Safely import a submodule and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


# General lazy import wrapper
def _lazy_import_wrapper(
    module_name: str, attr_name: str, fallback_fn: Callable[..., Any] = _missing
):
    """Create a lazy import wrapper for a specific function."""

    @functools.cache
    def _get_impl():
        if not has_flashinfer():
            return None
        mod = _get_submodule(module_name)
        return getattr(mod, attr_name, None) if mod else None

    def wrapper(*args, **kwargs):
        impl = _get_impl()
        if impl is None:
            return fallback_fn(*args, **kwargs)
        return impl(*args, **kwargs)

    return wrapper


def pin_host_range_buf(length: int) -> None:
    """Pin flashinfer's cached host-side arange buffer covering `length`.

    flashinfer's decode plan() builds its host qo_indptr from the cached
    `_get_range_buf` entry and copies it to the GPU with non_blocking=True on
    every call; the cache entry is unpinned, so the copy silently blocks the
    host. Replace the entry with a pinned copy so the copies stay async.
    """
    if not PIN_MEMORY:
        return
    mod = _get_submodule("flashinfer.utils")
    if mod is None:
        return
    get_range_buf = getattr(mod, "_get_range_buf", None)
    cache_buf = getattr(mod, "_cache_buf", None)
    ceil_pow2 = getattr(mod, "_ceil_pow2", None)
    if get_range_buf is None or cache_buf is None or ceil_pow2 is None:
        return
    get_range_buf(length, "cpu")  # Ensure the cache entry exists.
    key = (f"range_{ceil_pow2(length)}", "cpu")
    buf = cache_buf[key]
    if not buf.is_pinned():
        cache_buf[key] = buf.pin_memory()


# Create lazy wrappers for each function
flashinfer_trtllm_bf16_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_bf16_moe"
)
flashinfer_trtllm_fp8_block_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"
)
flashinfer_trtllm_fp8_per_tensor_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"
)
flashinfer_cutlass_fused_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "cutlass_fused_moe"
)
flashinfer_cutedsl_grouped_gemm_nt_masked = _lazy_import_wrapper(
    "flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"
)
flashinfer_prepare_bf16_fp4_weights = _lazy_import_wrapper(
    "flashinfer.gemm", "prepare_bf16_fp4_weights"
)
flashinfer_fp4_quantize = _lazy_import_wrapper("flashinfer", "fp4_quantize")
flashinfer_mxfp4_quantize = _lazy_import_wrapper("flashinfer", "mxfp4_quantize")
nvfp4_batched_quantize = _lazy_import_wrapper("flashinfer", "nvfp4_batched_quantize")
silu_and_mul_scaled_nvfp4_experts_quantize = _lazy_import_wrapper(
    "flashinfer", "silu_and_mul_scaled_nvfp4_experts_quantize"
)
scaled_fp4_grouped_quantize = _lazy_import_wrapper(
    "flashinfer", "scaled_fp4_grouped_quantize"
)
nvfp4_block_scale_interleave = _lazy_import_wrapper(
    "flashinfer.fp4_quantization", "block_scale_interleave"
)
flashinfer_cute_dsl_fused_moe_nvfp4 = _lazy_import_wrapper(
    "flashinfer", "cute_dsl_fused_moe_nvfp4"
)
flashinfer_convert_sf_to_mma_layout = _lazy_import_wrapper(
    "flashinfer.cute_dsl.utils", "convert_sf_to_mma_layout"
)
flashinfer_b12x_fused_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "b12x_fused_moe"
)
flashinfer_get_hybrid_num_tokens_buckets = _lazy_import_wrapper(
    "flashinfer.fused_moe.utils", "get_hybrid_num_tokens_buckets"
)
trtllm_fp4_block_scale_moe = _lazy_import_wrapper(
    "flashinfer", "trtllm_fp4_block_scale_moe"
)
flashinfer_trtllm_batch_decode_with_kv_cache_mla = _lazy_import_wrapper(
    "flashinfer.decode",
    "trtllm_batch_decode_with_kv_cache_mla",
    fallback_fn=_missing_sparse_mla,
)
flashinfer_trtllm_batch_decode_sparse_mla_dsv4 = _lazy_import_wrapper(
    "flashinfer.decode",
    "trtllm_batch_decode_sparse_mla_dsv4",
    fallback_fn=_missing_sparse_mla,
)
flashinfer_xqa_batch_decode_with_kv_cache = _lazy_import_wrapper(
    "flashinfer.decode",
    "xqa_batch_decode_with_kv_cache",
)
flashinfer_packed_fused_kda_decode = _lazy_import_wrapper(
    "flashinfer", "packed_fused_kda_decode"
)
flashinfer_recurrent_kda = _lazy_import_wrapper(
    "flashinfer.kda",
    "recurrent_kda",
)
flashinfer_fused_kda_decode = _lazy_import_wrapper(
    "flashinfer.kda_decode",
    "fused_kda_decode",
)


# Special case for autotune since it returns a context manager
autotune = _lazy_import_wrapper(
    "flashinfer.autotuner",
    "autotune",
    fallback_fn=lambda *args, **kwargs: contextlib.nullcontext(),
)


@functools.cache
def has_flashinfer_comm() -> bool:
    """Return `True` if FlashInfer comm module is available."""
    return has_flashinfer() and importlib.util.find_spec("flashinfer.comm") is not None


@functools.cache
def has_flashinfer_packed_fused_kda_decode() -> bool:
    """Return whether FlashInfer's packed fused KDA decode API is available."""
    if not has_flashinfer():
        return False
    module = _get_submodule("flashinfer")
    return bool(module and callable(getattr(module, "packed_fused_kda_decode", None)))


@functools.cache
def has_flashinfer_nvlink_two_sided() -> bool:
    """Return `True` if FlashInfer mnnvl all2all is available."""
    if not has_flashinfer_comm():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.comm", "Mapping"),
        ("flashinfer.comm.mnnvl", "MnnvlMemory"),
        ("flashinfer.comm.trtllm_alltoall", "MnnvlMoe"),
        ("flashinfer.comm.trtllm_alltoall", "MoEAlltoallInfo"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_nvlink_one_sided() -> bool:
    """Return `True` if FlashInfer trtllm_moe_alltoall module is available."""
    if not has_flashinfer_comm():
        return False
    return importlib.util.find_spec("flashinfer.comm.trtllm_moe_alltoall") is not None


@functools.cache
def has_flashinfer_moe() -> bool:
    """Return `True` if FlashInfer MoE module is available."""
    return (
        has_flashinfer()
        and importlib.util.find_spec("flashinfer.fused_moe") is not None
    )


@functools.cache
def has_flashinfer_sm90_nope_mla() -> bool:
    """FlashInfer SM90 NoPE MLA (FP8 KV with in-kernel dequant, kpe=0).

    Feature-detected via the ``ckv_scale_arr`` run() kwarg introduced with
    the SM90 NoPE support (FlashInfer >= 0.6.18), so dev builds carry the
    gate without a version parse.
    """
    if not has_flashinfer():
        return False
    try:
        import inspect

        from flashinfer.mla import BatchMLAPagedAttentionWrapper
    except ImportError:
        return False
    try:
        params = inspect.signature(BatchMLAPagedAttentionWrapper.run).parameters
    except (TypeError, ValueError):
        return False
    return (
        "ckv_scale_arr" in params
        and params["ckv_scale_arr"].kind is inspect.Parameter.KEYWORD_ONLY
    )


@functools.cache
def has_flashinfer_sparse_mla_sm120() -> bool:
    """Return ``True`` if FlashInfer sparse MLA decode support is available."""
    if not has_flashinfer():
        return False
    try:
        from flashinfer.autotuner import autotune
        from flashinfer.decode import (
            trtllm_batch_decode_sparse_mla_dsv4,
            trtllm_batch_decode_with_kv_cache_mla,
        )
    except ImportError:
        return False
    return (
        callable(trtllm_batch_decode_sparse_mla_dsv4)
        and callable(trtllm_batch_decode_with_kv_cache_mla)
        and callable(autotune)
    )


@functools.cache
def has_flashinfer_sparse_mla_sm120_config(num_q_heads: int, top_k: int) -> bool:
    """Return whether FlashInfer ships an SM120 DSV4 decode specialization.

    The public sparse MLA API predates some DSV4 shapes, so checking only that
    the callable exists can select a package that later aborts or rejects a
    valid vLLM configuration. Inspect FlashInfer's dispatch table until it
    exposes a public capability query.
    """
    if not has_flashinfer_sparse_mla_sm120():
        return False
    mod = _get_submodule("flashinfer.mla._sparse_mla_sm120")
    dispatch = getattr(mod, "_DECODE_DSV4_DISPATCH", None) if mod else None
    return dispatch is not None and (int(num_q_heads), int(top_k)) in dispatch


@functools.cache
def has_flashinfer_cutedsl() -> bool:
    """Return ``True`` if FlashInfer cutedsl module is available."""
    return (
        has_flashinfer() and importlib.util.find_spec("flashinfer.cute_dsl") is not None
    )


@functools.cache
def has_flashinfer_bf16_fp4() -> bool:
    """Return ``True`` if FlashInfer's CuTe-DSL W4A16 GEMM is available."""
    if not has_flashinfer_cutedsl():
        return False
    mod = _get_submodule("flashinfer.gemm")
    return mod is not None and all(
        hasattr(mod, name) for name in ("mm_bf16_fp4", "prepare_bf16_fp4_weights")
    )


@functools.cache
def has_flashinfer_recurrent_kda() -> bool:
    """Return whether FlashInfer recurrent KDA prefill is available."""
    if not has_flashinfer():
        return False
    mod = _get_submodule("flashinfer.kda")
    return mod is not None and callable(getattr(mod, "recurrent_kda", None))


@functools.cache
def has_flashinfer_fused_kda_decode() -> bool:
    """Return whether FlashInfer fused KDA decode is available."""
    if not has_flashinfer():
        return False
    mod = _get_submodule("flashinfer.kda_decode")
    return (
        mod is not None
        and bool(getattr(mod, "_FUSED_KDA_DECODE_AVAILABLE", False))
        and callable(getattr(mod, "fused_kda_decode", None))
    )


@functools.cache
def has_flashinfer_trtllm_fused_moe() -> bool:
    """Return `True` if FlashInfer TRTLLM fused MoE is available."""
    if not has_flashinfer_moe():
        return False
    required_functions = [
        ("flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_mxint4_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_bf16_moe"),
    ]
    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return `True` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_moe():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.fused_moe", "cutlass_fused_moe"),
        ("flashinfer", "fp4_quantize"),
        ("flashinfer", "nvfp4_block_scale_interleave"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutedsl_grouped_gemm_nt_masked() -> bool:
    """Return ``True`` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_cutedsl():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"),
        ("flashinfer", "scaled_fp4_grouped_quantize"),
        ("flashinfer", "silu_and_mul_scaled_nvfp4_experts_quantize"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutedsl_moe_nvfp4() -> bool:
    """Return ``True`` if FlashInfer cute_dsl_fused_moe_nvfp4 is available."""
    if not has_flashinfer_cutedsl():
        return False
    mod = _get_submodule("flashinfer")
    return mod is not None and hasattr(mod, "cute_dsl_fused_moe_nvfp4")


@functools.cache
def has_flashinfer_b12x_gemm() -> bool:
    """Return True if FlashInfer b12x FP4 GEMM backend is available (SM120+)."""
    if not has_flashinfer_cutedsl():
        return False
    mod = _get_submodule("flashinfer.gemm")
    if mod is None:
        return False
    # FlashInfer 0.6.11 renamed Sm120BlockScaledDenseGemmKernel ->
    # Sm120B12xBlockScaledDenseGemmKernel (commit 223f2a49). Accept either.
    return hasattr(mod, "Sm120B12xBlockScaledDenseGemmKernel") or hasattr(
        mod, "Sm120BlockScaledDenseGemmKernel"
    )


@functools.cache
def has_flashinfer_b12x_moe() -> bool:
    """Return ``True`` if FlashInfer CuteDSL SM12x fused MoE is available."""
    if not has_flashinfer_moe():
        return False

    required_functions = [
        ("flashinfer.fused_moe", "b12x_fused_moe"),
        ("flashinfer.cute_dsl.utils", "convert_sf_to_mma_layout"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_nvidia_artifactory() -> bool:
    """Return `True` if NVIDIA's artifactory is accessible.

    This checks connectivity to the kernel inference library artifactory
    which is required for downloading certain cubin kernels like TRTLLM FHMA.
    """
    # If we have pre-downloaded cubins, we can assume the cubins are available.
    if has_flashinfer_cubin():
        return True

    try:
        # Use a short timeout to avoid blocking for too long
        response = requests.get(FLASHINFER_CUBINS_REPOSITORY, timeout=5)
        accessible = response.status_code == 200
        if accessible:
            logger.debug_once("NVIDIA artifactory is accessible")
        else:
            logger.warning_once(
                "NVIDIA artifactory returned failed status code: %d",
                response.status_code,
            )
        return accessible
    except Exception as e:
        logger.warning_once("Failed to connect to NVIDIA artifactory: %s", e)
        return False


@functools.cache
def supports_trtllm_attention(is_prefill: bool = False) -> bool:
    """Return whether TRTLLM attention is available on the current platform
    for the given attention phase.

    SM90 (Hopper) and SM12x support the XQA decode kernel but not TRTLLM
    prefill. SM100+ supports TRTLLM for both phases. All others are unsupported.
    """
    # Batch-invariant mode disables TRTLLM attention
    if envs.VLLM_BATCH_INVARIANT:
        return False

    # Requires NVIDIA artifactory to be accessible to download cubins
    if not has_nvidia_artifactory():
        return False

    # SM90 and SM12x have XQA decode only.
    if current_platform.is_device_capability(
        90
    ) or current_platform.is_device_capability_family(120):
        return not is_prefill

    # SM100/SM103 has both prefill and decode TRTLLM kernels.
    return current_platform.is_device_capability_family(100)


def force_use_trtllm_attention() -> bool | None:
    """This function should only be called during initialization stage when vllm config
    is set.
    Return `None` if --attention-config.use_trtllm_attention is not set,
    return `True` if TRTLLM attention is forced to be used,
    return `False` if TRTLLM attention is forced to be not used.
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    return vllm_config.attention_config.use_trtllm_attention


def can_use_trtllm_attention(
    num_qo_heads: int, num_kv_heads: int, is_prefill: bool = False
) -> bool:
    """Check if the current configuration supports TRTLLM attention."""
    if force_use_trtllm_attention() is False:
        return False
    return supports_trtllm_attention(is_prefill=is_prefill) and (
        num_qo_heads % num_kv_heads == 0
    )


def use_trtllm_attention(
    num_qo_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    max_seq_len: int,
    dcp_world_size: int,
    kv_cache_dtype: str,
    q_dtype: torch.dtype,
    is_prefill: bool,
    # None means auto-detection, True means force on, False means force off
    force_use_trtllm: bool | None = None,
    has_sinks: bool = False,
    has_spec: bool = False,
) -> bool:
    """Return `True` if TRTLLM attention is used."""
    # CLI argument is set to 0 - respect it
    if force_use_trtllm is not None and not force_use_trtllm:
        return False

    # TRTLLM prefill attends only the DCP-local KV shard and has no
    # cross-rank LSE combine, so it cannot be used with DCP; fall back to
    # FlashInfer's DCP prefill path. TRTLLM decode under DCP is selected
    # separately (all-gathered query heads + LSE combine in forward).
    if dcp_world_size > 1:
        logger.warning_once(
            "TRTLLM prefill does not support DCP, reverting to FlashInfer"
        )
        return False

    # The platform is not supported
    if not supports_trtllm_attention(is_prefill=is_prefill):
        if force_use_trtllm:
            logger.warning_once(
                "TRTLLM attention is not supported on this platform for %s, "
                "but --attention-config.use_trtllm_attention is set to 1",
                "prefill" if is_prefill else "decode",
            )
        return False

    # The combination of query and key heads is not supported
    if num_qo_heads % num_kv_heads != 0:
        if force_use_trtllm:
            logger.warning_once(
                "TRTLLM attention is not supported for this combination of "
                "query and key heads, but --attention-config.use_trtllm_attention is "
                "set to 1"
            )
        return False

    if has_spec and not is_prefill:
        # Speculative decoding requires TRTLLM attention for decodes
        logger.info_once("Using TRTLLM attention (enabled for speculative decoding).")
        return True

    # Must use TRTLLM attention if query is FP8 quantized
    if q_dtype == current_platform.fp8_dtype():
        logger.info_once("Using TRTLLM attention (query is quantized).")
        return True

    # If sinks are being used, we must use TRTLLM attention as it's
    # the only backend that supports them
    if has_sinks:
        logger.info_once("Using TRTLLM attention (required for attention sinks).")
        return True

    if force_use_trtllm is None:
        # CLI argument not set - use auto-detection
        if is_prefill:
            # Prefill auto-detection
            use_trtllm = kv_cache_dtype == "auto"
        elif (
            current_platform.is_device_capability(90)
            or current_platform.is_device_capability_family(120)
        ) and kv_cache_dtype.startswith("fp8"):
            # SM90/SM12x + FP8 KV cache: prefer the XQA decode kernel.
            use_trtllm = True
        else:
            # Decode auto-detection
            use_trtllm = num_tokens <= 256 and kv_cache_dtype == "auto"
        if use_trtllm:
            logger.warning_once(
                "Using TRTLLM %s attention (auto-detected).",
                "prefill" if is_prefill else "decode",
            )
        return use_trtllm

    # CLI argument is set to 1 - respect it
    logger.info_once(
        "Using TRTLLM attention (--attention-config.use_trtllm_attention is set to 1)"
    )
    return True


if has_flashinfer():
    from vllm.utils.torch_utils import direct_register_custom_op

    def _flashinfer_concat_mla_k(
        k: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        """Custom op wrapper for flashinfer's concat_mla_k.

        This is an in-place operation that concatenates k_nope and k_pe into k.

        The kernel is optimized for DeepSeek V3 dimensions:
        - num_heads=128
        - nope_dim=128
        - rope_dim=64

        Key optimizations:
        - Warp-based processing with software pipelining
        - Vectorized memory access (int2 for nope, int for rope)
        - L2 prefetching for next row while processing current
        - Register reuse for rope values across all heads

        Args:
            k: Output tensor, shape [num_tokens, num_heads, nope_dim + rope_dim].
                Modified in-place.
            k_nope: The nope part of k, shape [num_tokens, num_heads, nope_dim].
            k_pe: The rope part of k (shared), shape [num_tokens, 1, rope_dim].
                  This is broadcast to all heads.

        """
        from flashinfer.concat_ops import concat_mla_k

        concat_mla_k(k, k_nope, k_pe)

    # Register flashinfer concat_mla_k custom op
    direct_register_custom_op(
        op_name="flashinfer_concat_mla_k",
        op_func=_flashinfer_concat_mla_k,
        mutates_args=["k"],  # k tensor is modified in-place
    )

    @torch.library.custom_op(
        "vllm::flashinfer_mm_fp4",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_mm_fp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        use_8x4_sf_layout: bool,
        backend: str,
        block_size: int = 16,
        use_nvfp4: bool = True,
    ) -> torch.Tensor:
        from flashinfer import mm_fp4 as flashinfer_mm_fp4_

        return flashinfer_mm_fp4_(
            A,
            B,
            A_scale,
            B_scale,
            g_scale,
            dtype,
            block_size=block_size,
            use_8x4_sf_layout=use_8x4_sf_layout,
            use_nvfp4=use_nvfp4,
            backend=backend,
        )

    @torch.library.register_fake(
        "vllm::flashinfer_mm_fp4",
    )
    def flashinfer_mm_fp4_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        use_8x4_sf_layout: bool,
        backend: str,
        block_size: int = 16,
        use_nvfp4: bool = True,
    ) -> torch.Tensor:
        return torch.empty(A.shape[0], B.shape[1], dtype=dtype, device=A.device)

    @torch.library.custom_op(
        "vllm::flashinfer_mm_bf16_fp4",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_mm_bf16_fp4(
        A: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        global_scale: torch.Tensor,
    ) -> torch.Tensor:
        from flashinfer.gemm import mm_bf16_fp4

        return mm_bf16_fp4(
            A,
            B,
            B_scale,
            global_scale,
            backend="cute-dsl",
        )

    @torch.library.register_fake(
        "vllm::flashinfer_mm_bf16_fp4",
    )
    def flashinfer_mm_bf16_fp4_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        global_scale: torch.Tensor,
    ) -> torch.Tensor:
        output_size = B.shape[0] if B.dtype == torch.uint8 else B.shape[1] // 2
        return torch.empty(A.shape[0], output_size, dtype=A.dtype, device=A.device)

    @torch.library.custom_op(
        "vllm::flashinfer_mxfp4_quantize",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_mxfp4_quantize(
        a: torch.Tensor,
        backend: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer import mxfp4_quantize as _mxfp4_quantize

        return _mxfp4_quantize(a, backend=backend)

    @torch.library.register_fake("vllm::flashinfer_mxfp4_quantize")
    def flashinfer_mxfp4_quantize_fake(
        a: torch.Tensor,
        backend: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, k = a.shape
        sf_vec_size = 32
        padded_m = cdiv(m, 128) * 128
        sf_cols = cdiv(k // sf_vec_size, 4) * 4
        return (
            torch.empty(m, k // 2, dtype=torch.uint8, device=a.device),
            torch.empty(padded_m, sf_cols, dtype=torch.uint8, device=a.device),
        )

    @torch.library.custom_op(
        "vllm::bmm_fp8",
        mutates_args=[],
        device_types="cuda",
    )
    def bmm_fp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import bmm_fp8 as bmm_fp8_

        return bmm_fp8_(A, B, A_scale, B_scale, dtype, None, backend)

    @torch.library.register_fake(
        "vllm::bmm_fp8",
    )
    def bmm_fp8_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(
            A.shape[0], A.shape[1], B.shape[2], dtype=dtype, device=A.device
        )

    @torch.library.custom_op(
        "vllm::flashinfer_nvfp4_quantize",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_nvfp4_quantize(
        a: torch.Tensor, a_global_sf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer import SfLayout
        from flashinfer import nvfp4_quantize as nvfp4_quantize_

        return nvfp4_quantize_(
            a, a_global_sf, sfLayout=SfLayout.layout_8x4, do_shuffle=False
        )

    @torch.library.register_fake(
        "vllm::flashinfer_nvfp4_quantize",
    )
    def flashinfer_nvfp4_quantize_fake(
        a: torch.Tensor, a_global_sf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = a.shape

        round_up = lambda x, y: (x + y - 1) // y * y

        rounded_m = round_up(m, 8)
        scale_n = n // 16
        rounded_n = round_up(scale_n, 4)

        return torch.empty(m, n // 2, dtype=torch.uint8, device=a.device), torch.empty(
            rounded_m, rounded_n, dtype=torch.uint8, device=a.device
        )

    @torch.library.custom_op(
        "vllm::flashinfer_mxfp8_quantize_8x4",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_mxfp8_quantize_8x4(
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer import SfLayout
        from flashinfer import mxfp8_quantize as mxfp8_quantize_

        return mxfp8_quantize_(
            a,
            backend="cuda",
            sf_swizzle_layout=SfLayout.layout_8x4,
        )

    @torch.library.register_fake(
        "vllm::flashinfer_mxfp8_quantize_8x4",
    )
    def flashinfer_mxfp8_quantize_8x4_fake(
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, k = a.shape
        scale_size = cdiv(m, 8) * 8 * cdiv(k // 32, 4) * 4
        return (
            torch.empty(m, k, dtype=torch.float8_e4m3fn, device=a.device),
            torch.empty(scale_size, dtype=torch.uint8, device=a.device),
        )

    @torch.library.custom_op(
        "vllm::mm_mxfp8",
        mutates_args=[],
        device_types="cuda",
    )
    def mm_mxfp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str = "cutlass",
        use_8x4_sf_layout: bool = False,
    ) -> torch.Tensor:
        from flashinfer import mm_mxfp8 as mm_mxfp8_

        return mm_mxfp8_(
            A,
            B,
            A_scale,
            B_scale,
            out=None,
            out_dtype=out_dtype,
            backend=backend,
            use_8x4_sf_layout=use_8x4_sf_layout,
        )

    @torch.library.register_fake(
        "vllm::mm_mxfp8",
    )
    def mm_mxfp8_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str = "cutlass",
        use_8x4_sf_layout: bool = False,
    ) -> torch.Tensor:
        # A is [m, k], B is [k, n] -> output [m, n]
        return torch.empty(A.shape[0], B.shape[1], dtype=out_dtype, device=A.device)


def flashinfer_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor | None,
    out_dtype: torch.dtype,
    backend: str,
    block_size: int = 16,
    use_nvfp4: bool = True,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert a.shape[1] == b.shape[1]

    if alpha is None:
        alpha = torch.ones(1, dtype=torch.float32, device=a.device)

    if backend in ("cutlass", "cudnn"):
        block_scale_a = block_scale_a.view(torch.uint8)
        block_scale_b = block_scale_b.view(torch.uint8)

    use_8x4_sf_layout = True if backend == "trtllm" and a.shape[0] <= 32 else False  # noqa: SIM210

    return flashinfer_mm_fp4(
        a,
        b.t(),
        block_scale_a,
        block_scale_b.t(),
        alpha,
        out_dtype,
        use_8x4_sf_layout=use_8x4_sf_layout,
        backend=backend,
        block_size=block_size,
        use_nvfp4=use_nvfp4,
    )


def flashinfer_scaled_fp4_mm_out(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    out_dtype: torch.dtype | None,
    use_8x4_sf_layout: bool,
    backend: str,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2 and out.ndim == 2
    assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2
    assert a.stride(-1) == 1
    assert a.shape[1] == b.shape[0]
    assert out.shape == (a.shape[0], b.shape[1])
    assert out.device.type == "cuda"

    if backend in ("cutlass", "cudnn"):
        if block_scale_a.dtype != torch.uint8:
            block_scale_a = block_scale_a.view(torch.uint8)
        if block_scale_b.dtype != torch.uint8:
            block_scale_b = block_scale_b.view(torch.uint8)

    from flashinfer import mm_fp4 as flashinfer_mm_fp4_

    flashinfer_mm_fp4_(
        a,
        b,
        block_scale_a,
        block_scale_b,
        alpha,
        out_dtype or out.dtype,
        out=out,
        block_size=16,
        use_8x4_sf_layout=use_8x4_sf_layout,
        backend=backend,
    )
    return out


def flashinfer_scaled_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert scale_a.numel() == 1 and scale_b.numel() == 1
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a.device.type == "cuda" and b.device.type == "cuda"
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
    assert scale_a.device.type == "cuda" and scale_b.device.type == "cuda"

    output = bmm_fp8(
        a.unsqueeze(0),
        b.unsqueeze(0),
        scale_a,
        scale_b,
        out_dtype,
        "auto",
    ).view(a.shape[0], b.shape[1])

    if bias is not None:
        output = output + bias
    return output


def flashinfer_scaled_fp8_mm_out(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2 and out.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert out.shape == (a.shape[0], b.shape[1])
    assert scale_a.numel() == 1 and scale_b.numel() == 1
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert out.device.type == "cuda"
    assert a.is_contiguous()

    from flashinfer import bmm_fp8 as bmm_fp8_

    bmm_fp8_(
        a.unsqueeze(0),
        # FlashInfer expects the weight in the same column-major view layout
        # consumed by flashinfer_scaled_fp8_mm, so keep the transposed view.
        b.unsqueeze(0),
        scale_a,
        scale_b,
        out_dtype or out.dtype,
        out.unsqueeze(0),
        "auto",
    )
    return out


def flashinfer_quant_nvfp4_8x4_sf_layout(
    a: torch.Tensor, a_global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return flashinfer_nvfp4_quantize(a, a_global_sf)


flashinfer_fp8_blockscale_gemm = _lazy_import_wrapper(
    "flashinfer.gemm", "fp8_blockscale_gemm_sm90"
)


@functools.cache
def has_flashinfer_fp8_blockscale_gemm() -> bool:
    """Return `True` if FlashInfer block-scale FP8 GEMM is available."""
    return (
        has_flashinfer()
        and current_platform.is_device_capability(90)
        and hasattr(_get_submodule("flashinfer.gemm"), "fp8_blockscale_gemm_sm90")
    )


@functools.cache
def is_flashinfer_fp8_blockscale_gemm_supported() -> bool:
    """Return `True` if FlashInfer block-scale FP8 GEMM is supported."""
    return (
        envs.VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER
        and has_flashinfer_fp8_blockscale_gemm()
    )


def should_use_flashinfer_for_blockscale_fp8_gemm(
    is_flashinfer_supported: bool,
    output_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    weight_shape: tuple[int, int],
):
    if not is_flashinfer_supported:
        return False

    # Verify DeepGEMM N/K dims requirements
    # NOTE: Also synchronized with test_w8a8_block_fp8_deep_gemm_matmul
    # test inside kernels/quantization/test_block_fp8.py
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    should_use_flashinfer = (
        output_dtype == torch.bfloat16
        and input_dtype == torch.bfloat16
        and weight_dtype == torch.float8_e4m3fn
        and weight_shape[0] % N_MULTIPLE == 0
        and weight_shape[1] % K_MULTIPLE == 0
    )

    return should_use_flashinfer


_MIN_CUDNN_FP8 = 91701  # cuDNN >= 9.17.1 required for FP8 ViT attention


@functools.cache
def is_flashinfer_cudnn_fp8_prefill_attn_supported() -> bool:
    """Check if FP8 ViT attention is supported on this platform.

    Requires Blackwell (SM 100) or newer, the FlashInfer cuDNN backend,
    and cuDNN >= 9.17.1.

    cuDNN's FP8 SDPA forward path with bf16/fp16 output (used by
    ``MMEncoderAttention._forward_flashinfer``) gates internally on
    ``prop.major >= 10``; on Hopper it raises a misleading
    ``cudnnGraphNotSupportedError: ... cuDNN version 9.13.0 and newer``
    even when the installed cuDNN is new enough. See PR #38065 for the
    original Blackwell-only design intent.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # cuDNN SDPA FP8 with bf16/fp16 output requires Blackwell (SM 100) or newer.
    if not current_platform.has_device_capability(100):
        return False

    try:
        supported = current_platform.get_supported_vit_attn_backends()
        if AttentionBackendEnum.FLASHINFER not in supported:
            return False
    except (ImportError, AttributeError):
        return False

    try:
        import torch.backends.cudnn as cudnn

        if cudnn.is_available() and cudnn.version() < _MIN_CUDNN_FP8:
            return False
    except (ImportError, AttributeError):
        pass

    return True


NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE = 1.0 / (448.0 * 6.0)


def activation_to_flashinfer_int(activation: "MoEActivation") -> int:
    return activation_to_flashinfer_type(activation).value


def has_flashinfer_situ_activation() -> bool:
    try:
        from flashinfer.fused_moe.core import ActivationType
    except ImportError:
        return False
    return hasattr(ActivationType, "Situ")


def activation_to_flashinfer_type(activation: "MoEActivation") -> "ActivationType":
    from flashinfer.fused_moe.core import ActivationType
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    if activation == MoEActivation.SITU:
        situ = getattr(ActivationType, "Situ", None)
        if situ is None:
            raise ValueError("The installed FlashInfer does not support SITU")
        return situ

    # silu and gelu are mapped to their gated versions SwiGLU and GeGLU respectively
    ACTIVATION_TO_FI_ACTIVATION = {
        MoEActivation.SILU_NO_MUL: ActivationType.Silu,
        MoEActivation.GELU_NO_MUL: ActivationType.Gelu,
        MoEActivation.SILU: ActivationType.Swiglu,
        # SwiGLU-OAI uses Swiglu; the OAI alpha/beta/clamp come from gemm1_* args.
        MoEActivation.SWIGLUOAI: ActivationType.Swiglu,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE: ActivationType.Swiglu,
        MoEActivation.GELU: ActivationType.Geglu,
        MoEActivation.GELU_TANH: ActivationType.Geglu,
        MoEActivation.RELU2_NO_MUL: ActivationType.Relu2,
    }
    return ACTIVATION_TO_FI_ACTIVATION[activation]


def quantize_nvfp4_per_token_input(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize NVFP4 activations with one FP32 decode scale per token."""
    from flashinfer import SfLayout, nvfp4_quantize

    return nvfp4_quantize(
        hidden_states,
        NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
    )


def swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)
    )


def rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(
    gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor, is_gated_activation: bool
):
    """Shuffle weights for FI TRT-LLM Format."""
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

    epilogue_tile_m = 128
    num_experts = gemm1_weights.shape[0]
    hidden_size = gemm1_weights.shape[-1]
    intermediate_size = gemm1_weights.shape[1] // 2

    # Reorder rows of W1 for fused gated activation
    gemm1_weights_fp8_interleaved = []
    for i in range(num_experts):
        gemm1_weights_fp8_interleaved.append(
            reorder_rows_for_gated_act_gemm(gemm1_weights[i])
            if is_gated_activation
            else gemm1_weights[i]
        )

    # Stack weights and scales for all experts
    gemm1_weights_fp8_interleaved = torch.stack(gemm1_weights_fp8_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_fp8_shuffled = []
    gemm2_weights_fp8_shuffled = []
    for i in range(num_experts):
        gemm1_weights_fp8_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp8_interleaved[i].view(torch.uint8), epilogue_tile_m
            )
        )

        gemm2_weights_fp8_shuffled.append(
            shuffle_matrix_a(gemm2_weights[i].view(torch.uint8), epilogue_tile_m)
        )

    # Stack weights for all experts
    gemm1_weights.data = torch.stack(gemm1_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )
    gemm2_weights.data = torch.stack(gemm2_weights_fp8_shuffled).view(
        torch.float8_e4m3fn
    )


def convert_moe_weights_to_flashinfer_trtllm_block_layout(
    cache_permute_indices: dict[torch.Size, torch.Tensor],
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    is_gated_act_gemm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert expert weights to FlashInfer's block layout.

    This reorders W13 and W2 in place into the expected epilogue-tiled block
    layout and returns views of the shuffled weight tensors. Using one expert
    as scratch space avoids allocating another full copy of both weights.
    """
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise ValueError(
            "Unquantized Moe Backend FlashInfer TRTLLM requires bfloat16 weights"
        )

    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    block_k = 128

    # Reorder rows of W13 and W2 for fused gated activation and convert to the
    # block layout expected by the FlashInfer kernel.
    num_experts = w13_weight.shape[0]

    def _copy_permuted_expert_to_block_layout(
        out: torch.Tensor,
        expert_uint8: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> None:
        expert_blocks = expert_uint8.view(
            expert_uint8.shape[0], out.shape[0], block_k
        ).permute(1, 0, 2)
        torch.index_select(
            expert_blocks,
            1,
            source_indices.to(expert_uint8.device),
            out=out,
        )

    def _convert_weight_in_place(
        weight: torch.Tensor,
        is_w13: bool,
    ) -> torch.Tensor:
        rows, cols = weight[0].view(torch.uint8).shape
        block_layout_shape = (num_experts, cols // block_k, rows, block_k)
        expert_scratch = torch.empty(
            block_layout_shape[1:],
            dtype=torch.uint8,
            device=weight.device,
        )

        for i in range(num_experts):
            expert_uint8 = weight[i].view(torch.uint8)
            if is_w13:
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    cache_permute_indices,
                    expert_uint8,
                    epilogue_tile_m,
                    is_gated_act_gemm=is_gated_act_gemm,
                )
                if is_gated_act_gemm:
                    permute_indices = (
                        permute_indices + expert_uint8.shape[0] // 2
                    ) % expert_uint8.shape[0]
            else:
                permute_indices = get_w2_permute_indices_with_cache(
                    cache_permute_indices,
                    expert_uint8,
                    epilogue_tile_m,
                )

            _copy_permuted_expert_to_block_layout(
                expert_scratch,
                expert_uint8,
                permute_indices,
            )
            expert_uint8.view(-1).copy_(expert_scratch.view(-1))

        return weight.view(torch.uint8).view(block_layout_shape).view(torch.bfloat16)

    return (
        _convert_weight_in_place(w13_weight, is_w13=True),
        _convert_weight_in_place(w2_weight, is_w13=False),
    )


def align_fp4_moe_weights_for_fi(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    is_act_and_mul: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer kernels' alignment constraints hold.

    Some FlashInfer FP4 MoE kernels require the intermediate size
    used for GEMM to be divisible by a small alignment value. When this is
    not satisfied (e.g. with certain tensor-parallel sizes), we pad the
    gate/up and down projection weights along the intermediate dim.
    """
    # Current local intermediate size (per partition) is the K dimension of
    # the down projection.
    num_experts, hidden_size, intermediate = w2.shape
    intermediate *= 2  # because of packed FP4

    padded_intermediate = round_up(intermediate, min_alignment)

    if padded_intermediate == intermediate:
        return w13, w13_scale, w2, w2_scale, intermediate

    logger.info_once(
        "Padding intermediate size from %d to %d for up/down projection weights.",
        intermediate,
        padded_intermediate,
    )

    up_mult = 2 if is_act_and_mul else 1
    padded_gate_up_dim = up_mult * padded_intermediate

    # Pad w13 and w2 along its intermediate dimension.
    padded_w13 = w13.new_zeros((num_experts, padded_gate_up_dim, hidden_size // 2))
    if is_act_and_mul:
        # Keep the two logical projections independently aligned. Copying the
        # fused [gate, up] tensor contiguously would move the up projection into
        # the padded tail of gate when intermediate is rounded up.
        padded_w13[:, :intermediate, :] = w13[:, :intermediate, :]
        padded_w13[:, padded_intermediate : padded_intermediate + intermediate, :] = (
            w13[:, intermediate:, :]
        )
    else:
        padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate // 2))
    padded_w2[:, :, : w2.shape[2]] = w2

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, padded_gate_up_dim, hidden_size // 16)
    )
    if is_act_and_mul:
        padded_w13_scale[:, :intermediate, :] = w13_scale[:, :intermediate, :]
        padded_w13_scale[
            :, padded_intermediate : padded_intermediate + intermediate, :
        ] = w13_scale[:, intermediate:, :]
    else:
        padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale

    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, hidden_size, padded_intermediate // 16)
    )
    padded_w2_scale[:, :, : w2_scale.shape[2]] = w2_scale

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_intermediate


def align_fp4_moe_hidden_dim_for_fi(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    min_alignment: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_experts, gate_up_dim, packed_hidden_size = w13.shape
    hidden_size = packed_hidden_size * 2
    padded_hidden_size = round_up(hidden_size, min_alignment)

    if padded_hidden_size == hidden_size:
        return w13, w13_scale, w2, w2_scale, hidden_size

    logger.warning_once(
        "Padding hidden size from %d to %d for FlashInfer NVFP4 MoE weights. "
        "This requires activation slicing at runtime and may cause "
        "performance degradation.",
        hidden_size,
        padded_hidden_size,
    )

    padded_w13 = w13.new_zeros((num_experts, gate_up_dim, padded_hidden_size // 2))
    padded_w13[:, :, :packed_hidden_size] = w13

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, gate_up_dim, padded_hidden_size // 16)
    )
    padded_w13_scale[:, :, : w13_scale.shape[2]] = w13_scale

    padded_w2 = w2.new_zeros((num_experts, padded_hidden_size, w2.shape[2]))
    padded_w2[:, : w2.shape[1], :] = w2

    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, padded_hidden_size, w2_scale.shape[2])
    )
    padded_w2_scale[:, : w2_scale.shape[1], :] = w2_scale

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_hidden_size


def align_moe_weights_for_fi(
    w13: torch.Tensor, w2: torch.Tensor, is_act_and_mul: bool, min_alignment: int = 16
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer kernels' alignment constraints hold.

    Some FlashInfer MoE kernels require the (gated) intermediate size
    used for GEMM to be divisible by a small alignment value. When this is
    not satisfied (e.g. with certain tensor-parallel sizes), we pad the
    gate/up and down projection weights along the intermediate dim.
    """
    # Current local intermediate size (per partition) is the K dimension of
    # the down projection.
    num_experts, hidden_size, intermediate = w2.shape

    padded_intermediate = round_up(intermediate, min_alignment)

    if padded_intermediate == intermediate:
        return w13, w2, intermediate

    logger.info_once(
        "Padding intermediate size from %d to %d for up/down projection weights.",
        intermediate,
        padded_intermediate,
    )

    up_mult = 2 if is_act_and_mul else 1
    padded_gate_up_dim = up_mult * padded_intermediate

    # Pad w13 and w2 along its intermediate dimension.
    padded_w13 = w13.new_zeros((num_experts, padded_gate_up_dim, hidden_size))
    if is_act_and_mul:
        padded_w13[:, :intermediate, :] = w13[:, :intermediate, :]
        padded_w13[:, padded_intermediate : padded_intermediate + intermediate, :] = (
            w13[:, intermediate:, :]
        )
    else:
        padded_w13[:, :intermediate, :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))
    padded_w2[:, :, :intermediate] = w2

    return padded_w13, padded_w2, padded_intermediate


def _shuffle_deepseek_fp8_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess DeepSeek FP8 block-scale weights for the FlashInfer TRT-LLM
    kernel using the shuffle + BlockMajorK layout variant.

    Returns 4D weight tensors in BlockMajorK layout
    (E, K/block_k, Mn, block_k)
    """
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    epilogue_tile_m = 64
    block_k = 128

    def shuffle_to_block_major_k(w: torch.Tensor) -> torch.Tensor:
        # shuffle_matrix_a's row permutation depends only on (M,
        # epilogue_tile_m), so it is computed once and applied to every expert
        # in a single gather instead of once per expert. Gathering through the
        # BlockMajorK-permuted view also folds convert_to_block_layout into
        # that same kernel. Per-expert loops here cost minutes for a MoE this
        # wide (~24k tiny launches plus a host round-trip each).
        num_experts, m, k = w.shape
        rows = get_shuffle_matrix_a_row_indices(
            w[0].view(torch.uint8), epilogue_tile_m
        ).to(w.device)
        blocked = w.view(torch.uint8).view(num_experts, m, k // block_k, block_k)
        out = blocked.permute(0, 2, 1, 3)[:, :, rows, :].contiguous()
        return out.view(torch.float8_e4m3fn)

    return shuffle_to_block_major_k(w13), shuffle_to_block_major_k(w2)


def _shuffle_mxfp8_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess MXFP8 weights and scales for the FlashInfer TRT-LLM kernel.

    All three transforms (gate/up row reorder, ``shuffle_matrix_a`` weight
    shuffle, ``shuffle_matrix_sf_a`` scale shuffle) are fixed row/index
    permutations that depend only on the per-expert matrix shape, so the
    permutation is computed once and applied to every expert in a single
    gather instead of once per expert. ``block_scale_interleave`` accepts a
    batched ``(E, M, K)`` scale tensor directly. Output is bit-identical to
    the per-expert loop but ~20x faster (a 288-expert MoE otherwise costs
    seconds per layer at load).
    """
    from flashinfer.fused_moe.core import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
    )
    from flashinfer.quantization.fp4_quantization import block_scale_interleave
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    epilogue_tile_m = 128

    w13_u = w13.view(torch.uint8)
    w2_u = w2.view(torch.uint8)

    # 1. Interleave gate/up rows (gated activation GEMM layout).
    if is_gated:
        gate_idx = get_reorder_rows_for_gated_act_gemm_row_indices(
            w13_u[0].reshape(w13_u.shape[1], -1)
        )
        w13_u = w13_u[:, gate_idx]
        w13_scale = w13_scale[:, gate_idx]

    def shuffle_weights(t: torch.Tensor) -> torch.Tensor:
        # Row permutation depends only on (M, epilogue_tile_m).
        idx = get_shuffle_matrix_a_row_indices(t[0], epilogue_tile_m).to(t.device)
        return t[:, idx].view(torch.float8_e4m3fn)

    def shuffle_scales(s: torch.Tensor) -> torch.Tensor:
        # shuffle_matrix_sf_a == row-gather (same indices as the weight shuffle)
        # followed by the 128x4 block-scale interleave, which is batch-capable.
        idx = get_shuffle_matrix_a_row_indices(
            s[0].view(torch.uint8).reshape(s.shape[1], -1), epilogue_tile_m
        ).to(s.device)
        interleaved = block_scale_interleave(s[:, idx])
        return interleaved.reshape(s.shape)

    return (
        shuffle_weights(w13_u),
        shuffle_weights(w2_u),
        shuffle_scales(w13_scale),
        shuffle_scales(w2_scale),
    )


def prepare_fp8_moe_layer_for_fi(
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_scale: torch.Tensor,
    w2_input_scale: torch.Tensor | None,
    is_trtllm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Fp8 MoE weights to flashinfer kernel format.

    Note that for trtllm we update the model state dict
    with the scale format needed for these kernels.

    Note that for per-tensor, we update the layer's
    intermediate size if the weights needed padding.
    """
    assert hasattr(layer.moe_config, "is_act_and_mul")
    block_quant = (
        hasattr(layer, "weight_block_size") and layer.weight_block_size is not None
    )
    is_mxfp8 = block_quant and w13_scale.dtype == torch.uint8
    is_deepseek_fp8 = block_quant and not is_mxfp8
    is_gated = layer.activation.is_gated

    # MXFP8 TRT-LLM requires W31 swap + reorder + shuffle.
    if is_mxfp8 and is_trtllm:
        # FlashInfer TRT-LLM SwiGLU expects [up; gate] but vLLM stores
        # [gate; up].  Swap both weights and scales before interleaving.
        if layer.moe_config.is_act_and_mul:
            w13 = swap_w13_to_w31(w13)
            # Scales may be 2D [E, flat] from _quantize_mxfp8_moe_weight;
            # reshape to 3D so swap_w13_to_w31 can flip the two halves,
            # then flatten back.
            if w13_scale.ndim == 2:
                num_rows = w13.shape[1]  # 2 * intermediate_size
                w13_scale = w13_scale.reshape(w13_scale.shape[0], num_rows, -1)
                w13_scale = swap_w13_to_w31(w13_scale)
                w13_scale = w13_scale.reshape(w13_scale.shape[0], -1)
            else:
                w13_scale = swap_w13_to_w31(w13_scale)

        w13, w2, w13_scale, w2_scale = _shuffle_mxfp8_moe_weights(
            w13, w2, w13_scale, w2_scale, is_gated
        )
        return w13, w2, w13_scale, w2_scale

    # Some FI MoE kernels require internal alignment of 16
    # for the gate-up proj. Pad the weights to respect this.
    if not block_quant:
        min_alignment = 16 if is_gated else 128
        w13, w2, new_intermediate = align_moe_weights_for_fi(
            w13,
            w2,
            layer.moe_config.is_act_and_mul,
            min_alignment,
        )
        layer.moe_config.intermediate_size_per_partition = new_intermediate

    # FI kernels require W31 layout rather than W13.
    if layer.moe_config.is_act_and_mul:
        w13 = swap_w13_to_w31(w13)
        if block_quant:
            w13_scale = swap_w13_to_w31(w13_scale)

    # DeepSeekFp8 TRT-LLM: shuffle weights into BlockMajorK layout.
    if is_deepseek_fp8 and is_trtllm:
        w13, w2 = _shuffle_deepseek_fp8_moe_weights(w13, w2)

    # FI TRT-LLM FP8 per-tensor MoE kernel requires weight shuffle
    # and registration of alpha scales.
    if is_trtllm and not block_quant:
        assert w13_input_scale is not None
        assert w2_input_scale is not None

        rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(w13, w2, is_gated)

    # Clamp block scales to avoid NaN from the FlashInfer CUTLASS kernel.
    # Some FP8 models have near-zero block scales (~1e-23) for dead/unused
    # experts. The CUTLASS kernel doesn't handle these correctly on Hopper
    # (SM 9.0), producing NaN instead of near-zero output. Clamping to a
    # small minimum prevents this without affecting model accuracy since
    # these experts' effective weights are already zero.
    if block_quant:
        _FI_CUTLASS_MIN_BLOCK_SCALE = 1e-10
        w13_scale.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)
        w2_scale.clamp_(min=_FI_CUTLASS_MIN_BLOCK_SCALE)

    return w13, w2, w13_scale, w2_scale


__all__ = [
    "has_flashinfer",
    "flashinfer_bf16_mm",
    "autotune_bf16_only",
    "has_flashinfer_bf16_gemm",
    "is_flashinfer_bf16_gemm_supported",
    "is_flashinfer_cutedsl_bf16_gemm_supported",
    "flashinfer_trtllm_fp8_block_scale_moe",
    "flashinfer_cutlass_fused_moe",
    "flashinfer_cutedsl_grouped_gemm_nt_masked",
    "flashinfer_prepare_bf16_fp4_weights",
    "flashinfer_fp4_quantize",
    "silu_and_mul_scaled_nvfp4_experts_quantize",
    "scaled_fp4_grouped_quantize",
    "nvfp4_block_scale_interleave",
    "flashinfer_cute_dsl_fused_moe_nvfp4",
    "flashinfer_b12x_fused_moe",
    "flashinfer_get_hybrid_num_tokens_buckets",
    "flashinfer_convert_sf_to_mma_layout",
    "trtllm_fp4_block_scale_moe",
    "flashinfer_trtllm_batch_decode_with_kv_cache_mla",
    "flashinfer_trtllm_batch_decode_sparse_mla_dsv4",
    "flashinfer_xqa_batch_decode_with_kv_cache",
    "flashinfer_recurrent_kda",
    "flashinfer_fused_kda_decode",
    "autotune",
    "has_flashinfer_moe",
    "has_flashinfer_comm",
    "has_flashinfer_nvlink_two_sided",
    "has_flashinfer_nvlink_one_sided",
    "has_flashinfer_cutlass_fused_moe",
    "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
    "has_flashinfer_recurrent_kda",
    "has_flashinfer_fused_kda_decode",
    "has_flashinfer_cutedsl_moe_nvfp4",
    "has_flashinfer_bf16_fp4",
    "has_flashinfer_b12x_moe",
    "has_flashinfer_b12x_gemm",
    "has_flashinfer_fp8_blockscale_gemm",
    "has_nvidia_artifactory",
    "supports_trtllm_attention",
    "can_use_trtllm_attention",
    "use_trtllm_attention",
    "flashinfer_mxfp4_quantize",
    "flashinfer_scaled_fp4_mm",
    "flashinfer_scaled_fp4_mm_out",
    "flashinfer_scaled_fp8_mm",
    "flashinfer_scaled_fp8_mm_out",
    "flashinfer_quant_nvfp4_8x4_sf_layout",
    "flashinfer_fp8_blockscale_gemm",
    "should_use_flashinfer_for_blockscale_fp8_gemm",
    "is_flashinfer_fp8_blockscale_gemm_supported",
    "is_flashinfer_cudnn_fp8_prefill_attn_supported",
    "NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE",
    "activation_to_flashinfer_int",
    "has_flashinfer_situ_activation",
    "activation_to_flashinfer_type",
    "quantize_nvfp4_per_token_input",
    "swap_w13_to_w31",
    "rotate_weights_for_fi_trtllm_fp8_per_tensor_moe",
    "convert_moe_weights_to_flashinfer_trtllm_block_layout",
    "align_fp4_moe_weights_for_fi",
    "align_fp4_moe_hidden_dim_for_fi",
    "align_moe_weights_for_fi",
    "_shuffle_deepseek_fp8_moe_weights",
    "_shuffle_mxfp8_moe_weights",
    "prepare_fp8_moe_layer_for_fi",
]
