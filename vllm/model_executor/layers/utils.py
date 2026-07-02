# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

import contextlib
import itertools
import threading
from collections.abc import Callable
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MOE_LAYER_ROUTER_GATE_SUFFIXES = {
    "gate",
    "router",
    "router_gate",
    "shared_expert_gate",
    "expert_gate",
}


def is_layer_moe_router_gate(prefix: str) -> bool:
    if not prefix:
        return False
    return prefix.rsplit(".", 1)[-1] in MOE_LAYER_ROUTER_GATE_SUFFIXES


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


_DYNAMO_CACHE_SIZE_LIMIT = 65536
_DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT = 1048576
_VLLM_ROOT = Path(__file__).resolve().parents[3]
_INDUCTOR_CUTLASS_DIR_FALLBACKS = (
    _VLLM_ROOT.parent / "pytorch" / "third_party" / "cutlass",
)

_inductor_max_autotune_gemm_forced = False
_dynamo_compile_caches_forced = False
_unquant_bf16_linear_cache: dict[tuple, Callable] = {}
_unquant_bf16_linear_lock = threading.Lock()
_unquant_bf16_linear_id_gen = itertools.count()
_unquant_bf16_linear_capture_safe_keys: set[tuple] = set()
_unquant_bf16_linear_torch_compile_configured = False
_unquant_bf16_linear_torch_compile_disabled = False


def _set_inductor_cutlass_dir_if_needed(inductor_config) -> None:
    cuda_config = getattr(inductor_config, "cuda", None)
    if cuda_config is None or not hasattr(cuda_config, "cutlass_dir"):
        return

    configured_cutlass_dir = getattr(cuda_config, "cutlass_dir", None)
    if configured_cutlass_dir:
        with contextlib.suppress(TypeError, OSError):
            if Path(configured_cutlass_dir).exists():
                return

    for fallback in _INDUCTOR_CUTLASS_DIR_FALLBACKS:
        if fallback.exists():
            resolved_fallback = str(fallback.resolve())
            cuda_config.cutlass_dir = resolved_fallback
            cutlass_config = getattr(inductor_config, "cutlass", None)
            if cutlass_config is not None and hasattr(cutlass_config, "cutlass_dir"):
                cutlass_config.cutlass_dir = resolved_fallback
            logger.info(
                "Set inductor CUTLASS source directory to %s", resolved_fallback
            )
            return


def force_large_dynamo_compile_caches() -> None:
    """Raise Dynamo cache limits for many per-shape compiled linear kernels."""
    global _dynamo_compile_caches_forced
    if _dynamo_compile_caches_forced:
        return
    _dynamo_compile_caches_forced = True

    try:
        import torch._dynamo.config as dynamo_config
    except Exception:
        logger.warning(
            "torch._dynamo.config could not be imported; "
            "leaving Dynamo cache limits untouched.",
            exc_info=True,
        )
        return

    if hasattr(dynamo_config, "cache_size_limit"):
        dynamo_config.cache_size_limit = max(
            dynamo_config.cache_size_limit,
            _DYNAMO_CACHE_SIZE_LIMIT,
        )
    dynamo_config.accumulated_cache_size_limit = max(
        getattr(dynamo_config, "accumulated_cache_size_limit", 0),
        _DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT,
    )
    if hasattr(dynamo_config, "recompile_limit"):
        dynamo_config.recompile_limit = max(
            dynamo_config.recompile_limit,
            _DYNAMO_CACHE_SIZE_LIMIT,
        )
    if hasattr(dynamo_config, "accumulated_recompile_limit"):
        dynamo_config.accumulated_recompile_limit = max(
            dynamo_config.accumulated_recompile_limit,
            _DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT,
        )
    if hasattr(dynamo_config, "fail_on_recompile_limit_hit"):
        dynamo_config.fail_on_recompile_limit_hit = False
    if hasattr(dynamo_config, "fail_on_cache_limit_hit"):
        dynamo_config.fail_on_cache_limit_hit = False


def force_inductor_max_autotune_gemm_on_small_gpus() -> None:
    """Enable inductor GEMM autotuning for the BF16 linear compile path."""
    global _inductor_max_autotune_gemm_forced
    if _inductor_max_autotune_gemm_forced:
        return
    _inductor_max_autotune_gemm_forced = True

    force_large_dynamo_compile_caches()

    try:
        import torch._inductor.config as inductor_config
        import torch._inductor.utils as inductor_utils
    except Exception:
        logger.warning(
            "torch._inductor could not be imported; leaving inductor config untouched.",
            exc_info=True,
        )
        return

    original_big_gpu = getattr(inductor_utils, "is_big_gpu", None)
    if original_big_gpu is not None and not getattr(
        original_big_gpu, "__vllm_big_gpu_override__", False
    ):

        def _forced_big_gpu(*args, **kwargs) -> bool:
            return True

        _forced_big_gpu.__vllm_big_gpu_override__ = True
        inductor_utils.is_big_gpu = _forced_big_gpu
        cache_clear = getattr(original_big_gpu, "cache_clear", None)
        if callable(cache_clear):
            with contextlib.suppress(Exception):
                cache_clear()

    try:
        from torch._inductor.codegen.cuda import cuda_env

        original_blackwell = getattr(cuda_env, "is_datacenter_blackwell_arch", None)
        if original_blackwell is not None and not getattr(
            original_blackwell, "__vllm_blackwell_family_override__", False
        ):

            def _forced_blackwell_family_arch() -> bool:
                if original_blackwell():
                    return True
                if torch.cuda.is_available():
                    major, _ = torch.cuda.get_device_capability()
                    return major >= 10
                return False

            _forced_blackwell_family_arch.__vllm_blackwell_family_override__ = True
            cuda_env.is_datacenter_blackwell_arch = _forced_blackwell_family_arch
            cache_clear = getattr(original_blackwell, "cache_clear", None)
            if callable(cache_clear):
                with contextlib.suppress(Exception):
                    cache_clear()
    except Exception:
        logger.debug("Could not patch inductor Blackwell architecture helper.")

    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm = True
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
    inductor_config.coordinate_descent_tuning = True
    if hasattr(inductor_config, "triton"):
        inductor_config.triton.unique_kernel_names = True
        if hasattr(inductor_config.triton, "enable_persistent_tma_matmul"):
            inductor_config.triton.enable_persistent_tma_matmul = True
    if hasattr(inductor_config, "fx_graph_cache"):
        inductor_config.fx_graph_cache = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Ensure CUTLASS source dir is set so the inductor backend can find it.
    # The default points to ../third_party/cutlass/ (developer source tree),
    # which does not exist in pip-installed or container environments.
    _set_inductor_cutlass_dir_if_needed(inductor_config)


def _configure_unquant_bf16_linear_torch_compile_once() -> None:
    global _unquant_bf16_linear_torch_compile_configured
    if _unquant_bf16_linear_torch_compile_configured:
        return

    force_inductor_max_autotune_gemm_on_small_gpus()
    _unquant_bf16_linear_torch_compile_configured = True


def _unquant_bf16_linear_cache_key(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple:
    return (
        tuple(x.shape),
        tuple(x.stride()),
        x.dtype,
        tuple(weight.shape),
        tuple(weight.stride()),
        weight.dtype,
        None if bias is None else tuple(bias.shape),
        None if bias is None else tuple(bias.stride()),
        None if bias is None else bias.dtype,
        x.device.type,
        x.device.index,
    )


def _compile_unquant_bf16_linear_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> Callable:
    _configure_unquant_bf16_linear_torch_compile_once()
    compile_id = next(_unquant_bf16_linear_id_gen)
    name_suffix = f"_{compile_id}"

    if bias is None:

        def _linear_no_bias(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(x, weight, None)

        _linear_no_bias.__name__ += name_suffix
        compiled = torch.compile(
            _linear_no_bias,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
        compiled(x, weight)
        return compiled

    def _linear_with_bias(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, weight, bias)

    _linear_with_bias.__name__ += name_suffix
    compiled = torch.compile(
        _linear_with_bias,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    compiled(x, weight, bias)
    return compiled


def _get_or_create_unquant_bf16_linear_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    allow_new_compile: bool,
) -> Callable | None:
    key = _unquant_bf16_linear_cache_key(x, weight, bias)
    compiled = _unquant_bf16_linear_cache.get(key)
    if compiled is not None:
        return compiled
    if not allow_new_compile:
        return None

    compiled = _compile_unquant_bf16_linear_kernel(x, weight, bias)
    _unquant_bf16_linear_cache[key] = compiled
    return compiled


def _should_use_unquant_bf16_linear_torch_compile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    if _unquant_bf16_linear_torch_compile_disabled:
        return False
    if torch._dynamo.is_compiling():
        return False
    if not x.is_cuda:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if bias is not None and bias.dtype != torch.bfloat16:
        return False
    return x.is_contiguous()


def _apply_unquant_bf16_linear_torch_compile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    global _unquant_bf16_linear_torch_compile_disabled

    key = _unquant_bf16_linear_cache_key(x, weight, bias)
    is_capturing = torch.cuda.is_current_stream_capturing()
    if is_capturing and key not in _unquant_bf16_linear_capture_safe_keys:
        return None

    compiled = _get_or_create_unquant_bf16_linear_kernel(
        x,
        weight,
        bias,
        allow_new_compile=not is_capturing,
    )
    if compiled is None:
        return None

    try:
        output = compiled(x, weight) if bias is None else compiled(x, weight, bias)
    except Exception:
        if is_capturing:
            return None
        logger.warning(
            "Disabling compiled BF16 linear fast path after torch.compile failure.",
            exc_info=True,
        )
        _unquant_bf16_linear_torch_compile_disabled = True
        return None

    if not is_capturing:
        with _unquant_bf16_linear_lock:
            _unquant_bf16_linear_capture_safe_keys.add(key)
    return output


def _torch_compile_bf16_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    if _should_use_unquant_bf16_linear_torch_compile(x, weight, bias):
        output = _apply_unquant_bf16_linear_torch_compile(x, weight, bias)
        if output is not None:
            return output
    return torch.nn.functional.linear(x, weight, bias)


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


def use_aiter_triton_gemm(n, m, k, dtype):
    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
    )


def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950

    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]

    cu_count = num_compute_units()

    # Next ^2 of n
    N_p2 = 1 << (n - 1).bit_length()
    # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
    # and each working on a 512-shard of K, how many CUs would we need?
    rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
    # How many of 4 waves in a group can work on same 16 Ms at same time?
    # This reduces the Ms each group works on, i.e. increasing the number of CUs needed.
    GrpsShrB = min(N_p2 // 16, 4)
    # Given the above, how many CUs would we need?
    CuNeeded = rndup_cus * GrpsShrB
    # candidate for atomic reduce count splitk?
    fits_wvsplitkrc = (
        N_p2 * m * ((k + 512 - 1) // 512)
    ) <= 128 * 1024 * 12  # deterministic
    fits_wvsplitkrc &= CuNeeded <= cu_count

    use_skinny_reduce_counting = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and on_gfx950()
        and x.dtype in [torch.float16, torch.bfloat16]
        and x.dim() == 2
        and (
            10 <= n <= 128
            and k % 8 == 0
            and k > 512
            and m % 16 == 0
            and fits_wvsplitkrc
            and weight.is_contiguous()
        )
    )

    if use_skinny_reduce_counting:
        return ops.wvSplitKrc(x, weight, cu_count, bias)

    if use_aiter_triton_gemm(n, m, k, x.dtype):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        return gemm_a16w16(x, weight, bias)

    use_skinny = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and (on_gfx9() or on_gfx1x())
        and x.dtype in [torch.float16, torch.bfloat16]
        and k % 8 == 0
    )

    if use_skinny:
        x_view = x.reshape(-1, x.size(-1))
        if m > 8 and 0 < n <= 5:
            cu_count = num_compute_units()
            out = ops.wvSplitK(weight, x_view, cu_count, bias)
            return out.reshape(*x.shape[:-1], weight.shape[0])
        elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:
            out = ops.LLMM1(weight, x_view, 4)
            return out.reshape(*x.shape[:-1], weight.shape[0])

    if rocm_aiter_ops.is_tgemm_enabled():
        from aiter.tuned_gemm import tgemm

        return tgemm.mm(x, weight, bias)

    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm",
    op_func=rocm_unquantized_gemm_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    return (
        torch.cpu._is_amx_tile_supported()
        and (dtype in (torch.bfloat16, torch.int8))
        and k % 32 == 0
        and n % 16 == 0
    )


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    # skip for missing layers
    if layer.weight.is_meta:
        layer.cpu_linear = torch.nn.functional.linear
        return

    if layer.weight.ndim != 2:
        # this is not a linear layer
        # For now it should be a causal_conv1d op
        if torch.cpu._is_amx_tile_supported():
            # prepack conv weight
            layer.weight.data = ops.causal_conv1d_weight_pack(
                layer.weight.view(
                    layer.weight.size(0),
                    layer.weight.size(2),
                )
            )
        return

    N, K = layer.weight.size()
    dtype = layer.weight.dtype

    # Zen CPU path: zentorch_linear_unary with optional eager weight prepacking.
    if current_platform.is_zen_cpu() and hasattr(
        torch.ops.zentorch, "zentorch_linear_unary"
    ):
        zen_weight = layer.weight.detach()
        is_prepacked = False

        if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
            torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
        ):
            zen_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                zen_weight
            )
            is_prepacked = True

        layer.cpu_linear = lambda x, weight, bias, _p=is_prepacked: (
            torch.ops.zentorch.zentorch_linear_unary(
                x, zen_weight, bias, is_weight_prepacked=_p
            )
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
            is_prepacked,
        )
        return

    if envs.VLLM_CPU_SGL_KERNEL and check_cpu_sgl_kernel(N, K, dtype):
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            bias_f32 = layer.bias.to(torch.float32)
        else:
            bias_f32 = None
        layer.cpu_linear = lambda x, weight, bias: torch.ops._C.weight_packed_linear(
            x, packed_weight, bias_f32 if bias is not None else None, True
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using sgl-kernel weight_packed_linear"
        )
        return
    elif (
        ops._supports_onednn
        and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
    ):
        try:
            origin_weight = layer.weight
            handler = ops.create_onednn_mm(origin_weight.t(), 32)
            layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias)
            if remove_weight:
                layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            logger.debug_once("CPU unquantized GEMM dispatch: using oneDNN onednn_mm")
            return
        except RuntimeError as e:
            logger.warning_once(
                "Failed to create oneDNN linear, fallback to torch linear."
                f" Exception: {e}"
            )

    # fallback case
    layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
        x, weight, bias
    )
    logger.debug_once(
        "CPU unquantized GEMM dispatch: using torch.nn.functional.linear (fallback)"
    )


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    elif (
        envs.VLLM_ENABLE_UNQUANT_BF16_LINEAR_TORCH_COMPILE
        and current_platform.is_cuda()
    ):
        return _torch_compile_bf16_unquantized_gemm
    else:
        return default_unquantized_gemm
