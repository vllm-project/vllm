# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

from collections.abc import Callable

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


def _tinygemm_bf16_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Real implementation: calls FlashInfer tinygemm via lazy wrapper."""
    from vllm.utils.flashinfer import flashinfer_tinygemm_bf16

    out = torch.empty(
        input.shape[0],
        weight.shape[0],
        dtype=torch.bfloat16,
        device=input.device,
    )
    flashinfer_tinygemm_bf16(input, weight, out, bias=bias)
    return out


def _tinygemm_bf16_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation for torch.compile graph tracing."""
    return torch.empty(
        input.shape[0],
        weight.shape[0],
        dtype=torch.bfloat16,
        device=input.device,
    )


_TINYGEMM_AVAILABLE = False


def _init_tinygemm():
    """Register tinygemm custom op if FlashInfer is available on SM90+."""
    global _TINYGEMM_AVAILABLE
    try:
        from vllm.utils.flashinfer import has_flashinfer

        if not has_flashinfer():
            return
        capability = current_platform.get_device_capability()
        if capability is None or capability[0] < 9:
            return
        direct_register_custom_op(
            "tinygemm_bf16",
            _tinygemm_bf16_impl,
            fake_impl=_tinygemm_bf16_fake,
        )
        _TINYGEMM_AVAILABLE = True
    except Exception:
        pass


_init_tinygemm()


_inductor_big_gpu_override_applied = False


def _maybe_override_inductor_is_big_gpu() -> None:
    """Bypass inductor's 68-SM gate for max-autotune-gemm Triton templates.

    `torch._inductor.utils.is_big_gpu` compares `multi_processor_count`
    against a hard-coded 68 (3080 baseline). On sub-68-SM CUDA devices
    (notably GB10 / DGX Spark with 48 SMs) this causes max-autotune-gemm
    to silently drop Triton templates from the autotune pool, even when
    the user explicitly includes them in
    `config.max_autotune_gemm_backends`.

    When `VLLM_INDUCTOR_OVERRIDE_BIG_GPU=1`, monkey-patch the helper to
    always return True. The patch is global (affects any torch.compile
    call in the process) and one-shot. Must run before the first
    inductor autotune call, since `is_big_gpu` is `@functools.cache`d.
    """
    global _inductor_big_gpu_override_applied
    if _inductor_big_gpu_override_applied:
        return
    if not envs.VLLM_INDUCTOR_OVERRIDE_BIG_GPU:
        return
    try:
        import torch._inductor.utils as inductor_utils
    except Exception:
        logger.warning(
            "VLLM_INDUCTOR_OVERRIDE_BIG_GPU set but torch._inductor.utils "
            "could not be imported; leaving is_big_gpu untouched.",
            exc_info=True,
        )
        _inductor_big_gpu_override_applied = True
        return
    original = getattr(inductor_utils, "is_big_gpu", None)
    if original is None:
        logger.warning(
            "VLLM_INDUCTOR_OVERRIDE_BIG_GPU set but "
            "torch._inductor.utils.is_big_gpu not found; "
            "torch version may have renamed it."
        )
        _inductor_big_gpu_override_applied = True
        return
    if getattr(original, "__vllm_big_gpu_override__", False):
        _inductor_big_gpu_override_applied = True
        return

    def _forced_big_gpu(index_or_device=0) -> bool:  # noqa: ARG001
        return True

    _forced_big_gpu.__vllm_big_gpu_override__ = True  # type: ignore[attr-defined]
    inductor_utils.is_big_gpu = _forced_big_gpu
    # Clear any cached result from a prior call in this process.
    cache_clear = getattr(original, "cache_clear", None)
    if callable(cache_clear):
        try:
            cache_clear()
        except Exception:
            pass
    logger.info(
        "VLLM_INDUCTOR_OVERRIDE_BIG_GPU=1: patched "
        "torch._inductor.utils.is_big_gpu to always return True. "
        "Triton GEMM templates are now eligible for max-autotune-gemm "
        "on sub-68-SM CUDA devices."
    )
    _inductor_big_gpu_override_applied = True


_maybe_override_inductor_is_big_gpu()


# State for the torch.compile-wrapped BF16 linear fast path.
# Keyed by (x.ndim, weight.shape, bias_is_not_none); each distinct shape
# triggers one inductor compile with mode="max-autotune-no-cudagraphs",
# letting inductor pick between triton/aten/cutlass without compiling
# the whole model.
_unquant_bf16_linear_cache: dict[tuple, Callable] = {}
_unquant_bf16_linear_capture_safe_keys: set[tuple] = set()
_unquant_bf16_linear_torch_compile_configured = False
_unquant_bf16_linear_torch_compile_disabled = False


def _configure_unquant_bf16_linear_torch_compile_once() -> None:
    global _unquant_bf16_linear_torch_compile_configured
    if _unquant_bf16_linear_torch_compile_configured:
        return

    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True

    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024

    _unquant_bf16_linear_torch_compile_configured = True


def _unquant_bf16_linear_eager(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def _unquant_bf16_linear_cache_key(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple:
    return (x.ndim, tuple(weight.shape), bias is not None)


def _get_or_create_unquant_bf16_linear_kernel(
    key: tuple, allow_new_compile: bool
) -> Callable | None:
    compiled = _unquant_bf16_linear_cache.get(key)
    if compiled is not None:
        return compiled

    if not allow_new_compile:
        return None

    _configure_unquant_bf16_linear_torch_compile_once()
    compiled = torch.compile(
        _unquant_bf16_linear_eager,
        backend="inductor",
        mode="max-autotune-no-cudagraphs",
        dynamic=True,
    )
    _unquant_bf16_linear_cache[key] = compiled
    return compiled


def _should_use_unquant_bf16_linear_torch_compile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    if _unquant_bf16_linear_torch_compile_disabled:
        return False
    # Avoid nested compile: if dynamo is already tracing (e.g. vLLM's
    # piecewise model-level torch.compile), let it see the plain F.linear.
    if torch._dynamo.is_compiling():
        return False
    if not x.is_cuda or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if bias is not None and bias.dtype != torch.bfloat16:
        return False
    if not x.is_contiguous():
        return False
    return True


def _apply_unquant_bf16_linear_torch_compile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    global _unquant_bf16_linear_torch_compile_disabled

    key = _unquant_bf16_linear_cache_key(x, weight, bias)
    is_capturing = torch.cuda.is_current_stream_capturing()

    # Never compile during CUDA graph capture; only reuse kernels whose
    # shape key has been observed and compiled in an earlier warm-up.
    if is_capturing and key not in _unquant_bf16_linear_capture_safe_keys:
        return None

    compiled = _get_or_create_unquant_bf16_linear_kernel(
        key, allow_new_compile=not is_capturing
    )
    if compiled is None:
        return None

    try:
        output = compiled(x, weight, bias)
    except Exception:
        if is_capturing:
            return None
        logger.warning(
            "Disabling compiled BF16 linear fast path after "
            "torch.compile failure.",
            exc_info=True,
        )
        _unquant_bf16_linear_torch_compile_disabled = True
        return None

    if not is_capturing:
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


def _tinygemm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    num_tokens = x.numel() // x.shape[-1]
    if (
        num_tokens <= 8
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.shape[0] % 16 == 0
        and x.is_contiguous()
        and weight.is_contiguous()
        and (bias is None or bias.dtype == torch.bfloat16)
    ):
        if bias is None:
            bias = torch.zeros(
                weight.shape[0],
                dtype=torch.bfloat16,
                device=x.device,
            )
        out_shape = (*x.shape[:-1], weight.shape[0])
        result = torch.ops.vllm.tinygemm_bf16(
            x.view(num_tokens, -1),
            weight,
            bias,
        )
        return result.view(out_shape)
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

    if not use_skinny:
        return torch.nn.functional.linear(x, weight, bias)

    x_view = x.reshape(-1, x.size(-1))
    if m > 8 and 0 < n <= 4:
        cu_count = num_compute_units()
        out = ops.wvSplitK(weight, x_view, cu_count, bias)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:
        out = ops.LLMM1(weight, x_view, 4)
        return out.reshape(*x.shape[:-1], weight.shape[0])
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
    elif _TINYGEMM_AVAILABLE:
        return _tinygemm_unquantized_gemm
    else:
        return default_unquantized_gemm
