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
from vllm.utils.platform_utils import get_cu_count
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


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    # Shuffle weight along the last dimension so that
    # we folded the weights to adjance location
    # Example:
    # input:
    #       [[1, 2, 3, 4, 5, 6],
    #        [7, 8, 9, 10, 11, 12]]
    # output:
    #       [[1, 4, 2, 5, 3, 6],
    #        [7, 10, 8, 11, 9, 12]]
    # This will be used together with triton swiglu kernel
    shape = w.shape
    N = shape[-1]
    first = w[..., : N // 2]
    second = w[..., N // 2 :]

    stacked = torch.stack((first, second), dim=-1)
    w_shuffled = stacked.reshape(shape)
    return w_shuffled


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
    from vllm.platforms.rocm import on_gfx9, on_gfx950

    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]

    cu_count = get_cu_count()
    if use_aiter_triton_gemm(n, m, k, x.dtype):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        return gemm_a16w16(x, weight, bias)

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
    fits_wvsplitkrc = CuNeeded <= cu_count

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
            and x.is_contiguous()
        )
    )
    if use_skinny_reduce_counting:
        x_view = x.reshape(-1, x.size(-1))
        out = ops.wvSplitKrc(weight, x_view, cu_count, bias)
        return out.reshape(*x.shape[:-1], weight.shape[0])

    use_skinny = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and on_gfx9()
        and x.dtype in [torch.float16, torch.bfloat16]
        and k % 8 == 0
        and x.is_contiguous()
    )

    if use_skinny is not True:
        return torch.nn.functional.linear(x, weight, bias)

    x_view = x.reshape(-1, x.size(-1))
    if m > 8 and 0 < n <= 4:
        cu_count = get_cu_count()
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
        torch._C._cpu._is_amx_tile_supported()
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
    else:
        return default_unquantized_gemm
