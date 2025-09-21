# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""
from typing import Callable, Optional

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

from vllm.layers.quantization.kernels.tma_persistent_gemm import matmul_tma_persistent

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
    first = w[..., :N // 2]
    second = w[..., N // 2:]

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
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                    output_tokens_tensor: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor) -> torch.Tensor:
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
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                   vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties
    apply_repetition_penalties(logits, prompt_mask, output_mask,
                               repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(layer: torch.nn.Module,
                             x: torch.Tensor,
                             weight: torch.Tensor,
                             bias: Optional[torch.Tensor] = None):
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx9
    k = weight.shape[1]
    use_skinny = (envs.VLLM_ROCM_USE_SKINNY_GEMM and on_gfx9() and \
                    x.dtype in [torch.float16, torch.bfloat16] \
                    and k % 8 == 0 and bias is None)

    if use_skinny is not True:
        return torch.nn.functional.linear(x, weight, bias)

    x_view = x.view(-1, x.size(-1))
    n = x_view.shape[0]
    m = weight.shape[0]
    cu_count = current_platform.get_cu_count()

    if m > 8 and 0 < n <= 4:
        out = ops.wvSplitK(weight, x_view, cu_count)
        return out.view(*x.shape[:-1], weight.shape[0])
    elif m % 4 == 0 and n == 1 and k <= 8192:
        out = ops.LLMM1(weight, x_view, 4)
        return out.view(*x.shape[:-1], weight.shape[0])
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_impl_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(layer: torch.nn.Module,
                          x: torch.Tensor,
                          weight: torch.Tensor,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm_impl(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm_impl",
    op_func=rocm_unquantized_gemm_impl,
    mutates_args=[],
    fake_impl=rocm_unquantized_gemm_impl_fake,
    dispatch_key=current_platform.dispatch_key,
)


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    return (torch._C._cpu._is_amx_tile_supported()
            and (dtype in (torch.bfloat16, torch.int8)) and k % 32 == 0
            and n % 16 == 0)


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    N, K = layer.weight.size()
    dtype = layer.weight.dtype
    if envs.VLLM_CPU_SGL_KERNEL and check_cpu_sgl_kernel(N, K, dtype):
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            bias_f32 = layer.bias.to(torch.float32)
        else:
            bias_f32 = None
        layer.cpu_linear = (
            lambda x, weight, bias: torch.ops._C.weight_packed_linear(
                x, packed_weight, bias_f32
                if bias is not None else None, True))
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0),
                                              requires_grad=False)
    elif ops._supports_onednn:
        origin_weight = layer.weight
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0),
                                              requires_grad=False)
        handler = ops.create_onednn_mm(origin_weight.t(), 32)
        layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(
            handler, x, bias)
    else:
        layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
            x, weight, bias)


def cpu_unquantized_gemm(layer: torch.nn.Module,
                         x: torch.Tensor,
                         weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None):
    return layer.cpu_linear(x, weight, bias)


def fp8_tma_linear(x: torch.Tensor,
                   weight: torch.Tensor,
                   bias: Optional[torch.Tensor] = None,
                   layer=None):
    """Optimized FP8 TMA linear function with CUDA graph capture support.

    Args:
        x: Input tensor, will be converted to FP8 if needed
        weight: Weight tensor in FP8E4M3FN format (already transposed)
        bias: Optional bias tensor
        layer: Layer instance for accessing vLLM config and persistent buffers

    Returns:
        Output tensor in BF16 format
    """
    try:
        ## Do not convert the input to FP8 here, convert it inside of the kernel
        # if x.dtype != torch.float8_e4m3fn:
        #     x = x.to(torch.float8_e4m3fn)

        # Weight should already be FP8 from PureFp8LinearMethod
        assert weight.dtype == torch.float8_e4m3fn, f"Expected FP8 weight, got {weight.dtype}"

        # Handle arbitrary input shapes like torch.nn.functional.linear
        input_shape = x.shape
        x_2d = x.view(-1, x.size(-1))  # Flatten to 2D for matmul
        num_tokens = x_2d.shape[0]

        # Initialize persistent buffers if not available
        if layer is not None:
            # Hardcoded batch sizes for CUDA graph capture

            # Simple padding to next power of 2 up to max batch size
            M_cap = num_tokens
            for size in layer.cudagraph_batch_sizes:
                if num_tokens <= size:
                    M_cap = size
                    break

            # CUDA graph capture and replay for TMA persistent kernel
            if M_cap <= layer.cudagraph_batch_sizes[-1]:
                graph_key = M_cap  # Simple batch size key since weight is fixed

                if graph_key not in layer._tma_graphs:
                    # TMA kernel format preparation
                    if weight.shape[1] != x_2d.shape[1]:
                        weight_for_tma = weight.T
                    else:
                        weight_for_tma = weight

                    # Create input/output buffers for this batch size and weight
                    padded_input_shape = (M_cap, x_2d.shape[1])
                    output_shape = (M_cap, weight_for_tma.shape[0])

                    layer._tma_inputs[graph_key] = torch.zeros(
                        padded_input_shape, dtype=x.dtype, device=x.device)
                    layer._tma_outputs[graph_key] = torch.zeros(
                        output_shape, dtype=torch.bfloat16, device=x.device)
                    layer._tma_weights[graph_key] = weight_for_tma

                    # Create memory pool and stream for stable capture
                    # Check if memory_pool is available and functional
                    memory_pool = None
                    try:
                        if hasattr(torch.cuda, 'memory_pool'):
                            memory_pool = torch.cuda.memory_pool()
                    except Exception:
                        # Memory pool API exists but not functional, continue without it
                        memory_pool = None

                    stream = torch.cuda.Stream()

                    # Capture the graph
                    layer._tma_graphs[graph_key] = torch.cuda.CUDAGraph()
                    if memory_pool is not None:
                        with torch.cuda.graph(layer._tma_graphs[graph_key], pool=memory_pool, stream=stream):
                            layer._tma_outputs[graph_key] = matmul_tma_persistent(
                                layer._tma_inputs[graph_key],
                                layer._tma_weights[graph_key],
                                bias=bias
                            )
                    else:
                        # Fallback for older PyTorch versions without memory_pool
                        with torch.cuda.graph(layer._tma_graphs[graph_key], stream=stream):
                            layer._tma_outputs[graph_key] = matmul_tma_persistent(
                                layer._tma_inputs[graph_key],
                                layer._tma_weights[graph_key],
                                bias=bias
                            )

                # Copy input data to buffer
                if M_cap > num_tokens:
                    layer._tma_inputs[graph_key][:num_tokens] = x_2d
                    # layer._tma_inputs[graph_key][num_tokens:] = 0 ## remove this line to reduce the overhead
                else:
                    layer._tma_inputs[graph_key].copy_(x_2d)

                # Replay the graph
                layer._tma_graphs[graph_key].replay()

                # Get output and slice back to actual size if padded
                if M_cap > num_tokens:
                    output_2d = layer._tma_outputs[graph_key][:num_tokens]
                else:
                    output_2d = layer._tma_outputs[graph_key]
            else:
                # Fallback for large batch sizes - use TMA kernel without graph
                if weight.shape[1] != x_2d.shape[1]:
                    weight_for_tma = weight.T
                else:
                    weight_for_tma = weight

                output_2d = matmul_tma_persistent(
                    x_2d, weight_for_tma,
                    bias=bias
                )

        else:
            # Fallback for cases without layer context
            # TMA kernel expects: a=(M,K), b=(N,K) where a.shape[1] == b.shape[1]
            if weight.shape[1] != x_2d.shape[1]:
                weight_for_tma = weight.T
            else:
                weight_for_tma = weight

            # Use customized TMA persistent kernel for all cases
            output_2d = matmul_tma_persistent(
                x_2d, weight_for_tma,
                bias=bias
            )

        # Reshape back to match expected output shape
        # For TMA kernel: weight_for_tma is (N, K), so N is at dimension 0
        output_features = weight_for_tma.shape[0] if 'weight_for_tma' in locals() else weight.shape[0]
        output_shape = input_shape[:-1] + (output_features,)
        output = output_2d.view(output_shape)

        return output
    except ImportError as e:
        # Fallback to regular linear if TMA kernel not available
        print(f"Warning: TMA kernel not available ({e}), falling back to regular linear")
        return torch.nn.functional.linear(x, weight, bias)
    except Exception as e:
        # Fallback for any other errors
        print(f"Warning: TMA kernel failed ({e}), falling back to regular linear") 
        return torch.nn.functional.linear(x, weight, bias)


def unquantized_gemm_with_fp8_support(layer: torch.nn.Module,
                                      x: torch.Tensor,
                                      weight: torch.Tensor,
                                      bias: Optional[torch.Tensor] = None):
    """Unquantized GEMM with FP8 TMA kernel support."""
    # Check if weight is FP8 - if so, use TMA kernel
    # print(weight.dtype)
    # print(x.dtype)
    if weight.dtype == torch.float8_e4m3fn:
        # x = x.to(torch.float8_e4m3fn)
        return fp8_tma_linear(x, weight, bias, layer=layer)
    else:
        # Use regular linear for non-FP8 weights
        # return default_unquantized_gemm
        return torch.nn.functional.linear(x, weight, bias)

def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return unquantized_gemm_with_fp8_support
