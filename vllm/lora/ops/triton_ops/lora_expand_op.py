# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm import envs
from vllm.lora.ops.triton_ops.kernel_utils import do_expand_kernel
from vllm.lora.ops.triton_ops.utils import (
    _get_lora_b_ptr,
    get_lora_op_configs,
    supports_pdl,
)
from vllm.model_executor.warmup.jit_warmup import WarmupIntRange
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    kernel_launcher,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _lora_expand_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,  # 1
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,  # 1
    output_d0_stride,
    output_d1_stride,  # 1
    output_hs_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_mn = tl.program_id(axis=0)
    pid_m = pid_mn % cta_m_num
    pid_n = (pid_mn // cta_m_num) % cta_n_num

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # When the output dimensions of each slice are the same,cur_n=N, otherwise
    # cur_n=tl.load(output_hs_ptr + slice_id), this situation exists in GQA's
    # qkv linear.
    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)
    if pid_n * BLOCK_N >= curr_N:
        # Early exit CTA.
        return

    # num rows this CTA should process.
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    # Identify all rows that this CTA should process.
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (
        token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    )

    # Load all relevant row indices.
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_expand_kernel(
        pid_n,
        lora_id,
        slice_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        curr_N,
        K,
        cta_m_len,
        ram,  # array identifying the rows of Input ptr to operate on
        slice_start_loc,
        # input ptr strides
        input_d0_stride,
        input_d1_stride,
        input_d2_stride,
        # lora ptr strides
        ls_d0_ptr,
        ls_d1_ptr,
        ls_d2_ptr,
        # out ptr strides
        output_d0_stride,
        output_d1_stride,
        # constants
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SAME_STRIDE,
        SLICE_NUM,
        EVEN_K,
        CAST_TYPE,
        ADD_INPUTS,
        USE_GDC,
    )

class LoRAExpandKernel(VllmTritonJitKernel["LoRAExpandKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        input_dtype: torch.dtype
        weight_dtype: torch.dtype
        output_dtype: torch.dtype
        m: int
        n: int
        k: int
        input_d0_stride: int
        input_d1_stride: int
        input_d2_stride: int
        lora_d0_stride: int
        lora_d1_stride: int
        lora_d2_stride: int
        output_d0_stride: int
        output_d1_stride: int
        block_m: int
        block_n: int
        block_k: int
        even_k: bool
        add_inputs: bool
        cast_type: bool
        slice_num: int
        same_stride: bool
        lora_pointer_table: bool
        metadata_table: bool
        use_gdc: bool
        num_warps: int
        num_ctas: int
        num_stages: int

    kernel = staticmethod(_lora_expand_kernel)


    def dispatch(
        self,
        *,
        m: int,
        n: int,
        k: int,
        max_loras: int,
        input_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        output_dtype: torch.dtype,
        lora_d0_stride: int,
        lora_d1_stride: int,
        lora_d2_stride: int,
        output_d0_stride: int,
        output_d1_stride: int,
        add_inputs: bool,
        cast_type: bool,
        slice_num: int,
        same_stride: bool,
        lora_pointer_table: bool,
        metadata_table: bool,
        use_gdc: bool,
    ) -> "LoRAExpandKernel.CompileKey":
        config = get_lora_op_configs(
            op_type="expand",
            max_loras=max_loras,
            batch=m,
            hidden_size=n,
            rank=k,
            num_slices=slice_num,
            add_inputs=add_inputs,
        )
        return self.CompileKey(
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            m=triton_scalar_specialization_rep(m),
            n=triton_scalar_specialization_rep(n),
            k=triton_scalar_specialization_rep(k),
            input_d0_stride=triton_scalar_specialization_rep(m * k),
            input_d1_stride=triton_scalar_specialization_rep(k),
            input_d2_stride=1,
            lora_d0_stride=triton_scalar_specialization_rep(lora_d0_stride),
            lora_d1_stride=triton_scalar_specialization_rep(lora_d1_stride),
            lora_d2_stride=triton_scalar_specialization_rep(lora_d2_stride),
            output_d0_stride=triton_scalar_specialization_rep(output_d0_stride),
            output_d1_stride=triton_scalar_specialization_rep(output_d1_stride),
            block_m=config["block_m"],
            block_n=config["block_n"],
            block_k=config["block_k"],
            even_k=k % config["block_k"] == 0,
            add_inputs=add_inputs,
            cast_type=cast_type,
            slice_num=slice_num,
            same_stride=same_stride,
            lora_pointer_table=lora_pointer_table,
            metadata_table=metadata_table,
            use_gdc=use_gdc,
            num_warps=config["num_warps"],
            num_ctas=config["num_ctas"],
            num_stages=config["num_stages"],
        )

    def get_warmup_keys(
        self,
        *,
        max_tokens: int,
        max_loras: int,
        **compile_key_fields: Any,
    ) -> list["LoRAExpandKernel.CompileKey"]:
        return self._trace_dispatch(self.dispatch)(
            m=WarmupIntRange(1, max_tokens + 1, advance=lambda value: value * 2),
            max_loras=max_loras,
            use_gdc=(False, True),
            **compile_key_fields,
        )

    def warmup_inputs(
        self, compile_key: "LoRAExpandKernel.CompileKey"
    ) -> dict[str, Any]:
        metadata = TritonWarmupTensor(torch.int32)
        lora_metadata: int | TritonWarmupTensor
        lora_metadata = (
            TritonWarmupTensor(torch.int64)
            if compile_key.metadata_table
            else compile_key.lora_d0_stride
        )
        return dict(
            input_ptr=TritonWarmupTensor(
                compile_key.input_dtype,
                shape=(1, 1, 1),
                strides=(
                    compile_key.input_d0_stride,
                    compile_key.input_d1_stride,
                    compile_key.input_d2_stride,
                ),
            ),
            lora_ptr=TritonWarmupTensor(
                torch.uint64
                if compile_key.lora_pointer_table
                else compile_key.weight_dtype
            ),
            out_ptr=TritonWarmupTensor(
                compile_key.output_dtype,
                shape=(1, 1),
                strides=(
                    compile_key.output_d0_stride,
                    compile_key.output_d1_stride,
                ),
            ),
            M=compile_key.m,
            N=compile_key.n,
            K=compile_key.k,
            token_indices_sorted_by_lora_ids=metadata,
            num_tokens_per_lora=metadata,
            lora_token_start_loc=metadata,
            lora_ids=metadata,
            slice_start_loc=lora_metadata,
            input_d0_stride=compile_key.input_d0_stride,
            input_d1_stride=compile_key.input_d1_stride,
            input_d2_stride=compile_key.input_d2_stride,
            ls_d0_ptr=lora_metadata,
            ls_d1_ptr=lora_metadata,
            ls_d2_ptr=lora_metadata,
            output_d0_stride=compile_key.output_d0_stride,
            output_d1_stride=compile_key.output_d1_stride,
            output_hs_ptr=lora_metadata,
            BLOCK_M=compile_key.block_m,
            BLOCK_N=compile_key.block_n,
            BLOCK_K=compile_key.block_k,
            EVEN_K=compile_key.even_k,
            ADD_INPUTS=compile_key.add_inputs,
            CAST_TYPE=compile_key.cast_type,
            SLICE_NUM=compile_key.slice_num,
            SAME_STRIDE=compile_key.same_stride,
            USE_GDC=compile_key.use_gdc,
            launch_pdl=compile_key.use_gdc,
            grid=(1, 1, 1),
            num_warps=compile_key.num_warps,
            num_ctas=compile_key.num_ctas,
            num_stages=compile_key.num_stages,
        )

    @kernel_launcher
    def __call__(
        self,
        *args: Any,
        grid: tuple[int, ...],
        num_warps: int,
        num_ctas: int,
        num_stages: int,
        **kwargs: Any,
    ) -> LaunchSpec:
        return grid, dict(
            num_warps=num_warps,
            num_ctas=num_ctas,
            num_stages=num_stages,
        )


_LORA_EXPAND_KERNEL = LoRAExpandKernel()


@torch.inference_mode()
def _lora_expand(
    inputs: torch.Tensor,  # shape [num_slices, num_tokens, lora_rank]
    lora_b_weights: list[torch.Tensor],  # shape [num_lora, hidden_size, lora_rank]
    output_tensor: torch.Tensor,  # shape [num_tokens, hidden_size * num_slices]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens]
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    no_lora_flag_cpu: torch.Tensor,  # shape [1]
    num_active_loras: torch.Tensor,  # CPU tensor [1], number of active LoRAs
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (list[torch.Tensor]): lora'b weight
        output_tensor (torch.Tensor): output tensor
        token_lora_mapping (torch.Tensor): A tensor mapping each input token
            to the lora-id related to that token. A value of -1 indicates that
            LoRA doesn't apply to that token.
        token_indices_sorted_by_lora_ids (torch.Tensor): Row/Token indices from
            the A matrix grouped by LoRA IDs.
        num_tokens_per_lora (torch.Tensor): num_tokens_per_lora[i] is the number
            of tokens that are to be processed by LoRA ID lora_ids[i]
        lora_token_start_loc (torch.Tensor): A cumulative sum of
            num_tokens_per_lora. lora_token_start_loc[0] is always 0 so that
            lora_token_start_loc[i], along with num_tokens_per_lora[i]
            identifies the region in token_indices_sorted_by_lora_ids that
            LoRA lora_ids[i] should process.
        lora_ids (torch.Tensor): LoRA ids to process.
        no_lora_flag_cpu (torch.Tensor): A CPU tensor of size 1, that indicates
            if there are any requests that require LoRA.
        offset_start (int, optional): Offset start for output_tensor.
            Defaults to 0.
        add_inputs (bool, optional): Whether to add the input tensor to the
            output tensor. Defaults to False.
    """

    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        # None of the inputs require LoRA.
        return

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(0) == len(lora_b_weights)
    assert output_tensor.is_contiguous()

    # metadata sanity check.
    M = inputs.size(1)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    (
        slice_start_tensor,
        lora_ptr_tensor,
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        hidden_sizes_tensor,
        same_stride,
        MAX_N,
    ) = _get_lora_b_ptr(lora_b_weights, offset_start, inputs.device)

    K = lora_b_weights[0].shape[-1]  # K= rank
    ADD_INPUTS = add_inputs
    MAX_LORAS = lora_ids.size(0)
    CAST_TYPE = False
    NUM_SLICES = len(lora_b_weights)

    # Triton kernel configs.
    kernel_config = get_lora_op_configs(
        op_type="expand",
        max_loras=MAX_LORAS,
        batch=M,
        hidden_size=MAX_N,
        rank=K,
        num_slices=NUM_SLICES,
        add_inputs=add_inputs,
    )
    BLOCK_M = kernel_config["block_m"]
    BLOCK_N = kernel_config["block_n"]
    BLOCK_K = kernel_config["block_k"]
    NUM_WARPS = kernel_config["num_warps"]
    NUM_CTAS = kernel_config["num_ctas"]
    NUM_STAGES = kernel_config["num_stages"]

    EVEN_K = K % BLOCK_K == 0  # type: ignore

    if inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [
        torch.float16,
        torch.bfloat16,
    ]:
        CAST_TYPE = True

    # TODO (varun): This grid formulation maximizes parallelization at the
    # cost of wasteful thread block launch when only a few input tokens require
    # LoRA. This might not be the best in all cases.
    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(MAX_N, BLOCK_N),
        NUM_SLICES,
        num_active_loras.item(),
    )

    # PDL only works when dual-stream is being used.
    use_gdc = supports_pdl(inputs.device) and envs.VLLM_LORA_ENABLE_DUAL_STREAM
    _LORA_EXPAND_KERNEL(
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        MAX_N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        hidden_sizes_tensor,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
        NUM_SLICES,
        same_stride,
        use_gdc,
        grid=grid,
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        num_stages=NUM_STAGES,
        launch_pdl=use_gdc,
    )

    return


try:
    direct_register_custom_op(
        op_name="lora_expand",
        op_func=_lora_expand,
        mutates_args=["output_tensor"],
    )
    lora_expand = torch.ops.vllm.lora_expand

except AttributeError:
    lora_expand = _lora_expand
