# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from vllm import _custom_ops as ops
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.distributed.parallel_state import get_dcp_group, is_global_first_rank
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    quantize_fp8_per_tensor,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.platforms import current_platform


class MLABmm(ABC):
    """Quantized MQA BMM operands derived from a kv_b_proj source weight."""

    @abstractmethod
    def qk(self, x: torch.Tensor, dcp_q_replicated: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def uv(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        pass


class UnquantizedMlaBmm(MLABmm):
    """BF16/FP16 MQA BMM operands."""

    def __init__(
        self,
        kv_b_proj: torch.nn.Module,
        num_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        act_dtype: torch.dtype,
        dcp_q_replicate: bool,
    ) -> None:
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            kv_b_proj, out_dtype=act_dtype
        ).T
        assert kv_b_proj_weight.shape == (
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{kv_lora_rank=}, "
            f"{num_heads=}, "
            f"{qk_nope_head_dim=}, "
            f"{v_head_dim=}"
        )
        w_uk, w_uv = kv_b_proj_weight.view(
            kv_lora_rank,
            num_heads,
            qk_nope_head_dim + v_head_dim,
        ).split([qk_nope_head_dim, v_head_dim], dim=-1)
        w_uk_t = w_uk.permute(1, 2, 0).contiguous()
        w_uv = w_uv.transpose(0, 1).contiguous()
        self.w_uk_t = w_uk_t
        self.w_uv = w_uv
        self.w_uk_t_dcp_qrep = (
            get_dcp_group().all_gather(w_uk_t, dim=0) if dcp_q_replicate else None
        )

    def qk(self, x: torch.Tensor, dcp_q_replicated: bool = False) -> torch.Tensor:
        w_uk_t = self.w_uk_t_dcp_qrep if dcp_q_replicated else self.w_uk_t
        assert w_uk_t is not None
        return torch.bmm(x, w_uk_t).transpose(0, 1)

    def uv(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.bmm(x, self.w_uv, out=out.transpose(0, 1))
        return out


class AmxMlaBmm(MLABmm):
    """AMX-packed MQA BMM operands."""

    def __init__(self, impl: object, kv_lora_rank: int) -> None:
        self.impl = impl
        self.kv_lora_rank = kv_lora_rank

    def qk(self, x: torch.Tensor, dcp_q_replicated: bool = False) -> torch.Tensor:
        num_heads, batch_size, _ = x.shape
        out = x.new_empty((num_heads, batch_size, self.kv_lora_rank))
        ops.bmm_cpu(
            out,
            x,
            self.impl._w_uk_packed,  # type: ignore[attr-defined]
            True,
            None,
        )
        return out.transpose(0, 1)

    def uv(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        ops.bmm_cpu(
            out.transpose(0, 1),
            x,
            self.impl._w_uv_packed,  # type: ignore[attr-defined]
            True,
            None,
        )
        return out


class Fp8MLABmm(MLABmm):
    def __init__(self, w_uk: torch.Tensor, w_uv: torch.Tensor) -> None:
        if not current_platform.is_rocm():
            raise ValueError("Fp8MLABmm requires ROCm.")

        self.w_k, self.w_k_scale = quantize_fp8_per_tensor(w_uk.transpose(0, 1))
        self.w_v, self.w_v_scale = quantize_fp8_per_tensor(w_uv.permute(1, 2, 0))
        self._warm_up()

    def _warm_up(self) -> None:
        # The kernel operates on non-padded inputs. Hence, pre-compiling
        # triton kernel to avoid runtime compilation for unseen batch sizes
        # Pre-compile for batch sizes 1 to 1024 to cover most use-cases.
        # On DS-R1, this step adds roughly 50s to the model loading time.
        max_batch_size = 1024  # [ToDo] Find the optimal upper limit
        pre_compilation_list = list(range(1, max_batch_size + 1))
        if is_global_first_rank():
            pre_compilation_list = tqdm(
                pre_compilation_list,
                desc="[Aiter Triton] Pre-compiling fp8 BMM kernel",
                total=max_batch_size,
            )

        for m in pre_compilation_list:
            x = torch.empty(
                (self.w_k.shape[0], m, self.w_k.shape[2]),
                dtype=torch.bfloat16,
                device=self.w_k.device,
            )
            rocm_aiter_ops.triton_fp8_bmm(
                x,
                self.w_k,
                self.w_k_scale,
                group_size=128,
                transpose_bm=True,
            )

            x = torch.empty(
                (self.w_v.shape[0], m, self.w_v.shape[2]),
                dtype=torch.bfloat16,
                device=self.w_v.device,
            )
            rocm_aiter_ops.triton_fp8_bmm(
                x,
                self.w_v,
                self.w_v_scale,
                group_size=128,
                transpose_bm=True,
            )

    def qk(self, x: torch.Tensor, dcp_q_replicated: bool = False) -> torch.Tensor:
        return rocm_aiter_ops.triton_fp8_bmm(
            x,
            self.w_k,
            self.w_k_scale,
            group_size=128,
            transpose_bm=True,
        )

    def uv(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return rocm_aiter_ops.triton_fp8_bmm(
            x,
            self.w_v,
            self.w_v_scale,
            group_size=128,
            transpose_bm=True,
            YQ=out,
        )


class Mxfp4MLABmm(MLABmm):
    def __init__(self, w_uk: torch.Tensor, w_uv: torch.Tensor) -> None:
        if not current_platform.is_rocm():
            raise ValueError("Mxfp4MLABmm requires ROCm.")

        self.w_k, self.w_k_scale = mxfp4_quantize(w_uk)
        self.w_k = self.w_k.transpose(0, 1)
        self.w_k_scale = self.w_k_scale.transpose(0, 1)
        self.w_v, self.w_v_scale = mxfp4_quantize(w_uv.permute(1, 2, 0))

    def qk(self, x: torch.Tensor, dcp_q_replicated: bool = False) -> torch.Tensor:
        out = x.new_empty((x.shape[1], x.shape[0], self.w_k.shape[1]))
        return rocm_aiter_ops.batched_gemm_a16wfp4(
            x,
            self.w_k,
            self.w_k_scale,
            out,
            transpose_bm=True,
            prequant=True,
        )

    def uv(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return rocm_aiter_ops.batched_gemm_a16wfp4(
            x,
            self.w_v,
            self.w_v_scale,
            out,
            transpose_bm=True,
            prequant=True,
        )


def create_online_mla_bmm(
    quant_method: object,
    weight: torch.Tensor,
    num_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> MLABmm | None:
    """Quantize MLA BMM weights while kv_b_proj is still in high precision."""
    if not is_aiter_found_and_supported():
        return None

    kv_b_proj_weight = weight.T.view(
        kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim
    )
    w_uk, w_uv = kv_b_proj_weight.split([qk_nope_head_dim, v_head_dim], dim=-1)

    if isinstance(quant_method, Fp8PerTensorOnlineLinearMethod):
        return Fp8MLABmm(w_uk, w_uv)

    if isinstance(quant_method, Mxfp4OnlineLinearMethod):
        return Mxfp4MLABmm(w_uk, w_uv)

    return None
