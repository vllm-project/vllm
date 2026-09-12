# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private rowwise-FP8 proposal head for Qwen4Exp MTP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )


class Fp8ProposalHeadMethod(QuantizeMethodBase):
    """Dynamic-per-token W8A8 projection over a private FP8 weight copy."""

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise RuntimeError("FP8 proposal-head weights are derived after loading")

    def apply(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "The Qwen4Exp FP8 proposal head requires BF16 hidden states, "
                f"got {hidden_states.dtype}."
            )
        if hidden_states.shape[-1] != layer.weight_fp8.shape[1]:
            raise ValueError(
                "The Qwen4Exp FP8 proposal-head hidden width does not match "
                f"its weight: {hidden_states.shape[-1]} != "
                f"{layer.weight_fp8.shape[1]}."
            )

        output_shape = (*hidden_states.shape[:-1], layer.weight_fp8.shape[0])
        hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        hidden_fp8, hidden_scale = ops.scaled_fp8_quant(
            hidden_2d, use_per_token_if_dynamic=True
        )
        logits = ops.cutlass_scaled_mm(
            hidden_fp8,
            layer.weight_fp8.t(),
            scale_a=hidden_scale,
            scale_b=layer.weight_scale.t(),
            out_dtype=hidden_states.dtype,
            bias=bias,
        )
        return logits.view(output_shape)


class Fp8ProposalHead(nn.Module):
    """LM-head adapter owning an immutable rowwise-E4M3 weight copy."""

    def __init__(self, source: VocabParallelEmbedding) -> None:
        super().__init__()
        self._validate_source(source)
        weight = source.weight.detach()

        # This fused quantizer avoids full-size FP32/absolute-value temporaries.
        weight_fp8, weight_scale = ops.scaled_fp8_quant(
            weight, use_per_token_if_dynamic=True
        )
        expected_scale_shape = (weight.shape[0], 1)
        if (
            weight_fp8.shape != weight.shape
            or weight_scale.shape != expected_scale_shape
        ):
            raise RuntimeError(
                "Unexpected Qwen4Exp proposal-head FP8 shapes: "
                f"weight={tuple(weight_fp8.shape)}, "
                f"scale={tuple(weight_scale.shape)}."
            )
        if weight_fp8.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"Unexpected proposal-head dtype: {weight_fp8.dtype}.")
        if weight_scale.dtype != torch.float32:
            raise RuntimeError(
                f"Unexpected proposal-head scale dtype: {weight_scale.dtype}."
            )

        self.register_buffer("weight_fp8", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        self.quant_method = Fp8ProposalHeadMethod()
        self.tp_size = source.tp_size
        self.shard_indices = source.shard_indices
        self.num_embeddings_per_partition = source.num_embeddings_per_partition
        self.embedding_dim = source.embedding_dim

    @staticmethod
    def _validate_source(source: VocabParallelEmbedding) -> None:
        weight = source.weight
        if not isinstance(
            source.quant_method,
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        ):
            raise RuntimeError(
                "The Qwen4Exp FP8 proposal head requires an ordinary "
                "unquantized source LM head."
            )
        if getattr(source, "parallel_group", None) is not None:
            raise RuntimeError(
                "The Qwen4Exp FP8 proposal head does not support a custom "
                "LM-head parallel group."
            )
        if getattr(source, "bias", None) is not None:
            raise RuntimeError("The Qwen4Exp FP8 proposal head does not support bias.")
        if not weight.is_cuda:
            raise RuntimeError("The Qwen4Exp FP8 proposal head requires CUDA.")
        if weight.ndim != 2:
            raise RuntimeError(
                "The Qwen4Exp FP8 proposal head requires a 2-D weight, "
                f"got {tuple(weight.shape)}."
            )
        if weight.dtype != torch.bfloat16:
            raise RuntimeError(
                "The Qwen4Exp FP8 proposal head requires a BF16 weight, "
                f"got {weight.dtype}."
            )
        if not weight.is_contiguous():
            raise RuntimeError(
                "The Qwen4Exp FP8 proposal head requires a contiguous weight; "
                "it will not make a full-size contiguous copy."
            )
        if weight.shape[0] % 16 or weight.shape[1] % 16:
            raise RuntimeError(
                "The Qwen4Exp FP8 proposal-head dimensions must be divisible "
                f"by 16, got {tuple(weight.shape)}."
            )
        major, minor = torch.cuda.get_device_capability(weight.device)
        capability = major * 10 + minor
        if not ops.cutlass_scaled_mm_supports_fp8(capability):
            raise RuntimeError(
                "The Qwen4Exp FP8 proposal head requires CUTLASS FP8 scaled "
                f"MM support; SM{capability} is unsupported."
            )

    @property
    def storage_bytes(self) -> int:
        return self.weight_fp8.nbytes + self.weight_scale.nbytes


__all__ = ["Fp8ProposalHead", "Fp8ProposalHeadMethod"]
