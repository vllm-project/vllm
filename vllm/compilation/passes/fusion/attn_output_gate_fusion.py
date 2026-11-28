# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.utils.torch_utils import _USE_LAYERNAME, _encode_layer_name

from ..vllm_inductor_pass import VllmFusionPatternMatcherPass, VllmPatternReplacement
from .attn_quant_fusion import ATTN_OP, RESHAPE_OP

logger = init_logger(__name__)


class AttnOutputGatePattern(VllmPatternReplacement[..., torch.Tensor]):
    """Moves `attn * sigmoid(gate)` into the Attention op's `output_gate` argument.
    With `gate_by_head`, the gate is a [T, H, D] view that the model flattens with a
    copy; the op takes the view, so the copy is removed too."""

    def __init__(self, layer: Attention, dtype: torch.dtype, gate_by_head: bool):
        self._layer_name = layer.layer_name
        self._num_heads = layer.num_heads
        self._head_size = layer.head_size_v
        self._dtype = dtype
        self._gate_by_head = gate_by_head

    def _flat_gate(self, gate: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if self._gate_by_head:
            return gate.reshape(num_tokens, self._num_heads * self._head_size)
        return gate

    def _op_gate(self, gate: torch.Tensor) -> torch.Tensor:
        if self._gate_by_head:
            return gate
        return RESHAPE_OP(gate, [-1, self._num_heads, self._head_size])

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        _ln = _encode_layer_name(self._layer_name)

        def _gated(q, k, v, output_attn, gate, kv_cache_dummy_dep, layer_name):
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
                output_gate=None,
            )
            attn_out_view = RESHAPE_OP(
                at1[1], [q.shape[0], self._num_heads * self._head_size]
            )
            return attn_out_view * torch.sigmoid(self._flat_gate(gate, q.shape[0]))

        if _USE_LAYERNAME:

            def _pattern_with_ln(  # type: ignore[misc]
                q, k, v, output_attn, gate, kv_cache_dummy_dep, layer_name
            ):
                return _gated(
                    q, k, v, output_attn, gate, kv_cache_dummy_dep, layer_name
                )

            return _pattern_with_ln

        def _pattern(q, k, v, output_attn, gate, kv_cache_dummy_dep):
            return _gated(q, k, v, output_attn, gate, kv_cache_dummy_dep, _ln)

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        _ln = _encode_layer_name(self._layer_name)

        def _fused(q, k, v, output_attn, gate, kv_cache_dummy_dep, layer_name):
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
                output_gate=self._op_gate(gate),
            )
            return RESHAPE_OP(at1[1], [-1, self._num_heads * self._head_size])

        if _USE_LAYERNAME:

            def _replacement_with_ln(  # type: ignore[misc]
                q, k, v, output_attn, gate, kv_cache_dummy_dep, layer_name
            ):
                return _fused(
                    q, k, v, output_attn, gate, kv_cache_dummy_dep, layer_name
                )

            return _replacement_with_ln

        def _replacement(q, k, v, output_attn, gate, kv_cache_dummy_dep):
            return _fused(q, k, v, output_attn, gate, kv_cache_dummy_dep, _ln)

        return _replacement

    def get_inputs(self):
        dtype = self._dtype
        num_heads = self._num_heads
        head_size = self._head_size
        if self._gate_by_head:
            gate = self.empty(5, num_heads, 2 * head_size, dtype=dtype)[..., head_size:]
        else:
            gate = self.empty(5, num_heads * head_size, dtype=dtype)
        inputs: list = [
            self.empty(5, num_heads, head_size, dtype=dtype),
            self.empty(5, num_heads, head_size, dtype=dtype),
            self.empty(5, num_heads, head_size, dtype=dtype),
            self.empty(5, num_heads, head_size, dtype=dtype),
            gate,
            self.empty(0, dtype=dtype),
        ]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self._layer_name))
        return inputs


class AttnOutputGateFusionPass(VllmFusionPatternMatcherPass):
    """Fuses the sigmoid output gate of gated attention into attention layers whose
    implementation supports it, for compile ranges up to
    `attn_output_gate_fusion_max_token_num` tokens."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "attn_output_gate_fusion")
        self.max_token_num = (
            config.compilation_config.pass_config.attn_output_gate_fusion_max_token_num
        )

        dtype = config.model_config.dtype
        layers = list(get_layers_from_vllm_config(config, Attention).values())

        for layer in layers:
            if layer.impl.fused_output_gate_supported():
                for gate_by_head in (True, False):
                    self.register(AttnOutputGatePattern(layer, dtype, gate_by_head))
                if _USE_LAYERNAME:
                    break

        self.dump_patterns(config, self.pm_pass)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num
