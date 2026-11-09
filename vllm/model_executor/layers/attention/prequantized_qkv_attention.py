# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer for models that hand the AITER FA backend FP8 Q/K/V.

The generic `Attention` layer and `AttentionImpl` interface only know about
floating-point Q/K/V. A model whose fused producer already emits FP8 Q/K/V with
per-(sequence, KV head) descales uses `PrequantizedQKVAttention` instead. It
overrides `Attention._dispatch_attention` to call one model-owned custom op,
which calls `forward_with_prequantized_qkv` on the backend implementation.
That method exists only on the ROCm AITER FA impl. `AttentionImpl`, the
`unified_attention_with_output` schema and `Attention.forward`'s arguments
do not change; the PrequantizedQKV bundle travels through `forward`'s
`**dispatch_kwargs`.
"""

from typing import Any, NamedTuple

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)


class PrequantizedQKV(NamedTuple):
    """Prequantized Q/K/V tensors and their per-sequence/per-head descales.

    The tensors have the same logical token/head layout as the corresponding
    floating-point Q/K/V inputs. Descales are `[max_num_seqs, num_kv_heads]`.

    A producer may leave entries for decode-only tokens undefined when the
    backend explicitly uses the floating-point Q/K/V inputs for decode.
    The backend consumes this bundle only for phases the producer quantized.
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    query_descale: torch.Tensor
    key_descale: torch.Tensor
    value_descale: torch.Tensor


class PrequantizedQKVAttention(Attention):
    """`Attention` that can also consume `PrequantizedQKV`.

    Behaves exactly like `Attention` unless the backend implementation
    advertises `supports_prequantized_qkv_input` (only the ROCm AITER FA impl
    does, on gfx950 with an FP8 KV cache) and the caller passes
    `prequantized_qkv`.
    """

    # Set by the model when it passes ``prequantized_qkv`` on every forward.
    # ``supports_prequantized_qkv_input`` only says the backend can consume it;
    # the backend also reads this flag to decide how to size FP8 workspaces.
    receives_prequantized_qkv: bool = False

    @property
    def supports_prequantized_qkv_input(self) -> bool:
        return bool(getattr(self.impl, "supports_prequantized_qkv_input", False))

    def _dispatch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        prequantized_qkv: PrequantizedQKV | None = None,
    ) -> None:
        if prequantized_qkv is None:
            super()._dispatch_attention(query, key, value, output)
            return
        if not self.supports_prequantized_qkv_input:
            raise ValueError(
                f"{self.layer_name} received prequantized QKV, but its attention "
                "backend does not support it"
            )
        assert key is not None and value is not None
        prequantized_qkv = PrequantizedQKV(
            query=prequantized_qkv.query.view(-1, self.num_heads, self.head_size),
            key=prequantized_qkv.key.view(-1, self.num_kv_heads, self.head_size),
            value=prequantized_qkv.value.view(-1, self.num_kv_heads, self.head_size_v),
            query_descale=prequantized_qkv.query_descale,
            key_descale=prequantized_qkv.key_descale,
            value_descale=prequantized_qkv.value_descale,
        )
        kv_cache_dummy_dep = self._update_kv_cache(key, value)
        if self.use_direct_call:
            unified_prequantized_attention_with_output(
                query,
                key,
                value,
                output,
                self.layer_name,
                *prequantized_qkv,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
        else:
            torch.ops.vllm.unified_prequantized_attention_with_output(
                query,
                key,
                value,
                output,
                _encode_layer_name(self.layer_name),
                *prequantized_qkv,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )


@eager_break_during_capture
@maybe_transfer_kv_layer
def unified_prequantized_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    prequantized_query: torch.Tensor,
    prequantized_key: torch.Tensor,
    prequantized_value: torch.Tensor,
    prequantized_query_descale: torch.Tensor,
    prequantized_key_descale: torch.Tensor,
    prequantized_value_descale: torch.Tensor,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    assert isinstance(self, PrequantizedQKVAttention)
    impl: Any = self.impl  # only the AITER FA impl has this method
    impl.forward_with_prequantized_qkv(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        prequantized_qkv=PrequantizedQKV(
            prequantized_query,
            prequantized_key,
            prequantized_value,
            prequantized_query_descale,
            prequantized_key_descale,
            prequantized_value_descale,
        ),
    )


def unified_prequantized_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    prequantized_query: torch.Tensor,
    prequantized_key: torch.Tensor,
    prequantized_value: torch.Tensor,
    prequantized_query_descale: torch.Tensor,
    prequantized_key_descale: torch.Tensor,
    prequantized_value_descale: torch.Tensor,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_prequantized_attention_with_output",
    op_func=unified_prequantized_attention_with_output,
    fake_impl=unified_prequantized_attention_with_output_fake,
    mutates_args=["output"],
)
