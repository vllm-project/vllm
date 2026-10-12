# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from functools import cache

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_i64

logger = init_logger(__name__)


@cache
def _zero_weight(device: torch.device) -> torch.Tensor:
    return torch.zeros(128, device=device, dtype=torch.bfloat16)


def _fused_muse_qk_rope_cache_impl(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: float,
    eps: float,
    layer_name: LayerNameType,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache import (
        fused_qkv_split_qk_norm_rope_cache,
    )

    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
    if slot_mapping is None:
        q = qkv.new_zeros((qkv.shape[0], num_heads, head_dim))
        k = qkv.new_zeros((qkv.shape[0], num_kv_heads, head_dim))
        v = qkv.new_zeros((qkv.shape[0], num_kv_heads, head_dim))
    else:
        key_cache, value_cache = attn_layer.impl._split_kv_cache(kv_cache)
        cos, sin = cos_sin_cache.chunk(2, dim=-1)
        weight = _zero_weight(qkv.device)
        q, k, v = fused_qkv_split_qk_norm_rope_cache(
            qkv,
            weight,
            weight,
            cos,
            sin,
            positions,
            key_cache,
            value_cache,
            slot_mapping,
            num_heads,
            num_kv_heads,
            head_dim,
            is_neox=True,
            eps=eps,
            q_scale=q_scale,
            kv_cache_layout="NHD",
        )
    return q, k, v, qkv.new_empty(0)


def _fused_muse_qk_rope_cache_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: float,
    eps: float,
    layer_name: LayerNameType,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        qkv.new_empty((qkv.shape[0], num_heads, head_dim)),
        qkv.new_empty((qkv.shape[0], num_kv_heads, head_dim)),
        qkv.new_empty((qkv.shape[0], num_kv_heads, head_dim)),
        qkv.new_empty(0),
    )


direct_register_custom_op(
    op_name="fused_muse_qk_rope_cache",
    op_func=_fused_muse_qk_rope_cache_impl,
    mutates_args=[],
    fake_impl=_fused_muse_qk_rope_cache_fake,
)


class MuseGlimmerQkRopeCachePattern:
    def __init__(self, layer: Attention, eps: float, q_scale: float) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_dim = layer.head_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.q_scale = q_scale
        self.rope = MatcherRotaryEmbedding(
            is_neox=True,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        return (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)

    def pattern(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self._norm(q.reshape(-1, self.head_dim))
        q = q.reshape(-1, self.q_size) * self.q_scale
        k = self._norm(k.reshape(-1, self.head_dim))
        k = k.reshape(-1, self.kv_size)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        q, k = self.rope(positions, q, k, cos_sin_cache)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        encoded = _encode_layer_name(self.layer_name)
        dummy = torch.ops.vllm.unified_kv_cache_update(
            k, v, encoded if layer_name is None else layer_name
        )
        return dummy, q, k, v

    def replacement(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = _encode_layer_name(self.layer_name)
        q, k, v, dummy = torch.ops.vllm.fused_muse_qk_rope_cache(
            qkv,
            positions,
            cos_sin_cache,
            self.q_scale,
            self.eps,
            encoded if layer_name is None else layer_name,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        return dummy, q, k, v

    def register(self, patterns: PatternMatcherPass) -> None:
        inputs = [
            empty_bf16(5, self.q_size + 2 * self.kv_size),
            empty_i64(5),
            empty_bf16(4096, self.head_dim),
        ]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self.layer_name))

            def pattern(qkv, positions, cos_sin_cache, layer_name):
                return self.pattern(qkv, positions, cos_sin_cache, layer_name)

            def replacement(qkv, positions, cos_sin_cache, layer_name):
                return self.replacement(qkv, positions, cos_sin_cache, layer_name)

        else:

            def pattern(qkv, positions, cos_sin_cache):
                return self.pattern(qkv, positions, cos_sin_cache)

            def replacement(qkv, positions, cos_sin_cache):
                return self.replacement(qkv, positions, cos_sin_cache)

        def trace_fn(fn, args, **kwargs):
            graph = pm.fwd_only(fn, args, **kwargs)
            view_to_reshape(graph)
            return graph

        search_graph = trace_fn(pattern, inputs)
        search_pattern = pm.fx_to_pattern(
            search_graph,
            ignore_types=(int, torch.SymInt),
            argnames=list(inspect.signature(pattern).parameters),
        )
        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            trace_fn,
            patterns,
            search_fn_pattern=search_pattern,
        )


class MuseGlimmerQkRopeCacheFusionPass(VllmPatternMatcherPass):
    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns = PatternMatcherPass(pass_name="muse_qk_rope_cache_fusion")

        from vllm.model_executor.models.muse_glimmer import (
            _muse_glimmer_query_prescale,
            _muse_glimmer_use_qk_norm,
            _text_config,
        )
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionBackend,
        )

        text_config = _text_config(config.model_config.hf_config)
        if not _muse_glimmer_use_qk_norm(text_config):
            return

        for _, layer in get_layers_from_vllm_config(config, Attention).items():
            layer_idx = extract_layer_index(layer.layer_name)
            if (
                text_config.no_rope_layers[layer_idx] != 1
                or layer.get_attn_backend() is not AiterFlashAttentionBackend
                or layer.kv_cache_dtype != "auto"
                or layer.head_size != 128
                or not layer.impl.fused_qk_norm_rope_kvcache_supported()
            ):
                continue
            MuseGlimmerQkRopeCachePattern(
                layer,
                text_config.rms_norm_eps,
                _muse_glimmer_query_prescale(text_config),
            ).register(self.patterns)
            if _USE_LAYERNAME:
                break

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.info("Muse QK-norm+RoPE+KV-cache fusion: %d matches", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, MuseGlimmerQkRopeCachePattern)
