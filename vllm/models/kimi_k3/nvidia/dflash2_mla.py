# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1 DFlash2 draft with MLA layers (`dflash_config.attention_mode="mla"`).

The DFlash2 block drafter (grouped convolutions, candidate selector) on K3's
MLA attention. Sliding-window layers are block-causal: every query row attends
the last `sliding_window` positions up to its own, read with the trtllm-gen
sparse MLA decode the target's DSA layers use.
"""

import math
from collections.abc import Iterable
from dataclasses import replace

import torch
from torch import nn

from vllm.compilation.decorators import ignore_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen3 import Qwen3MLP
from vllm.model_executor.models.qwen3_dflash import _resolve_layer_attention
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3DecoderLayer,
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DraftContextKV,
    K3DSparkAttention,
    _duplicate_context_kv_weights,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    _TRTLLM_GEN_MAX_REQUESTS,
    _get_workspace_buffer,
    _kv_counters,
    _pad_query_heads,
    trtllm_gen_query_heads,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    align_blocks_to_rows,
    flat_kv_row_view,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayerConfig,
    is_quantized_kv_cache,
)


class _WindowRows:
    """Cache rows of every query row's window, shared by one step's layers."""

    def __init__(self) -> None:
        self._rows: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self) -> None:
        self._rows = None

    def get(
        self,
        positions: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        block_size: int,
        block_stride_rows: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows `[num_tokens, window]` (int32, newest first, -1 padded) and the
        number of valid rows per token."""
        if self._rows is not None:
            return self._rows
        decode = attn_metadata.decode
        assert decode is not None
        block_table = decode.block_table
        num_tokens = positions.shape[0]
        query_end = attn_metadata.query_start_loc[1 : attn_metadata.num_decodes + 1]
        token = torch.arange(num_tokens, device=positions.device, dtype=query_end.dtype)
        req = torch.searchsorted(query_end, token, right=True)
        req = req.clamp_(max=block_table.shape[0] - 1)

        offsets = torch.arange(window, device=positions.device)
        key_pos = positions.unsqueeze(1) - offsets
        valid = key_pos >= 0
        key_pos = key_pos.clamp_(min=0)
        block = (key_pos // block_size).clamp_(max=block_table.shape[1] - 1)
        rows = block_table[req.unsqueeze(1), block] * block_stride_rows
        rows += key_pos % block_size
        rows = rows.masked_fill_(~valid, -1).to(torch.int32)
        counts = (positions + 1).clamp(max=window).to(torch.int32)
        self._rows = rows, counts
        return self._rows


class K3DFlash2Attention(K3DSparkAttention):
    def __init__(self, *, sliding_window: int | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sliding_window = sliding_window
        self.causal = True
        # Set by the model: one index build serves every layer of a step.
        self.window_rows = _WindowRows()
        if sliding_window is None:
            return
        if self.dcp_world_size > 1:
            raise NotImplementedError(
                "Sliding-window Kimi-K3 DFlash2 layers do not support decode "
                "context parallelism."
            )
        if (
            not is_quantized_kv_cache(self.kv_cache_dtype)
            or self.kv_cache_dtype == "fp8_ds_mla"
        ):
            raise NotImplementedError(
                "Sliding-window Kimi-K3 DFlash2 layers need a per-tensor fp8 "
                f"draft KV cache; got {self.kv_cache_dtype!r}."
            )
        # The sparse decode reads fp8 rows with an fp8 query.
        self.impl.supports_quant_query_input = True

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        layer_config = super().get_kv_cache_config(vllm_config)
        if self.sliding_window is None:
            return layer_config
        # The sparse decode reads the cache as flat rows, in whole pages.
        spec = layer_config.spec
        assert isinstance(spec, AttentionSpec)
        page_rows = math.gcd(spec.block_size, 64)
        return replace(layer_config, spec=align_blocks_to_rows(spec, page_rows))

    def _forward_mqa(
        self,
        mqa_q: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        window = self.sliding_window
        if window is None:
            return super()._forward_mqa(mqa_q, positions, attn_metadata)

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        kv_cache = self._attn_read_kv_cache()
        block_size = kv_cache.shape[1]
        kv_rows, block_stride_rows = flat_kv_row_view(kv_cache, block_size)
        rows, counts = self.window_rows.get(
            positions, attn_metadata, block_size, block_stride_rows, window
        )
        # Rows are flat, so pages need not follow blocks.
        page_rows = math.gcd(block_size, 64)
        pages = kv_rows[: kv_rows.shape[0] // page_rows * page_rows].view(
            -1, page_rows, kv_rows.shape[-1]
        )

        num_tokens, num_heads, _ = mqa_q.shape
        assert num_tokens <= _TRTLLM_GEN_MAX_REQUESTS
        padded_heads = trtllm_gen_query_heads(num_heads)
        if padded_heads != num_heads:
            mqa_q = _pad_query_heads(mqa_q, padded_heads)
        out = mqa_q.new_empty(
            (num_tokens, padded_heads, self.kv_lora_rank), dtype=torch.bfloat16
        )
        trtllm_batch_decode_with_kv_cache_mla(
            query=mqa_q.unsqueeze(1),
            kv_cache=pages.unsqueeze(1),
            workspace_buffer=_get_workspace_buffer(mqa_q.device),
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=rows.unsqueeze(1),
            seq_lens=counts,
            max_seq_len=window,
            out=out.unsqueeze(1),
            bmm1_scale=self.scale * self._q_scale_float * self._k_scale_float,
            bmm2_scale=self._k_scale_float,
            sparse_mla_top_k=window,
            **_kv_counters(num_tokens, padded_heads),
        )
        if padded_heads != num_heads:
            out = out[:, :num_heads].contiguous()
        return out, None


class K3DFlash2DecoderLayer(DFlash2Qwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        if not causal or getattr(config, "mla_use_nope", False):
            raise NotImplementedError(
                "Kimi-K3 DFlash2 MLA drafts support causal RoPE layers only."
            )
        self.hidden_size = config.hidden_size
        self.self_attn = K3DFlash2Attention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_rope=True,
            sliding_window=sliding_window,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._init_convs(vllm_config, config, quant_config, prefix)


@ignore_torch_compile
class K3DFlash2Model(DFlash2Qwen3Model, K3DraftContextKV):
    decoder_layer_cls = K3DFlash2DecoderLayer

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        self.window_rows = _WindowRows()
        for layer in self.layers:
            layer.self_attn.window_rows = self.window_rows
        kv_width = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        self.context_kv_proj = MergedColumnParallelLinear(
            self.config.hidden_size,
            [kv_width] * self.config.num_hidden_layers,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(
                prefix, f"layers.{start_layer_id}.self_attn.fused_qkv_a_proj"
            ),
            disable_tp=True,
        )
        self._max_num_context_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )

    def _build_fused_kv_buffers(self) -> None:
        self._build_fused_context_kv_metadata()

    def context_kv_modules(self) -> list[nn.Module]:
        return [self.hidden_norm, self.fc, self.context_kv_proj]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self._precompute_fused_context_kv(
            self.hidden_norm(context_states), context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.window_rows.reset()
        return super().forward(input_ids, positions, input_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = _duplicate_context_kv_weights(weights, len(self.layers))
        return super().load_weights(weights)


class K3DFlash2ForCausalLM(DFlash2Qwen3ForCausalLM):
    model_cls = K3DFlash2Model

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.layer_name for layer in self.model.layers]
