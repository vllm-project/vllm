# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedConfig

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.mistral.amd.indexer import MistralIndexer
from vllm.models.mistral.amd.ops import fusion_enabled
from vllm.models.mistral.amd.ops.ssmax import (
    fused_rope_ssmax_scaling_,
)
from vllm.models.mistral.amd.ssmax import SSMaxScaling
from vllm.models.mistral.common.layer_spec import MistralLayerSpec
from vllm.models.mistral.common.rope import MistralRotaryEmbedding
from vllm.utils.multi_stream_utils import execute_in_parallel
from vllm.utils.torch_utils import aux_stream
from vllm.v1.attention.backends.mla.index_group import (
    SparseMLAIndexGroupBuilder,
)


class _MistralMLA(MLAAttention):
    supports_pcp_dcp: ClassVar[bool] = True


class MistralMLAAttention(nn.Module):
    """Multi-head latent attention with optional sparse (DSA) selection.

    NoPE layers run the same 576-wide latent cache layout as RoPE layers: their
    rope query/key dims are zero, so attention reduces to the latent term.
    KV-sharing layers attend to the latent cache of ``spec.kv_source_layer``
    and skip their own cache writes. ``non_causal`` layers (DSpark drafts)
    attend bidirectionally within their uniform query block; their
    ``sliding_window`` counts keys back from the end of that block.
    For small batches the indexer runs on the aux stream, overlapping the
    Q/KV projections, and joins before attention reads its top-k.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PreTrainedConfig,
        spec: MistralLayerSpec,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
        index_group_builder: SparseMLAIndexGroupBuilder | None,
        zero_kv_inputs: tuple[torch.Tensor, torch.Tensor],
        quant_config: QuantizationConfig | None = None,
        non_causal: bool = False,
        sliding_window: int | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        if quant_config is None:
            quant_config = vllm_config.quant_config
        tp_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % tp_size == 0
        assert config.q_lora_rank is None, "q-LoRA is not supported"

        self.spec = spec
        self.num_local_heads = config.num_attention_heads // tp_size
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # Query/key head dim is fixed; NoPE layers spend the rope dims on nope.
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = (
            config.qk_nope_head_dim if spec.use_rope else self.qk_head_dim
        )
        self.v_head_dim = config.v_head_dim
        self.zero_kv_c, self.zero_k_pe = zero_kv_inputs

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        if not spec.shares_kv:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                config.hidden_size,
                self.kv_lora_rank + (self.qk_rope_head_dim if spec.use_rope else 0),
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # Registered here first so its weights load as `self_attn.kv_b_proj`.
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.o_proj",
        )
        self.ssmax = (
            SSMaxScaling(self.num_local_heads, tp_sharded=True)
            if getattr(config, "ssmax_scaling", False)
            else None
        )
        self.fuse_rope_ssmax = (
            self.ssmax is not None
            and spec.use_rope
            and fusion_enabled("mla_rope_ssmax")
        )
        self.indexer = (
            MistralIndexer(
                vllm_config,
                config,
                spec,
                cast(torch.Tensor, topk_indices_buffer),
                prefix=f"{prefix}.indexer",
            )
            if spec.has_indexer
            else None
        )
        self.indexer_stream = aux_stream() if spec.has_indexer else None
        self.indexer_events = (
            [torch.cuda.Event(), torch.cuda.Event()] if spec.has_indexer else []
        )

        kv_sharing_target_layer_name = None
        if spec.shares_kv:
            layers_prefix = prefix.rsplit(".", 2)[0]
            kv_sharing_target_layer_name = (
                f"{layers_prefix}.{spec.kv_source_layer}.self_attn.attn"
            )
        self.attn = _MistralMLA(
            num_heads=self.num_local_heads,
            scale=self.qk_head_dim**-0.5,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=None,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=spec.use_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
            index_group_builder=index_group_builder,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            non_causal_multi_token_decode=non_causal,
            sliding_window=sliding_window,
        )

    def _project_qkv(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        log_positions: torch.Tensor,
        rotary_emb: MistralRotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states)[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )

        if self.spec.shares_kv:
            kv_c_normed = self.zero_kv_c[:num_tokens]
            k_pe = self.zero_k_pe[:num_tokens]
        else:
            kv = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_c_normed = self.kv_a_layernorm(kv[:, : self.kv_lora_rank])
            if self.spec.use_rope:
                k_pe = kv[:, None, self.kv_lora_rank :]
            else:
                k_pe = self.zero_k_pe[:num_tokens]

        if self.fuse_rope_ssmax:
            assert self.ssmax is not None
            fused_rope_ssmax_scaling_(
                q,
                k_pe,
                positions,
                rotary_emb.cos_sin_cache,
                log_positions,
                self.ssmax.s,
                self.ssmax.b,
                self.qk_nope_head_dim,
                rotary_emb.is_neox_style,
            )
        elif self.spec.use_rope:
            q[..., self.qk_nope_head_dim :], k_pe = rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )
            if self.ssmax is not None:
                q = self.ssmax(q, log_positions)
        elif self.ssmax is not None:
            # NoPE: the zero rope dims are padded inside SSMax.
            q = self.ssmax(
                q, log_positions, out_head_dim=self.qk_head_dim + self.qk_rope_head_dim
            )
        else:
            q = F.pad(q, (0, self.qk_rope_head_dim))
        return q, kv_c_normed, k_pe

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        log_positions: torch.Tensor,
        rotary_emb: MistralRotaryEmbedding,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        indexer = self.indexer
        if indexer is None:
            q, kv_c_normed, k_pe = self._project_qkv(
                positions, hidden_states, log_positions, rotary_emb
            )
        else:
            # A fork/join must not straddle the eager break of a piecewise
            # capture; FULL graphs are captured outside of it.
            (q, kv_c_normed, k_pe), _ = execute_in_parallel(
                lambda: self._project_qkv(
                    positions, hidden_states, log_positions, rotary_emb
                ),
                [
                    lambda: indexer(
                        hidden_states,
                        positions,
                        log_positions,
                        rotary_emb if self.spec.indexer_use_rope else None,
                    )
                ],
                self.indexer_events[0],
                self.indexer_events[1:],
                [self.indexer_stream] if self.indexer_stream is not None else None,
                enable=num_tokens <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
                and not BreakableCUDAGraphCapture.is_active(),
            )
        if self.spec.use_sparse:
            self.attn.impl.record_logical_topk_ready()  # type: ignore[attr-defined]

        attn_out = self.attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(num_tokens, self.num_local_heads * self.v_head_dim),
        )
        return self.o_proj(attn_out)[0]
