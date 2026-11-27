# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Nemotron Shortcut language model for HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.utils import divide
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import IsHybrid, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .gdp import NemotronShortcutGDPMixer, gdp_state_dtype, gdp_state_shape
from .mlp import NemotronShortcutMLP
from .moe import NemotronShortcutMoE
from .sdsa import (
    SdsaAttention,
    SdsaBuffers,
    SdsaKVCacheGrouping,
    allocate_sdsa_buffers,
    get_sdsa_config,
)

logger = init_logger(__name__)


class NemotronShortcutAttention(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # NoPE: position information only enters the model through the GDP mixers.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_projected(*self.project(hidden_states))

    def project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        """The input projection, which precedes the attention op."""
        qkv, _ = self.qkv_proj(hidden_states)
        return (qkv,)

    def forward_projected(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class NemotronShortcutSparseAttention(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        sdsa_buffers: SdsaBuffers,
    ) -> None:
        super().__init__()
        assert config.num_key_value_heads == 1, "SDSA attends over one KV head"
        assert cache_config is not None
        self.num_heads = divide(
            config.num_attention_heads, get_tensor_model_parallel_world_size()
        )
        self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            total_num_kv_heads=1,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # A 128-dim query and key head: the indexer's key formats hold 128 values.
        if config.index_head_dim != 128:
            raise ValueError(
                f"SDSA serves index_head_dim 128, got {config.index_head_dim}"
            )
        self.indexer = nn.ModuleDict(
            {
                name: ReplicatedLinear(
                    config.hidden_size,
                    config.index_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.indexer.{name}",
                )
                for name in ("q_proj", "k_proj")
            }
        )
        self.attn = SdsaAttention(
            num_heads=self.num_heads,
            sdsa_config=get_sdsa_config(config, self.indexer["k_proj"]),
            buffers=sdsa_buffers,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_projected(*self.project(hidden_states))

    def project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The input projections, which precede the indexer and attention ops."""
        qkv, _ = self.qkv_proj(hidden_states)
        indexer_q, _ = self.indexer["q_proj"](hidden_states)
        indexer_k, _ = self.indexer["k_proj"](hidden_states)
        return qkv, indexer_q, indexer_k

    def forward_projected(
        self, qkv: torch.Tensor, indexer_q: torch.Tensor, indexer_k: torch.Tensor
    ) -> torch.Tensor:
        q, k, v = qkv.split(
            [self.num_heads * self.head_dim, self.head_dim, self.head_dim], dim=-1
        )
        output, _ = self.o_proj(self.attn(indexer_q, indexer_k, q, k, v))
        return output


class NemotronShortcutWideResidualRead(nn.Module):
    """Mixes the residual streams down to one backbone-width activation."""

    def __init__(self, config: PreTrainedConfig) -> None:
        super().__init__()
        self.num_streams = config.wide_residual_num_streams
        self.logit = nn.Parameter(torch.empty(self.num_streams))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        streams = streams.unflatten(-1, (self.num_streams, -1))
        factors = torch.sigmoid(self.logit.float()).to(streams.dtype)
        return torch.matmul(factors, streams)


class NemotronShortcutWideResidualConnection(nn.Module):
    """Per-branch read and write ports into the residual streams."""

    def __init__(self, config: PreTrainedConfig) -> None:
        super().__init__()
        self.num_streams = config.wide_residual_num_streams
        self.retention_max_forget = config.wide_residual_retention_max_forget
        self.read_logit = nn.Parameter(torch.empty(self.num_streams))
        self.write_logit = nn.Parameter(torch.empty(self.num_streams))
        if config.wide_residual_learned_retention:
            self.retention_logit = nn.Parameter(torch.empty(self.num_streams))
        else:
            self.register_parameter("retention_logit", None)

    def read(self, streams: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
        streams = streams.unflatten(-1, (self.num_streams, -1))
        factors = torch.sigmoid(self.read_logit.float()).to(streams.dtype)
        return norm(torch.matmul(factors, streams).to(norm.weight.dtype))

    def write(self, streams: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """`Y_k = gamma_k X_k + 2 sigmoid(w_k) U` for every stream `k`."""
        streams = streams.unflatten(-1, (self.num_streams, -1))
        update = update.to(streams.dtype).unsqueeze(-2)
        write = 2.0 * torch.sigmoid(self.write_logit.float())
        write = write.to(streams.dtype).unsqueeze(-1)
        if self.retention_logit is not None:
            retention = 1.0 - self.retention_max_forget * torch.sigmoid(
                -self.retention_logit.float()
            )
            streams = streams * retention.to(streams.dtype).unsqueeze(-1)
        return torch.addcmul(streams, update, write).flatten(-2)


class NemotronShortcutPlainResidualRead(nn.Module):
    """Wide-residual read for a block that carries one hidden state.

    The MTP head runs a shortcut-MoE block outside the residual streams, so the
    checkpoint has no read logits and the read is the identity.
    """

    def __init__(self, config: PreTrainedConfig | None = None) -> None:
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class NemotronShortcutPlainResidualConnection(nn.Module):
    """Ordinary pre-norm residual behind the wide-residual port interface."""

    def __init__(self, config: PreTrainedConfig | None = None) -> None:
        super().__init__()

    def read(self, hidden_states: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
        return norm(hidden_states.to(norm.weight.dtype))

    def write(self, hidden_states: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return hidden_states + update.to(hidden_states.dtype)


class NemotronShortcutGDPLayer(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = NemotronShortcutGDPMixer(
            config,
            model_config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.mixer",
        )
        self.residual_connection = NemotronShortcutWideResidualConnection(config)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        branch_input = self.residual_connection.read(streams, self.input_layernorm)
        return self.residual_connection.write(streams, self.mixer(branch_input))


class NemotronShortcutMLPLayer(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = NemotronShortcutMLP(
            config,
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.residual_connection = NemotronShortcutWideResidualConnection(config)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        branch_input = self.residual_connection.read(streams, self.input_layernorm)
        return self.residual_connection.write(streams, self.mlp(branch_input))


class NemotronShortcutAttentionLayer(nn.Module):
    @staticmethod
    def attention(
        config: PreTrainedConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        sdsa_buffers: SdsaBuffers | None,
    ) -> nn.Module:
        """SDSA if the model has SDSA's buffers, which it allocates when its
        config has index_topk; full attention if not."""
        if sdsa_buffers is None:
            return NemotronShortcutAttention(config, cache_config, quant_config, prefix)
        return NemotronShortcutSparseAttention(
            config, cache_config, quant_config, prefix, sdsa_buffers
        )

    def __init__(
        self,
        config,
        layer_idx,
        model_config,
        cache_config,
        quant_config,
        *_,
        prefix="",
        sdsa_buffers=None,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = self.attention(
            config, cache_config, quant_config, f"{prefix}.self_attn", sdsa_buffers
        )
        self.residual_connection = NemotronShortcutWideResidualConnection(config)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        x = self.residual_connection.read(streams, self.input_layernorm)
        return self.residual_connection.write(streams, self.self_attn(x))


class ShortcutMoELayer(nn.Module):
    """Mixer and MoE branches sharing one block.

    The routed branch reads the block input before the mixer writes back, so
    routing never observes the mixer output; the shared branch reads after.
    Subclasses own the mixer and split it at its attention or GDP op:
    `mixer_projections` runs the input projections and `mix` the rest.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        wide_residual: bool = True,
    ) -> None:
        super().__init__()
        # The MTP head reuses this block outside the residual streams, where the
        # ports degenerate to identity reads and ordinary residual adds.
        read_cls = (
            NemotronShortcutWideResidualRead
            if wide_residual
            else NemotronShortcutPlainResidualRead
        )
        conn_cls = (
            NemotronShortcutWideResidualConnection
            if wide_residual
            else NemotronShortcutPlainResidualConnection
        )
        self.shortcut_read = read_cls(config)
        self.shortcut_input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer_residual_connection = conn_cls(config)
        self.pre_moe_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = NemotronShortcutMoE(
            config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=f"{prefix}.moe",
        )
        self.shortcut_post_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.moe_shortcut_post_norm
            else None
        )
        self.moe_residual_connection = conn_cls(config)

    def mixer_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def mix(self, *projections: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        norm = self.shortcut_input_layernorm
        routed_input = norm(self.shortcut_read(streams).to(norm.weight.dtype))

        dispatch = self.moe.begin_staged(routed_input)
        mixer_input = self.mixer_residual_connection.read(streams, self.input_layernorm)
        projections = self.mixer_projections(mixer_input)
        # The attention or GDP op ends a piecewise CUDA graph piece, which must
        # not end with the dispatch still in flight.
        dispatch = self.moe.receive_staged(dispatch, projections)
        streams = self.mixer_residual_connection.write(streams, self.mix(*projections))

        shared_input = self.moe_residual_connection.read(
            streams, self.pre_moe_layernorm
        )
        combine = self.moe.run_staged_experts(dispatch, streams)
        moe_output = self.moe.finish_staged(combine, shared_input)
        if self.shortcut_post_norm is not None:
            moe_output = self.shortcut_post_norm(moe_output)
        return self.moe_residual_connection.write(streams, moe_output)


class ShortcutMoEAttentionLayer(ShortcutMoELayer):
    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        wide_residual: bool = True,
        sdsa_buffers: SdsaBuffers | None = None,
    ) -> None:
        super().__init__(
            config,
            layer_idx,
            model_config,
            cache_config,
            quant_config,
            parallel_config,
            prefix=prefix,
            wide_residual=wide_residual,
        )
        self.self_attn = NemotronShortcutAttentionLayer.attention(
            config, cache_config, quant_config, f"{prefix}.self_attn", sdsa_buffers
        )

    def mixer_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self.self_attn.project(hidden_states)

    def mix(self, *projections: torch.Tensor) -> torch.Tensor:
        return self.self_attn.forward_projected(*projections)


class NemotronShortcutMoEAttentionLayer(nn.Module):
    """Attention then MoE on a plain residual.

    The MTP block under `moe_shortcut_disable_mtp`: the routed and shared
    experts both read the post-attention state.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        sdsa_buffers: SdsaBuffers | None = None,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = NemotronShortcutAttentionLayer.attention(
            config, cache_config, quant_config, f"{prefix}.self_attn", sdsa_buffers
        )
        self.pre_moe_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = NemotronShortcutMoE(
            config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=f"{prefix}.moe",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states)
        )
        moe_input = self.pre_moe_layernorm(hidden_states)
        return hidden_states + self.moe(moe_input, moe_input)


class ShortcutMoEGDPLayer(ShortcutMoELayer):
    def __init__(
        self,
        config: PreTrainedConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            layer_idx,
            model_config,
            cache_config,
            quant_config,
            parallel_config,
            prefix=prefix,
        )
        self.mixer = NemotronShortcutGDPMixer(
            config, model_config, cache_config, quant_config, prefix=f"{prefix}.mixer"
        )

    def mixer_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self.mixer.project(hidden_states)

    def mix(self, *projections: torch.Tensor) -> torch.Tensor:
        return self.mixer.forward_projected(*projections)


def check_sdsa_indexer_loaded(model: nn.Module, loaded: set[str]) -> None:
    """Refuse to serve SDSA layers whose indexer the checkpoint did not provide."""
    if getattr(model, "_sdsa_indexer_validated", False):
        return
    has_sparse = any(
        isinstance(m, NemotronShortcutSparseAttention) for m in model.modules()
    )
    if has_sparse and not any(".indexer." in name for name in loaded):
        raise ValueError(
            "The checkpoint's config has index_topk, but the checkpoint has no "
            "SDSA indexer weights. Without index_topk the model runs full "
            'attention: --hf-overrides \'{"text_config": {"index_topk": null}}\'.'
        )

    if has_sparse:
        model._sdsa_indexer_validated = True


def skip_indexer_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Skip the SDSA indexer's weights, for a model that runs full attention.
    If the checkpoint has them, warn that the config has no `index_topk`."""
    warned = False
    for name, weight in weights:
        if ".indexer." not in name:
            yield name, weight
        elif not warned:
            warned = True
            logger.warning_once(
                "The checkpoint has SDSA indexer weights, but its config has no "
                "index_topk: running full attention and ignoring the indexer. "
                "To run SDSA, set index_topk: "
                '--hf-overrides \'{"text_config": {"index_topk": 2048}}\'.',
                scope="global",
            )


def load_nemotron_shortcut_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    mapper: WeightsMapper,
    check_indexer: bool = True,
) -> set[str]:
    """Load weights, skipping the SDSA indexer when the model runs dense, and
    check that SDSA layers got their indexer (unless the caller checks the
    whole load itself)."""
    if not any(isinstance(m, NemotronShortcutSparseAttention) for m in model.modules()):
        weights = skip_indexer_weights(weights)
    loaded = AutoWeightsLoader(model).load_weights(weights, mapper=mapper)
    if check_indexer:
        check_sdsa_indexer_loaded(model, loaded)
    return loaded


def check_supported(vllm_config: VllmConfig) -> None:
    """Refuse deployments the GDP mixers cannot serve, with every reason."""
    reasons = []
    if vllm_config.cache_config.use_replayssm:
        reasons.append(
            "ReplaySSM is not supported: the GDP verify kernel writes one state "
            "slot per draft token"
        )
    if reasons:
        raise ValueError(
            "Nemotron Shortcut cannot run this configuration: " + "; ".join(reasons)
        )


ALL_LAYER_TYPES = {
    "linear_attention": NemotronShortcutGDPLayer,
    "mlp": NemotronShortcutMLPLayer,
    "full_attention": NemotronShortcutAttentionLayer,
    "shortcut_moe_attn": ShortcutMoEAttentionLayer,
    "shortcut_moe_gdp": ShortcutMoEGDPLayer,
}


@support_torch_compile
class NemotronShortcutTextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        check_supported(vllm_config)
        config: PreTrainedConfig = vllm_config.model_config.hf_text_config
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )

        # The selection buffers that every SDSA layer of the model shares.
        self.sdsa_buffers = None
        if config.index_topk is not None:
            self.sdsa_buffers = allocate_sdsa_buffers(vllm_config, config.index_topk)

        def get_layer(prefix: str):
            layer_idx = extract_layer_index(prefix)
            layer_type = config.layer_types[layer_idx]
            args = (
                config,
                layer_idx,
                vllm_config.model_config,
                vllm_config.cache_config,
                vllm_config.quant_config,
                vllm_config.parallel_config,
            )
            if layer_type in ("full_attention", "shortcut_moe_attn"):
                return ALL_LAYER_TYPES[layer_type](
                    *args, prefix=prefix, sdsa_buffers=self.sdsa_buffers
                )
            return ALL_LAYER_TYPES[layer_type](*args, prefix=prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.residual_readout = NemotronShortcutWideResidualRead(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._make_empty_streams = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * config.wide_residual_num_streams
        )

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        # The streams cross PP stages in the dtype they are kept in.
        if self.config.residual_in_fp32:
            dtype = torch.float32
        return self._make_empty_streams(batch_size, dtype, device)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            # Every block reads and writes the replicated full-width stream.
            streams = inputs_embeds.repeat(1, self.config.wide_residual_num_streams)
            if self.config.residual_in_fp32:
                streams = streams.float()
        else:
            assert intermediate_tensors is not None
            streams = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            streams = layer(streams)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": streams})

        hidden_states = self.residual_readout(streams).to(self.norm.weight.dtype)
        return self.norm(hidden_states)


class NemotronShortcutForCausalLM(nn.Module, SupportsPP, IsHybrid, SdsaKVCacheGrouping):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.mtp.": None,  # consumed by NemotronShortcutMTP
            "model.language_model.": "model.",
            "model.vision_tower.": None,
            "model.multi_modal_projector.": None,
            "model.video_patch_conv.": None,
        },
        orig_to_new_stacked={
            ".self_attn.q_proj": (".self_attn.qkv_proj", "q"),
            ".self_attn.k_proj": (".self_attn.qkv_proj", "k"),
            ".self_attn.v_proj": (".self_attn.qkv_proj", "v"),
        },
    )
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    # A parent's AutoWeightsLoader calls load_weights once per contiguous run of
    # this model's weights, so a parent that nests it clears this and checks
    # the SDSA indexer once, after the whole load.
    check_sdsa_indexer_on_load = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.model = NemotronShortcutTextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return gdp_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config = vllm_config.model_config.hf_text_config
        return gdp_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.gdp_num_groups,
            config.gdp_num_heads,
            config.gdp_key_head_dim,
            config.gdp_head_dim,
            config.gdp_conv_kernel,
            config.gdp_num_householder,
            vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_nemotron_shortcut_weights(
            self,
            weights,
            self.hf_to_vllm_mapper,
            check_indexer=self.check_sdsa_indexer_on_load,
        )
