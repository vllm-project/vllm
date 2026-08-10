# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA implementation of the Dots3Note language model."""

import copy
from collections.abc import Iterable, Iterator
from dataclasses import replace

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    DCPGroupColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV2MLAAttention,
    DeepseekV2MLP,
    DeepseekV2MoE,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    sequence_parallel_chunk,
)
from vllm.models.deepseek_v32.nvidia.model import (
    DeepseekV32DecoderLayer,
    DeepseekV32ForCausalLM,
    DeepseekV32Model,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from .attention import (
    Dots3NoteFlashAttnPrefillBackend,
    Dots3NotePaddedSparseBackend,
    Dots3NoteTritonMLABackend,
)


def _padded_mlp_size(
    intermediate_size: int,
    quant_config: QuantizationConfig | None,
    is_sequence_parallel: bool = False,
) -> int:
    block_size = getattr(quant_config, "weight_block_size", None)
    if block_size is None or is_sequence_parallel:
        return intermediate_size
    tp_size = get_tensor_model_parallel_world_size()
    blocks = (intermediate_size + block_size[0] - 1) // block_size[0]
    return ((blocks + tp_size - 1) // tp_size) * block_size[0] * tp_size


class Dots3NoteMoE(DeepseekV2MoE):
    """DeepSeek MoE with NOTE's block-padded shared expert."""

    def __init__(
        self,
        config,
        parallel_config,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ) -> None:
        num_shared_experts = config.n_shared_experts
        routed_config = copy.copy(config)
        object.__setattr__(routed_config, "n_shared_experts", None)
        super().__init__(
            config=routed_config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            reduce_results=False,
            prefix=prefix,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
        )
        self.n_shared_experts = num_shared_experts
        self.reduce_results = reduce_results
        self.shared_experts = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=_padded_mlp_size(
                config.moe_intermediate_size * num_shared_experts,
                quant_config,
                self.is_sequence_parallel,
            ),
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            is_sequence_parallel=self.is_sequence_parallel,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        gather_output = self.is_sequence_parallel and not already_sequence_parallel
        if gather_output:
            hidden_states = sequence_parallel_chunk(hidden_states)
        assert self.shared_experts is not None
        output = super().forward(
            hidden_states, already_sequence_parallel=True
        ) + self.shared_experts(hidden_states)
        if gather_output:
            return tensor_model_parallel_all_gather(output, 0)[:num_tokens]
        if self.reduce_results:
            return tensor_model_parallel_all_reduce(output)
        return output


def _forward_note_mla(
    attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    g_proj: nn.Module,
    k_rope_only_layernorm: nn.Module,
    attention_gate_type: str,
    q_lora_scale: float,
    kv_lora_scale: float,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    qkv_lora = attention.fused_qkv_a_proj(hidden_states)[0]
    q_c, kv_lora = qkv_lora.split(
        [
            attention.q_lora_rank,
            attention.kv_lora_rank + attention.qk_rope_head_dim,
        ],
        dim=-1,
    )
    q_c = attention.q_a_layernorm(q_c) * q_lora_scale
    kv_c, k_pe = kv_lora.split(
        [attention.kv_lora_rank, attention.qk_rope_head_dim], dim=-1
    )
    kv_c_normed = attention.kv_a_layernorm(kv_c) * kv_lora_scale
    k_pe = k_rope_only_layernorm(k_pe).unsqueeze(1)

    q = attention.q_b_proj(q_c)[0]
    heads = attention.num_heads
    if attention.dcp_q_replicate:
        heads *= attention.q_b_proj.group_size
    q = q.view(-1, heads, attention.qk_head_dim)
    q[..., attention.qk_nope_head_dim :], k_pe = attention.rotary_emb(
        positions, q[..., attention.qk_nope_head_dim :], k_pe
    )

    if attention.indexer and attention.is_sparse and not attention.skip_topk:
        attention.indexer(hidden_states, q_c, positions, attention.indexer_rope_emb)
    if llama_4_scaling is not None:
        q *= llama_4_scaling

    q_dcp_replicated = None
    if attention.dcp_q_replicate:
        q_dcp_replicated, q = q, attention.q_b_proj._local_view(q)
    attn_out = attention.mla_attn(
        q,
        kv_c_normed,
        k_pe,
        output_shape=(
            hidden_states.shape[0],
            attention.num_heads * attention.v_head_dim,
        ),
        q_dcp_replicated=q_dcp_replicated,
    )

    gate = g_proj(hidden_states)[0]
    if attention_gate_type == "headwise":
        if gate.shape[-1] != attention.num_heads:
            rank = get_tensor_model_parallel_rank()
            gate = gate.narrow(-1, rank * attention.num_heads, attention.num_heads)
        attn_out = attn_out.view(-1, attention.num_heads, attention.v_head_dim)
        gate = torch.sigmoid(gate.float()).to(attn_out.dtype)
        attn_out = (attn_out * gate.unsqueeze(-1)).flatten(-2)
    else:
        gate = torch.sigmoid(gate.float()).to(attn_out.dtype)
        attn_out = attn_out * gate
    return attention.o_proj(attn_out)[0]


class Dots3NotePaddedMLAAttention(MLAAttention):
    """MLA layer whose physical cache rows match NOTE's SWA rows."""

    def __init__(self, *args, physical_head_size: int, **kwargs) -> None:
        kwargs["attn_backend"] = Dots3NotePaddedSparseBackend
        super().__init__(*args, **kwargs)
        assert physical_head_size >= self.head_size
        self.physical_head_size = physical_head_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MLAAttentionSpec:
        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MLAAttentionSpec)
        return replace(spec, head_size=self.physical_head_size)


class Dots3NoteFullAttention(DeepseekV2MLAAttention):
    """NOTE DSA on the existing DeepSeek sparse MLA attention."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        quant_config = vllm_config.quant_config
        local_config = copy.copy(config)
        object.__setattr__(
            local_config,
            "rope_parameters",
            {
                "rope_type": "default",
                "rope_theta": config.rope_parameters["rope_theta"],
            },
        )
        super().__init__(
            vllm_config=vllm_config,
            config=local_config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=False,
        )
        wrapper = self._modules.pop("mla_attn")
        inner_attention = wrapper.mla_attn
        vllm_config.compilation_config.static_forward_context.pop(
            inner_attention.layer_name
        )
        prefill_backend_cls = (
            type(inner_attention.prefill_backend)
            if inner_attention.prefill_backend is not None
            else None
        )
        wrapper.mla_attn = Dots3NotePaddedMLAAttention(
            num_heads=inner_attention.num_heads,
            scale=inner_attention.scale,
            q_lora_rank=inner_attention.q_lora_rank,
            kv_lora_rank=inner_attention.kv_lora_rank,
            qk_nope_head_dim=inner_attention.qk_nope_head_dim,
            qk_rope_head_dim=inner_attention.qk_rope_head_dim,
            v_head_dim=inner_attention.v_head_dim,
            kv_b_proj=inner_attention.kv_b_proj,
            dcp_q_replicate=inner_attention.dcp_q_replicate,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=inner_attention.layer_name,
            use_sparse=True,
            indexer=wrapper.indexer,
            topk_indices_buffer=topk_indices_buffer,
            prefill_backend_cls=prefill_backend_cls,
            physical_head_size=(config.swa_kv_lora_rank + config.swa_qk_rope_head_dim),
        )
        gate_type = config.attention_gate_type
        gate_cls = ReplicatedLinear if gate_type == "headwise" else ColumnParallelLinear
        gate_size = (
            config.num_attention_heads
            if gate_type == "headwise"
            else config.num_attention_heads * config.v_head_dim
        )
        self.g_proj = gate_cls(
            config.hidden_size,
            gate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.k_rope_only_layernorm = RMSNorm(
            config.qk_rope_head_dim, eps=config.rms_norm_eps
        )
        self.attention_gate_type = gate_type
        self.q_lora_scale = 1.0
        self.kv_lora_scale = 1.0
        if config.apply_mla_qkv_lora_rescale:
            self.q_lora_scale = (config.hidden_size / config.q_lora_rank) ** 0.5
            self.kv_lora_scale = (config.hidden_size / config.kv_lora_rank) ** 0.5
        self.mla_attn = wrapper

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _forward_note_mla(
            self.mla_attn,
            positions,
            hidden_states,
            g_proj=self.g_proj,
            k_rope_only_layernorm=self.k_rope_only_layernorm,
            attention_gate_type=self.attention_gate_type,
            q_lora_scale=self.q_lora_scale,
            kv_lora_scale=self.kv_lora_scale,
            llama_4_scaling=llama_4_scaling,
        )


class Dots3NoteSlidingAttention(nn.Module):
    """NOTE SWA constructed directly as dense sliding-window MLA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        del topk_indices_buffer
        num_heads = config.swa_num_attention_heads
        qk_nope_head_dim = config.swa_qk_nope_head_dim
        qk_rope_head_dim = config.swa_qk_rope_head_dim
        v_head_dim = config.swa_v_head_dim
        q_lora_rank = config.swa_q_lora_rank
        kv_lora_rank = config.swa_kv_lora_rank
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        quant_config = vllm_config.quant_config

        fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            config.hidden_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        qrep_enabled = (
            envs.VLLM_DCP_Q_REPLICATE
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.prefill_context_parallel_size <= 1
        )
        q_proj_cls = (
            DCPGroupColumnParallelLinear if qrep_enabled else ColumnParallelLinear
        )
        q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        q_b_proj = q_proj_cls(
            q_lora_rank,
            num_heads * qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        gate_type = config.swa_attention_gate_type
        gate_cls = ReplicatedLinear if gate_type == "headwise" else ColumnParallelLinear
        gate_size = num_heads if gate_type == "headwise" else num_heads * v_head_dim
        g_proj = gate_cls(
            config.hidden_size,
            gate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        k_rope_only_layernorm = RMSNorm(qk_rope_head_dim, eps=config.rms_norm_eps)
        rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": config.swa_rope_theta,
            },
            is_neox_style=False,
        )
        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "fp8_ds_mla":
            cache_config = copy.copy(cache_config)
            cache_config.cache_dtype = "fp8"
        apply_rescale = config.apply_mla_qkv_lora_rescale
        self.hidden_size = config.hidden_size
        self.num_heads = num_heads // tp_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.fused_qkv_a_proj = fused_qkv_a_proj
        self.q_a_layernorm = q_a_layernorm
        self.q_b_proj = q_b_proj
        self.kv_a_layernorm = kv_a_layernorm
        self.kv_b_proj = kv_b_proj
        self.rotary_emb = rotary_emb
        self.o_proj = o_proj
        self.g_proj = g_proj
        self.k_rope_only_layernorm = k_rope_only_layernorm
        self.indexer = None
        self.indexer_rope_emb = None
        self.is_sparse = False
        self.skip_topk = False
        self.dcp_q_replicate = getattr(q_b_proj, "qrep_active", False)
        self.attention_gate_type = gate_type
        self.q_lora_scale = (
            (config.hidden_size / q_lora_rank) ** 0.5 if apply_rescale else 1.0
        )
        self.kv_lora_scale = (
            (config.hidden_size / kv_lora_rank) ** 0.5 if apply_rescale else 1.0
        )
        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=qk_head_dim**-0.5,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            dcp_q_replicate=self.dcp_q_replicate,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            sliding_window=config.sliding_window_size,
            attn_backend=Dots3NoteTritonMLABackend,
            prefill_backend_cls=Dots3NoteFlashAttnPrefillBackend,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _forward_note_mla(
            self,
            positions,
            hidden_states,
            g_proj=self.g_proj,
            k_rope_only_layernorm=self.k_rope_only_layernorm,
            attention_gate_type=self.attention_gate_type,
            q_lora_scale=self.q_lora_scale,
            kv_lora_scale=self.kv_lora_scale,
            llama_4_scaling=llama_4_scaling,
        )


class Dots3NoteDecoderLayer(DeepseekV32DecoderLayer):
    """DeepSeek-V3.2 decoder orchestration with NOTE-local modules."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if config is None:
            config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.hidden_size = config.hidden_size
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.use_mha = False
        attention_cls = (
            Dots3NoteSlidingAttention
            if config.layer_types[layer_idx] == "sliding_attention"
            else Dots3NoteFullAttention
        )
        self.self_attn = attention_cls(
            vllm_config=vllm_config,
            config=config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        is_moe = (
            layer_idx < config.num_hidden_layers
            and config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        if is_moe:
            self.mlp = Dots3NoteMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
                apply_routed_scale_to_output=False,
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=_padded_mlp_size(
                    config.intermediate_size, quant_config
                ),
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=False,
            )
        self.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and isinstance(self.mlp, DeepseekV2MoE)
        )
        self.tp_size = parallel_config.tensor_parallel_size
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


class Dots3NoteModel(DeepseekV32Model):
    """DeepSeek-V3.2 runtime shell with NOTE-local decoder layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size
        self.is_v32 = True
        self._weight_block_size = getattr(quant_config, "weight_block_size", None)
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Dots3NoteDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.aux_hidden_state_layers = tuple[int, ...]()
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )

    def _pad_dense_mlp_weight(
        self, name: str, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        block_size = self._weight_block_size
        if block_size is None or ".mlp.experts." in name:
            return loaded_weight
        if not any(
            proj_name in name
            for proj_name in (".gate_proj.", ".up_proj.", ".down_proj.")
        ):
            return loaded_weight
        dim = 1 if ".down_proj." in name else 0
        block_step = 1 if name.endswith("weight_scale_inv") else block_size[0]
        multiple = get_tensor_model_parallel_world_size() * block_step
        pad = (-loaded_weight.shape[dim]) % multiple
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def _adapt_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        pending_indexer_fp8: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        for name, weight in weights:
            if name.startswith("mtp."):
                continue
            if ".indexer." in name:
                parts = name.split(".")
                if "layers" in parts:
                    layer_idx = int(parts[parts.index("layers") + 1])
                    if (
                        layer_idx < self.config.num_hidden_layers
                        and self.config.layer_types[layer_idx] == "sliding_attention"
                    ):
                        continue

                projection = next(
                    (
                        candidate
                        for candidate in ("wk", "weights_proj")
                        if f".indexer.{candidate}." in name
                    ),
                    None,
                )
                is_fp8_weight = (
                    name.endswith(".weight") and weight.dtype == torch.float8_e4m3fn
                )
                is_scale = "weight_scale" in name
                if projection is not None and (is_fp8_weight or is_scale):
                    layer_prefix = name.rsplit(f".{projection}.", 1)[0]
                    key = (layer_prefix, projection)
                    entry = pending_indexer_fp8.setdefault(key, {})
                    entry["weight" if is_fp8_weight else "scale"] = weight
                    if "weight" not in entry or "scale" not in entry:
                        continue
                    weight_fp8 = entry["weight"]
                    scale_inv = entry["scale"]
                    del pending_indexer_fp8[key]
                    group_shape = GroupShape(
                        weight_fp8.shape[0] // scale_inv.shape[0],
                        weight_fp8.shape[1] // scale_inv.shape[1],
                    )
                    dequantized = scaled_dequantize(
                        weight_fp8,
                        scale_inv,
                        group_shape=group_shape,
                        out_dtype=torch.bfloat16,
                    )
                    yield f"{layer_prefix}.{projection}.weight", dequantized
                    continue
            yield name, self._pad_dense_mlp_weight(name, weight)
        if pending_indexer_fp8:
            missing = ", ".join(
                f"{prefix}.{proj}" for prefix, proj in pending_indexer_fp8
            )
            raise ValueError(f"Incomplete FP8 indexer weights: {missing}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(self._adapt_weights(weights))


class Dots3NoteLanguageModelForCausalLM(DeepseekV32ForCausalLM):
    model_cls = Dots3NoteModel
