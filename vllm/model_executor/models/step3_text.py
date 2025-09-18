# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Jurassic model."""
from collections.abc import Iterable
from itertools import islice
from typing import Any, Optional

import torch
from torch import nn

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import AFDConfig, CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.afd_transfer.afd_connector.metadata import (
    AFDConnectorMetadata)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)

logger = init_logger(__name__)


class FusedMoEBlock(nn.Module):

    def __init__(self,
                 config: ModelConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.moe_num_experts}.")

        self.experts = FusedMoE(num_experts=config.moe_num_experts,
                                top_k=config.moe_top_k,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_intermediate_size,
                                reduce_results=False,
                                renormalize=config.norm_expert_weight,
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts")
        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.moe_num_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(orig_shape)


class Step3TextMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        intermediate_act = self.act_fn(gate_up)
        output, _ = self.down_proj(intermediate_act)
        return output


class Step3TextAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        rope_theta: int,
        share_q_dim: Optional[int] = None,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embedding: int = 8192,
        head_dim: int = 256,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        if num_kv_heads != 1:
            raise ValueError(f"Step3TextAttention num_kv_heads must be 1, "
                             f"but got {num_kv_heads}.")
        self.num_kv_heads = num_kv_heads

        self.head_dim = head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.q_size = share_q_dim if share_q_dim else self.head_dim

        self.qkv_proj = ReplicatedLinear(
            hidden_size,
            self.q_size + self.kv_size * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.inter_norm = RMSNorm(self.q_size, eps=norm_eps)
        self.wq = ColumnParallelLinear(
            self.q_size,
            self.head_dim * self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq",
        )
        self.rotary_emb = get_rope(self.head_dim,
                                   rotary_dim=self.head_dim,
                                   max_position=max_position_embedding,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling)
        scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scaling,
                              self.num_kv_heads,
                              cache_config=cache_config,
                              prefix=f"{prefix}.attn")

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.inter_norm(q)
        q = self.wq(q)[0]
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        residual, _ = self.o_proj(attn_output)
        return residual


class Step3TextDecoderLayer(nn.Module):

    def __init__(self,
                 config: ModelConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 afd_config: Optional[AFDConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        config = config.hf_config
        self.hidden_size = config.hidden_size
        rope_scaling = getattr(config, "rope_scaling", None)
        self.layer_idx = int(prefix.split("layers.")[1].split(".")[0])

        self.self_attn = Step3TextAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            norm_eps=config.rms_norm_eps,
            max_position_embedding=config.max_position_embedding,
            head_dim=config.head_dim,
            share_q_dim=config.share_q_dim,
            rope_theta=config.rope_theta,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn")

        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layers_idx = [
                int(i) for i in moe_layers_enum.strip().split(',')
            ]
        else:
            moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]

        if self.layer_idx in moe_layers_idx:
            self.moe = FusedMoEBlock(config=config,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.moe")
            self.share_expert = Step3TextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.share_expert_dim,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=f"{prefix}.share_expert")
            self.use_moe = True
        else:
            self.mlp = Step3TextMLP(hidden_size=config.hidden_size,
                                    intermediate_size=config.intermediate_size,
                                    hidden_act="silu",
                                    quant_config=quant_config,
                                    prefix=f"{prefix}.mlp")
            self.use_moe = False
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.graph_capture_active: bool = False
        self.should_capture_graph: bool = (afd_config
                                           and afd_config.is_attention_server)
        self.graph_attn_runners_by_stage: dict[int, dict[
            int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor,
                       Optional[torch.Tensor], torch.Tensor,
                       Optional[torch.Tensor]]]] = {}
        self.graph_capture_sizes: list[int] = []

    def _capture_cuda_graph_for_size(self, *, stage_idx: int, num_tokens: int,
                                     device: torch.device,
                                     hs_dtype: torch.dtype,
                                     pos_dtype: torch.dtype) -> None:
        if not self.graph_capture_active:
            return
        stage_graphs = self.graph_attn_runners_by_stage.setdefault(
            stage_idx, {})
        if num_tokens in stage_graphs:
            return

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            static_positions = torch.zeros(num_tokens,
                                           dtype=pos_dtype,
                                           device=device)
            static_hidden_states = torch.empty((num_tokens, self.hidden_size),
                                               dtype=hs_dtype,
                                               device=device)
            static_residual = torch.empty(
                (num_tokens, self.hidden_size), dtype=hs_dtype,
                device=device) if self.layer_idx > 0 else None

            self._compute_attn_output(static_hidden_states, static_residual,
                                      static_positions)

            with torch.cuda.graph(graph, stream=stream):
                static_hs_out, static_residual_out = self._compute_attn_output(
                    static_hidden_states, static_residual, static_positions)

        torch.cuda.current_stream().wait_stream(stream)
        stage_graphs[num_tokens] = (graph, static_positions,
                                    static_hidden_states, static_residual,
                                    static_hs_out, static_residual_out)
        if num_tokens not in self.graph_capture_sizes:
            self.graph_capture_sizes.append(num_tokens)
            self.graph_capture_sizes.sort()

    def _ensure_graph_for_size(self, *, stage_idx: int, size: int,
                               device: torch.device, hs_dtype: torch.dtype,
                               pos_dtype: torch.dtype) -> None:
        if not self.graph_capture_active:
            return
        stage_graphs = self.graph_attn_runners_by_stage.get(stage_idx)
        if stage_graphs is None or size not in stage_graphs:
            self._capture_cuda_graph_for_size(stage_idx=stage_idx,
                                              num_tokens=size,
                                              device=device,
                                              hs_dtype=hs_dtype,
                                              pos_dtype=pos_dtype)

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_moe:
            share_output = self.share_expert(hidden_states)
            moe_output = self.moe(hidden_states)
            return share_output + moe_output
        return self.mlp(hidden_states)

    def _compute_attn_output(
            self, hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor],
            positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        return hidden_states, residual

    def compute_attn_output(
            self, hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor],
            positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.should_capture_graph:
            return self._compute_attn_output(hidden_states, residual,
                                             positions)

        device = hidden_states.device
        hs_dtype = hidden_states.dtype
        pos_dtype = positions.dtype
        num_tokens = hidden_states.shape[0]
        afd_stage_idx = 0
        forward_ctx = get_forward_context()
        if forward_ctx.afd_metadata is not None:
            afd_stage_idx = forward_ctx.afd_metadata.afd_stage_idx

        self._ensure_graph_for_size(stage_idx=afd_stage_idx,
                                    size=num_tokens,
                                    device=device,
                                    hs_dtype=hs_dtype,
                                    pos_dtype=pos_dtype)

        stage_graphs = self.graph_attn_runners_by_stage.get(afd_stage_idx, {})
        chosen_size = None
        for size in self.graph_capture_sizes:
            if size >= num_tokens and size in stage_graphs:
                chosen_size = size
                break

        if chosen_size is None:
            return self._compute_attn_output(hidden_states, residual,
                                             positions)

        (graph, static_positions, static_hidden_states, static_residual,
         static_hs_out, static_residual_out) = stage_graphs[chosen_size]

        static_positions[:num_tokens].copy_(positions)
        static_hidden_states[:num_tokens].copy_(hidden_states)
        if residual is not None and static_residual is not None:
            static_residual[:num_tokens].copy_(residual)
        graph.replay()

        out_hidden = static_hs_out[:num_tokens].clone()
        if static_residual_out is not None:
            out_residual = static_residual_out[:num_tokens].clone()
        else:
            out_residual = out_hidden.clone()
        return out_hidden, out_residual

    def forward(
            self, positions: torch.Tensor, hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.compute_attn_output(
            hidden_states, residual, positions)
        ffn_output = self.compute_ffn_output(hidden_states)
        return ffn_output, residual


@support_torch_compile
class Step3TextModel(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.config = config

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Step3TextDecoderLayer(
                config=vllm_config.model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                afd_config=vllm_config.afd_config,
                prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def set_graph_capture_mode(self, enabled: bool) -> None:
        for idx in range(self.start_layer, self.end_layer):
            layer = self.layers[idx]
            if hasattr(layer, "graph_capture_active"):
                layer.graph_capture_active = enabled

    def compute_ffn_output(self, layer_idx: int,
                           hidden_states: torch.Tensor) -> torch.Tensor:
        layer = self.layers[layer_idx]
        return layer.compute_ffn_output(hidden_states)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        forward_ctx = get_forward_context()
        afd_metadata = (forward_ctx.afd_metadata
                        if forward_ctx is not None else None)

        if afd_metadata is not None:
            assert residual is None, "PP is not supported with AFD"
            num_stages = len(afd_metadata.afd_tokens_start_loc) - 1
            afd_connector = afd_metadata.afd_connector

            stage_hidden_states: list[torch.Tensor] = []
            stage_residual: list[Optional[torch.Tensor]] = []
            stage_positions: list[torch.Tensor] = []

            for stage_idx in range(num_stages):
                start = afd_metadata.afd_tokens_start_loc[stage_idx]
                end = start + afd_metadata.afd_tokens_lens[stage_idx]
                stage_hidden_states.append(hidden_states[start:end].clone())
                stage_residual.append(residual[start:end].clone(
                ) if residual is not None else None)
                stage_positions.append(positions[start:end])

            for layer_idx in range(self.start_layer, self.end_layer):
                layer = self.layers[layer_idx]

                for stage_idx in range(num_stages):
                    afd_metadata.afd_stage_idx = stage_idx

                    if layer_idx > 0:
                        stage_hidden_states[stage_idx].copy_(
                            afd_connector.recv_ffn_output())

                    current_hidden = stage_hidden_states[stage_idx]
                    current_residual = stage_residual[stage_idx]
                    current_positions = stage_positions[stage_idx]

                    current_hidden, current_residual = \
                        layer.compute_attn_output(
                            current_hidden, current_residual,
                            current_positions)

                    metadata = AFDConnectorMetadata.create_attention_metadata(
                        layer_idx=layer_idx,
                        stage_idx=stage_idx,
                        seq_len=current_hidden.shape[0],
                        dtype=current_hidden.dtype,
                        device=current_hidden.device,
                    )
                    afd_connector.send_attn_output(current_hidden, metadata)
                    stage_residual[stage_idx] = current_residual

            for stage_idx in range(num_stages):
                recv_hidden = afd_connector.recv_ffn_output()
                stage_hidden_states[stage_idx].copy_(recv_hidden)

            hidden_states = torch.cat([
                stage_hidden_states[i][:afd_metadata.afd_tokens_lens[i]]
                for i in range(num_stages)
            ],
                                      dim=0)

            if stage_residual[0] is not None:
                residual = torch.cat([
                    stage_residual[i][:afd_metadata.afd_tokens_lens[i]]
                    if stage_residual[i] is not None else
                    stage_hidden_states[i][:afd_metadata.afd_tokens_lens[i]]
                    for i in range(num_stages)
                ],
                                     dim=0)
            else:
                residual = None
        else:
            for layer in islice(self.layers, self.start_layer, self.end_layer):
                hidden_states, residual = layer(positions, hidden_states,
                                                residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Step3TextForCausalLM(nn.Module, SupportsPP):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.vllm_config = vllm_config

        self.model = Step3TextModel(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                if not lora_config else lora_config.lora_vocab_padding_size,
            )
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size)
            self.sampler = get_sampler()
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None):
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        qkv_params_mapping = [
            # (param_name, shard_name, relative_start_idx, relative_end_idx)
            (".qkv_proj", ".q_proj", 0, self.config.share_q_dim /
             (self.config.share_q_dim + self.config.head_dim * 2)),
            (".qkv_proj", ".k_proj", self.config.share_q_dim /
             (self.config.share_q_dim + self.config.head_dim * 2),
             (self.config.share_q_dim + self.config.head_dim) /
             (self.config.share_q_dim + self.config.head_dim * 2)),
            (".qkv_proj", ".v_proj",
             (self.config.share_q_dim + self.config.head_dim) /
             (self.config.share_q_dim + self.config.head_dim * 2),
             (self.config.share_q_dim + self.config.head_dim * 2) /
             (self.config.share_q_dim + self.config.head_dim * 2)),
        ]
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        expert_params_mapping = [
            (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
            (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
            (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2")
        ]

        disable_moe_stacked_params = [
            data[1] for data in expert_params_mapping
        ]

        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if any(disable_moe_stacked_param in name
                       for disable_moe_stacked_param in
                       disable_moe_stacked_params):
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    for expert_id in range(loaded_weight.shape[0]):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(param,
                                      loaded_weight_expert,
                                      name,
                                      shard_id=shard_id,
                                      expert_id=expert_id)
                    loaded_params.add(name)
                    break
                else:
                    for (param_name, weight_name, start_idx,
                         end_idx) in qkv_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        dim = param.shape[param.output_dim]
                        begin_idx = int(start_idx * dim)
                        end_idx = int(end_idx * dim)
                        param_slice = param.narrow(param.output_dim, begin_idx,
                                                   end_idx - begin_idx)
                        param_slice.copy_(loaded_weight)
                        loaded_params.add(name)
                        break
                    else:
                        if is_pp_missing_parameter(name, self):
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
        return loaded_params
