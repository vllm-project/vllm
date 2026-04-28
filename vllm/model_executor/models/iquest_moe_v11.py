# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
"""Inference-only OLMoE model compatible with HuggingFace weights."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.oe_embedding import OEEmbedding
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsLoRA
from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class IquestMoeRMSNorm(nn.Module):
    """RMSNorm (equivalent to T5LayerNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        result = (self.weight * hidden_states).to(input_dtype)
        return result

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class IquestMoEBlock(nn.Module):
    """A tensor-parallel MoE implementation for Olmoe that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(orig_shape)


class IquestMoeAttention(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.cache_config = vllm_config.cache_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 4096)

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.q_norm = IquestMoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # NOTE(yxing): check partial rope
        self.rotary_dim = getattr(config, "rotary_dim", None)
        rope_parameters = getattr(config, "rope_parameters", None)
        if self.rotary_dim is not None:
            partial_rotary_factor = self.rotary_dim / self.head_dim
            if rope_parameters is None:
                rope_parameters = {"partial_rotary_factor": partial_rotary_factor}
            else:
                rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )
        # NOTE(yxing): check shared kv cache
        self.shared_kv_num_layers = config.shared_kv_num_layers
        kv_sharing_target_layer_name = None
        self.cross_kv_cache = False
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        if self.shared_kv_num_layers:
            # use shared kv cache
            # attn name is like:
            # 'model.layers.0.self_attn.attn', 'model.layers.1.self_attn.attn'
            self.shared_kv_source_begin = config.shared_kv_source_begin
            self.shared_kv_target_begin = config.shared_kv_target_begin
            if (
                layer_idx >= self.shared_kv_target_begin
                and layer_idx < self.shared_kv_target_begin + self.shared_kv_num_layers
            ):
                current_layer_name = f"{prefix}.attn"
                layer_offset = layer_idx - self.shared_kv_target_begin
                target_layer_idx = self.shared_kv_source_begin + layer_offset
                kv_sharing_target_layer_name = current_layer_name.replace(
                    f"layers.{layer_idx}", f"layers.{target_layer_idx}"
                )
                self.cross_kv_cache = True
                self.k_norm = None
            else:
                self.k_norm = IquestMoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.k_norm = IquestMoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # NOTE(yxing): sink tokens
        self.num_sink_tokens = config.num_sink_tokens
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=self.cache_config,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            prefix=f"{prefix}.attn",
            enable_sinks_kv=self.num_sink_tokens > 0,
        )
        if self.num_sink_tokens:
            # TODO(yxing): refactor it to use cascade attention
            # use lse to merge attn states from sink_k and normal kv
            self.sink_k = torch.nn.Parameter(
                torch.zeros(
                    (self.num_sink_tokens, self.num_kv_heads, self.head_dim),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                ),
                requires_grad=False,
            )
            set_weight_attrs(self.sink_k, {"weight_loader": self.sinks_k_weight_loader})
            self.sinks_k = torch.zeros(
                (
                    self.max_num_seqs,
                    self.num_sink_tokens,
                    self.num_kv_heads,
                    self.head_dim,
                ),
                device=self.sink_k.device,
                dtype=config.torch_dtype,
            )
            self.sinks_v = torch.zeros_like(self.sinks_k)
            self.attn.populate_sinks_kv(sinks_k=self.sinks_k, sinks_v=self.sinks_v)

    def sinks_k_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        weight_shard_start = self.tp_rank * self.num_kv_heads
        weight_shard_end = (self.tp_rank + 1) * self.num_kv_heads
        loaded_weight = loaded_weight[weight_shard_start:weight_shard_end]

        # NOTE(yxing): load model with supporting tensor parallel
        new_loaded_weight = loaded_weight[:, None].expand(-1, self.head_dim)
        loaded_weight = new_loaded_weight[None, :, :]
        assert len(loaded_weight.shape) == 3, (
            f"expect shape of loaded_weight is 3-dim, now is {loaded_weight.shape}"
        )
        param.copy_(loaded_weight)
        self.sinks_k.copy_(
            param.unsqueeze(0).expand(self.max_num_seqs, -1, -1, -1).contiguous()
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        if self.cross_kv_cache:
            q, _ = self.rotary_emb(positions, q, None)
            attn_output = self.attn(q, None, None)
        else:
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class IquestMoeDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        self.layer_idx = extract_layer_index(prefix)

        self.self_attn = IquestMoeAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = IquestMoEBlock(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.attention_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_out_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.feed_forward_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.layer_idx == 0:
            self.ffn_out_norm = IquestMoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        # NOTE(yxing): post-norm is different for first layer and non-first layers
        if self.layer_idx == 0:
            norm_hidden_states = self.attention_norm(hidden_states)
            attn_output = self.self_attn(
                positions=positions, hidden_states=norm_hidden_states
            )
            h = hidden_states + self.attn_out_norm(attn_output)

            # fully connected
            hidden_states = self.mlp(self.feed_forward_norm(h))
            output = h + self.ffn_out_norm(hidden_states)
            return output
        else:
            x = self.attention_norm(hidden_states)
            attn_output = self.self_attn(positions=positions, hidden_states=x)
            h = x + self.attn_out_norm(attn_output)

            # fully connected
            ffn_out = self.mlp(self.feed_forward_norm(h))
            output = h + ffn_out
            return output


@support_torch_compile
class IquestMoeModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = IquestMoeDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = IquestMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_oe_embedding = vllm_config.model_config.get_enable_oe_embedding()
        if self.use_oe_embedding:
            self.over_encoding = OEEmbedding(vllm_config.model_config)

        self.num_sink_tokens = vllm_config.model_config.get_num_sink_tokens()

    def embed_input_ids(
        self, input_ids: torch.Tensor, oe_input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        embed_tokens = self.embed_tokens(input_ids)
        if self.use_oe_embedding:
            # TODO(yxing): currently, the input_ids should be tokens. In the
            # future, we need to support token embeddings
            return self.over_encoding(embed_tokens, oe_input_ids)
        else:
            return embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        oe_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TODO(yxing): support inputs_embeds later
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids, oe_input_ids=oe_input_ids)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(
                positions,
                hidden_states,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        # NOTE(yxing): for each expert, it includes
        # ('experts.w13_', 'experts.layer_idx.gate_proj.', expert_idx, 'w1'),
        # ('experts.w2_', 'experts.layer_idx.down_proj.', expert_idx, 'w2'),
        # ('experts.w13_', 'experts.layer_idx.up_proj.', expert_idx, 'w3')
        total_oe_heads = 0
        if self.use_oe_embedding:
            total_oe_heads = self.over_encoding.get_oe_total_heads()
        oe_heads_counter = 0
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if not self.num_sink_tokens and "sink_k" in name:
                        logger.warning_once("sink attention feature is disabled")
                        continue

                    if "experts.fc" in name:
                        # NOTE(yxing): for sonic moe model
                        # experts.fc -> experts.w13_.
                        # the shape of experts.fc is
                        #   [experts, 2 * intermidiate_size, hidden_size]
                        name = name.replace("experts.fc", "experts.w13_weight")
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        for expert_id in range(self.config.num_experts):
                            # w1 shard
                            weight_loader(
                                param,
                                loaded_weight[expert_id][
                                    : self.config.intermediate_size
                                ],
                                name,
                                shard_id="w1",
                                expert_id=expert_id,
                            )
                            # w3 shard
                            weight_loader(
                                param,
                                loaded_weight[expert_id][
                                    self.config.intermediate_size :
                                ],
                                name,
                                shard_id="w3",
                                expert_id=expert_id,
                            )
                        loaded_params.add(name)
                        continue

                    if not self.use_oe_embedding and "over_encoding" in name:
                        logger.warning_once("over encoding feature is disabled")
                        continue

                    if self.use_oe_embedding and "embedders" in name:
                        # NOTE(yxing): for over-encoding weights loader
                        # over_encoding.embedders.0.weight
                        # over_encoding.oe_embeder.weight
                        shard_id = int(name.split(".")[-2])
                        name = name.split(".")[0] + ".oe_embeder.weight"
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        oe_heads_counter += 1
                        if oe_heads_counter == total_oe_heads:
                            loaded_params.add(name)
                        continue

                    if "experts.proj" in name:
                        # NOTE(yxing): for sonic moe model
                        # experts.proj -> experts.w2_w2.
                        # the shape of experts.proj is
                        #       [experts, hidden_size, intermidiate_size]
                        name = name.replace("experts.proj", "experts.w2_weight")
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        for expert_id in range(self.config.num_experts):
                            weight_loader(
                                param,
                                loaded_weight[expert_id],
                                name,
                                shard_id="w2",
                                expert_id=expert_id,
                            )
                        loaded_params.add(name)
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name

                    # NOTE(yxing): for sonic moe model, the router name is:
                    #    layers.0.mlp.router.weight. We need to convert it to
                    #    layers.0.mlp.gate.weight
                    if "router.weight" in name:
                        name = name.replace("router.weight", "gate.weight")

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class IquestMoeV11ForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = IquestMoeDecoderLayer,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = IquestMoeModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        oe_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, inputs_embeds, oe_input_ids)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
