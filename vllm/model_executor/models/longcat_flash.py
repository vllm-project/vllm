# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Apache License, Version 2.0:
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
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Inference-only Flash model compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers import LongcatFlashConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction,
    composed_weight_loader,
    default_weight_loader,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def _scaled_weight_loader(scale: float) -> LoaderFunction:
    return composed_weight_loader(default_weight_loader, lambda x: x * scale)


def scale_mla_lora_norms_on_load(self_attn: nn.Module, config) -> None:
    """Fold LongCat's MLA LoRA scaling into the q/kv_a_layernorm weight loaders."""
    for norm, rank, enabled in (
        (self_attn.q_a_layernorm, config.q_lora_rank, config.mla_scale_q_lora),
        (self_attn.kv_a_layernorm, config.kv_lora_rank, config.mla_scale_kv_lora),
    ):
        if enabled:
            scale = (config.hidden_size / rank) ** 0.5
            set_weight_attrs(
                norm.weight, {"weight_loader": _scaled_weight_loader(scale)}
            )


class FlashMLP(nn.Module):
    """Flash MLP layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LongcatRouter(nn.Module):
    def __init__(
        self,
        config: LongcatFlashConfig,
        zero_expert_num: int,
        router_params_dtype: torch.dtype,
        prefix: str = "",
    ):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts + zero_expert_num
        self.classifier = GateLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            out_dtype=router_params_dtype,
            params_dtype=router_params_dtype,
            prefix=f"{prefix}.classifier",
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros((self.n_routed_experts), dtype=router_params_dtype)
        )

    def forward(self, hidden_states):
        logits, _ = self.classifier(hidden_states)
        return logits


class LongcatMoe(nn.Module):
    def __init__(
        self,
        config: LongcatFlashConfig,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.router_params_dtype = torch.float32

        self.router = LongcatRouter(
            config=config,
            zero_expert_num=config.zero_expert_num,
            router_params_dtype=self.router_params_dtype,
            prefix=f"{prefix}.gate",
        )

        assert config.zero_expert_type is not None
        self.experts = FusedMoEFactory(
            zero_expert_type=config.zero_expert_type,
            e_score_correction_bias=self.router.e_score_correction_bias,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=enable_eplb,
            routed_scaling_factor=config.routed_scaling_factor,
            router_logits_dtype=self.router_params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Align to MoERunner padded hidden size to avoid dim mismatch
        padded_hidden = self.experts.moe_config.hidden_dim
        if hidden_dim < padded_hidden:
            hidden_states_padded = torch.nn.functional.pad(
                hidden_states,
                (0, padded_hidden - hidden_dim),
                mode="constant",
                value=0.0,
            )
        else:
            hidden_states_padded = hidden_states

        router_logits_full = self.router(hidden_states_padded)

        # MoERunner handles routing memoization and zero expert computation
        # internally. Pass full router_logits (including zero experts) so that
        # zero experts can be properly identified in routing.
        final_hidden_states = self.experts(
            hidden_states=hidden_states_padded,
            router_logits=router_logits_full,  # Full logits (includes zero experts)
        )

        # Crop back to original hidden dimension if padded earlier
        if padded_hidden != hidden_dim:
            final_hidden_states = final_hidden_states[..., :hidden_dim]

        return final_hidden_states.view(num_tokens, hidden_dim)


class FlashDecoderLayer(nn.Module):
    """Flash decoder layer with dual attention and MLP structure."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: LongcatFlashConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = int(prefix.split(sep=".")[-1])
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # Dual attention structure
        self.self_attn = nn.ModuleList(
            [
                DeepseekV2MLAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    max_position_embeddings=max_position_embeddings,
                    cache_config=cache_config,
                    quant_config=None
                    if "self_attn" in getattr(config, "disable_quant_module", [])
                    else quant_config,
                    prefix=f"{prefix}.self_attn.{i}",
                )
                for i in range(2)
            ]
        )
        for self_attn in self.self_attn:
            scale_mla_lora_norms_on_load(self_attn, config)
        self.input_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for i in range(2)]
        )

        # Dual MLP structure
        self.mlps = nn.ModuleList(
            [
                FlashMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=None
                    if "mlps" in getattr(config, "disable_quant_module", [])
                    else quant_config,
                    prefix=f"{prefix}.mlps.{i}",
                )
                for i in range(2)
            ]
        )

        self.mlp = LongcatMoe(
            config=config,
            num_experts=config.n_routed_experts,
            top_k=config.moe_topk,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=(f"{prefix}.mlp"),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm[0](hidden_states)
        else:
            hidden_states, residual = self.input_layernorm[0](hidden_states, residual)

        hidden_states = self.self_attn[0](
            positions=positions,
            hidden_states=hidden_states,
            llama_4_scaling=None,
        )

        hidden_states, residual = self.post_attention_layernorm[0](
            hidden_states, residual
        )

        # moe
        hidden_states_copy = hidden_states.clone()
        moe_hidden_states = self.mlp(hidden_states_copy)

        # first mlp
        hidden_states = self.mlps[0](hidden_states)

        hidden_states, residual = self.input_layernorm[1](hidden_states, residual)

        # second_attn
        hidden_states = self.self_attn[1](
            positions=positions,
            hidden_states=hidden_states,
            llama_4_scaling=None,
        )
        hidden_states, residual = self.post_attention_layernorm[1](
            hidden_states, residual
        )

        # second_mlp
        hidden_states = self.mlps[1](hidden_states)

        hidden_states = hidden_states + moe_hidden_states

        return hidden_states, residual


@support_torch_compile
class FlashModel(nn.Module):
    """Flash model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_layers,
            lambda prefix: FlashDecoderLayer(
                vllm_config,
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        expert_params_mapping = self.get_expert_mapping()
        loaded_params: set[str] = set()

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # The MTP layers are loaded by LongCatFlashMTP.
            if name.startswith("mtp."):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp" in name and "mlps" not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, expert_shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if (
                        name_mapped.endswith(".bias") or name_mapped.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Skip loading kv_scale from ckpts towards new design.
                    if name.endswith(".kv_scale") and name not in params_dict:
                        continue
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LongcatFlashForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """Flash model for causal language modeling."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.quant_config = quant_config

        self.model = FlashModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
