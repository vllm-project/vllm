# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only FalconH1 model."""
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import FalconH1Config

from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba2_metadata import (
    Mamba2Metadata, prepare_mamba2_metadata)
from vllm.model_executor.layers.mamba.mamba_utils import get_mamba_state_shape
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .falcon_h1 import *
from .interfaces import HasInnerState, IsHybrid, SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

# class FalconH1MoEMLP(nn.Module):

#     def __init__(
#         self,
#         config: FalconH1Config,
#         quant_config: Optional[QuantizationConfig] = None,
#         bias: bool = False,
#     ) -> None:
#         super().__init__()

#         self.num_local_experts = config.expert_num
#         self.topk=config.topk

#         '''build experts'''
#         self.experts = torch.nn.ModuleList()
#         for _ in range(self.num_local_experts):
#             expert = FalconH1MLP(config)
#             self.experts.append(expert)

#         '''build router'''
#         self.weight = torch.nn.Parameter(
#             torch.empty((self.num_local_experts, config.hidden_size), dtype=torch.float32)
#         )
#         torch.nn.init.xavier_uniform_(self.weight)

#     def forward(self, x):

#         '''fixed parameters'''
#         inp_shape = x.shape # [token_num, hidden_size]
#         num_tokens = inp_shape[0]
#         hidden = inp_shape[-1]
#         num_experts = self.num_local_experts

#         """Routing , token-to-experts
#         Args:
#             input (torch.Tensor): Input tensor of shape [bs, seq, hidden].
#             weights (torch.Tensor): router's weights, [hidden, expert_num].
#         Returns:
#             routing_probs, token -> expert_prob
#             [[0.0000, 0.0000, 0.4006, 0.5994],
#             ...,
#             [0.0373, 0.0000, 0.9627, 0.0000]]
#             ------------
#             routing_map, token -> expert_idx
#             [[False, False,  True,  True],
#             ...,
#             [ True, False,  True, False]])
#         """
#         y = torch.mm(x, self.weight.to(x.dtype).t()) #y: [token_num, expert_num]
#         scores, top_indices = torch.topk(y, k=self.topk, dim=1)
#         probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(y)
#         routing_probs = torch.zeros_like(y).scatter(1, top_indices, probs)
#         routing_map = torch.zeros_like(y).int().scatter(1, top_indices, 1).bool()

#         """Dispatch: experts-to-tokens

#         Args: routing_map,routing_probs

#         Returns:
#             probs: [expert0{token4_prob, token2_prob,token8_prob}.....expertn]
#             x: [expert0{token4_idx, token2_idx, token8_idx}.....]

#         """
#         permuted_probs = None
#         num_local_tokens_per_expert = routing_map.sum(dim=0).long() # [token_num_e_1, ...., token_num_e_n]
#         num_out_tokens = routing_map.size(0) * self.topk
#         routing_map = routing_map.bool().T.contiguous() # expert-to-token, [expert_num, token_num]
#         '''
#         [False, False, False,  ..., False,  True,  True],
#         [False, False, False,  ...,  True, False, False],
#         [ True,  True,  True,  ...,  True,  True,  True],
#         [ True,  True,  True,  ..., False, False, False]]
#         '''
#         token_indices = (
#             torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
#         ) # [expert_num, token_num]
#         '''
#         [[   0,    1,    2,  ..., 1021, 1022, 1023],
#         [   0,    1,    2,  ..., 1021, 1022, 1023],
#         [   0,    1,    2,  ..., 1021, 1022, 1023],
#         [   0,    1,    2,  ..., 1021, 1022, 1023]]
#         '''

#         sorted_indices = token_indices.masked_select(routing_map) # [topk * token_num]
#         '''
#         [   8,    9,   12,  ..., 1015, 1016, 1017],
#         sorted_indices[:idx_1]->expert0
#         sorted_indices[idx_1:idx_2]->expert1
#         sorted_indices[idx_2:idx_3]->expert2
#         sorted_indices[idx_3:idx_4]->expert3
#         '''
#         probs = routing_probs.T.contiguous().masked_select(routing_map)  # [topk * token_num]
#         '''
#         [0.6458, 0.6458, 0.5577,  ..., 0.4983, 0.0520, 0.0520]
#         '''
#         x = x.index_select(0, sorted_indices) # [token_num * topk, hidden]

#         """compute:

#         Args:

#         Returns:

#         """
#         tokens_list = torch.split(x, num_local_tokens_per_expert.tolist())
#         probs_list = torch.split(probs, num_local_tokens_per_expert.tolist())

#         output_local_list = []

#         for expert, tokens, prob in zip(self.experts, tokens_list, probs_list):
#             output = expert(tokens) * prob.unsqueeze(-1)

#             output_local_list.append(output)
#         permuted_tokens = torch.cat(output_local_list, dim=0)

#         output_tokens = torch.zeros(
#         inp_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
#             )
#         # Scatter add the permuted_input back to the original positions
#         output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)

#         return output_tokens


class FalconH1SparseMoeBlock(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.expert_num:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")
        gate_multiplier, down_multiplier = config.mlp_multipliers
        self.experts = FusedMoE(num_experts=config.expert_num,
                                top_k=config.topk,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.intermediate_size,
                                reduce_results=False,
                                quant_config=quant_config,
                                renormalize=False,
                                gate_multiplier=gate_multiplier,
                                down_multiplier=down_multiplier,
                                prefix=f"{prefix}.experts")

        self.weight = ReplicatedLinear(config.hidden_size,
                                       config.expert_num,
                                       bias=False,
                                       quant_config=None)
        self.shared_expert = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1,
                                           hidden_dim)  # [token_num, hidden]
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = F.sigmoid(
                    self.shared_expert_gate(hidden_states)) * shared_output

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.weight(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states,
                                           router_logits=router_logits)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states)

        return final_hidden_states.view(orig_shape)


class FalconH1MoEParallelHybrid(nn.Module):
    """
    A hybrid decoder layer for FalconH1 where the input is processed
    in parallel through both the self-attention branch and the SSM (Mamba)
    branch. Their outputs are then summed to produce the final hidden state.

    This layer uses:
      - FalconH1AttentionDecoderLayer for the multi-head self-attention branch.
      - FalconH1SSMDecoderLayer for the state-space (Mamba) branch.
    """

    def __init__(
        self,
        config: FalconH1Config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # Instantiate the attention branch
        self.self_attn = FalconH1AttentionDecoderLayer(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        # In V1 all attention/ssm layers must have
        # different index in prefix
        ssm_layer_idx = config.num_hidden_layers + layer_idx
        ssm_prefix = prefix.split(".")[0] + f".{ssm_layer_idx}"

        # Instantiate the SSM branch
        self.mamba = FalconH1SSMDecoderLayer(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=ssm_prefix,
        )
        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.ssm_in_multiplier = config.ssm_in_multiplier

        self.attention_in_multiplier = config.attention_in_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier

        self.feed_forward = FalconH1SparseMoeBlock(config, prefix=prefix)

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Process input through the attention branch.
        # FalconH1AttentionDecoderLayer expects positions, hidden_states,
        # kv_cache, attn_metadata, and residual.
        attn_hidden, _ = self.self_attn(
            positions=positions,
            hidden_states=hidden_states * self.attention_in_multiplier,
            residual=residual,
            **kwargs,
        )

        # Process input through the SSM branch.
        # FalconH1SSMDecoderLayer expects hidden_states, attn_metadata,
        # residual, mamba_cache_params, and sequence_idx.
        ssm_hidden, _ = self.mamba(
            hidden_states=hidden_states * self.ssm_in_multiplier,
            residual=residual,
            mamba_cache_params=mamba_cache_params,
            mamba2_metadata=mamba2_metadata,
            **kwargs,
        )
        # Sum the outputs from both branches.
        # We assume both branches produce outputs of the same
        # dimensionality (config.hidden_size).
        hidden_states = (attn_hidden * self.attn_out_multiplier) + (
            ssm_hidden * self.ssm_out_multiplier)
        hidden_states = hidden_states + residual

        # feed-forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@support_torch_compile
class FalconH1MoEModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: FalconH1Config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank:

            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
            self.embedding_multiplier = config.embedding_multiplier
        else:
            self.embed_tokens = PPMissingLayer()
            self.embedding_multiplier = 1.0

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = FalconH1MoEParallelHybrid
            return layer_class(
                config,
                layer_idx,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.final_layernorm = RMSNorm(config.hidden_size,
                                           eps=config.rms_norm_eps)
        else:
            self.final_layernorm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        attn_metadata = get_forward_context().attn_metadata

        if not envs.VLLM_USE_V1:
            mamba2_metadata = prepare_mamba2_metadata(
                chunk_size=self.config.mamba_chunk_size,
                attn_metadata=attn_metadata,
            )
        else:
            # v1 get mamba2_metadata from forward_context
            mamba2_metadata = None

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds * self.embedding_multiplier
            else:
                hidden_states = (self.get_input_embeddings(input_ids) *
                                 self.embedding_multiplier)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            layer_mamba_cache_params = None
            if mamba_cache_params:
                layer_mamba_cache_params = mamba_cache_params.at_layer_idx(i)
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                mamba_cache_params=layer_mamba_cache_params,
                mamba2_metadata=mamba2_metadata,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
            })
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.expert_num)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:

            if "rotary_emb.inv_freq" in name:
                continue

            if "A_log" in name:
                name = name.replace("A_log", "A")

            if "mamba" in name:
                name = name.replace("mamba", "mamba.mamba")

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                # We have feed_forward.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to feed_forward.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for feed_forward.experts[0].gate_gate_up_proj, which breaks load.
                if "feed_forward.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if ((name.endswith(".bias") or name.endswith("_bias"))
                        and name not in params_dict):
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
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[
                        name]  # w_13: [expert_num, seq_len, intermediate_size]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    name_list = name.split(".")
                    if len(name_list) == 4 and name_list[
                            2] == "feed_forward" and name_list[3] == "weight":
                        name += ".weight"

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class FalconH1MoEForCausalLM(nn.Module, HasInnerState, SupportsLoRA,
                             SupportsPP, IsHybrid):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
        use_v1: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config
            use_v1: Get shapes for V1 (or V0)

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config

        intermediate_size = (int(hf_config.mamba_expand *
                                 hf_config.hidden_size)
                             if hf_config.mamba_d_ssm is None else
                             hf_config.mamba_d_ssm)

        return get_mamba_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=hf_config.mamba_n_groups,
            num_heads=hf_config.mamba_n_heads,
            head_dim=hf_config.mamba_d_head,
            state_size=hf_config.mamba_d_state,
            conv_kernel=hf_config.mamba_d_conv,
            use_v1=use_v1,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert (not cache_config.enable_prefix_caching
                ), "FalconH1 currently does not support prefix caching"

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = FalconH1MoEModel(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))
        self.tie_word_embeddings = config.tie_word_embeddings
        self.unpadded_vocab_size = config.vocab_size
        self.mamba_cache: Optional[MambaCacheManager] = None
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
            )
            self.lm_head_multiplier = config.lm_head_multiplier
            if self.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)
            # Used to track and store by the Mamba cache between steps.

            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                config.vocab_size,
                scale=config.lm_head_multiplier,
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        mamba_cache_params = None
        if not envs.VLLM_USE_V1:
            if self.mamba_cache is None:
                mamba_state_shape = \
                    self.get_mamba_state_shape_from_config(
                        self.vllm_config, use_v1=False)
                self.mamba_cache = MambaCacheManager(
                    self.vllm_config,
                    self.lm_head.weight.dtype if hasattr(
                        self.lm_head, 'weight') else torch.bfloat16,
                    self.config.num_hidden_layers,
                    *mamba_state_shape,
                )
            mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.model(
            input_ids,
            positions,
            mamba_cache_params,
            intermediate_tensors,
            inputs_embeds,
        )

        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        param_loaded = loader.load_weights(weights)
        if self.tie_word_embeddings:
            param_loaded.add("lm_head.weight")
        return param_loaded

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    # def load_weights(self, weights: Iterable[tuple[str,
    #                                                torch.Tensor]]) -> set[str]:
    #     stacked_params_mapping = [
    #         # (param_name, shard_name, shard_id)
    #         ("qkv_proj", "q_proj", "q"),
    #         ("qkv_proj", "k_proj", "k"),
    #         ("qkv_proj", "v_proj", "v"),
    #         ("gate_up_proj", "gate_proj", 0),
    #         ("gate_up_proj", "up_proj", 1),
    #     ]

    #     params_dict = dict(self.named_parameters())
    #     loaded_params: set[str] = set()
    #     for name, loaded_weight in weights:
    #         if "rotary_emb.inv_freq" in name:
    #             continue

    #         if "A_log" in name:
    #             name = name.replace("A_log", "A")

    #         if "mamba" in name:
    #             name = name.replace("mamba", "mamba.mamba")

    #         for param_name, weight_name, shard_id in stacked_params_mapping:
    #             if weight_name not in name:
    #                 continue

    #             name = name.replace(weight_name, param_name)
    #             # Skip loading extra bias for GPTQ models.
    #             if name.endswith(".bias") and name not in params_dict:
    #                 continue
    #             # Skip layers on other devices.
    #             if is_pp_missing_parameter(name, self):
    #                 continue
    #             param = params_dict[name]
    #             weight_loader = param.weight_loader
    #             weight_loader(param, loaded_weight, shard_id)
    #             break
    #         else:
    #             # Skip loading extra bias for GPTQ models.
    #             if name.endswith(".bias") and name not in params_dict:
    #                 continue
    #             if is_pp_missing_parameter(name, self):
    #                 continue
    #             if self.tie_word_embeddings and "lm_head" in name:
    #                 continue

    #             param = params_dict[name]
    #             weight_loader = getattr(param, "weight_loader",
    #                                     default_weight_loader)
    #             weight_loader(param, loaded_weight)
    #         loaded_params.add(name)

    #     if self.tie_word_embeddings:
    #         loaded_params.add("lm_head.weight")
    #     return loaded_params
