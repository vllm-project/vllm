# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch MAMBA model."""
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import MambaConfig

from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (HasInnerState,
                                                   IsAttentionFree, SupportsPP)
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType

from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

KVCache = tuple[torch.Tensor, torch.Tensor]


class MambaDecoderLayer(nn.Module):

    def __init__(self,
                 config: MambaConfig,
                 model_config: Optional[ModelConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 is_lora_enabled: Optional[bool] = False,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.is_falcon_mamba = config.model_type == "falcon_mamba"
        self.is_lora_enabled = is_lora_enabled
        mixer_rms_eps = config.mixer_rms_eps if self.is_falcon_mamba else None
        self.mixer = MambaMixer(hidden_size=config.hidden_size,
                                ssm_state_size=config.state_size,
                                conv_kernel_size=config.conv_kernel,
                                intermediate_size=config.intermediate_size,
                                time_step_rank=config.time_step_rank,
                                use_conv_bias=config.use_conv_bias,
                                use_bias=config.use_bias,
                                use_rms_norm=self.is_falcon_mamba,
                                rms_norm_has_weight=not self.is_falcon_mamba,
                                rms_norm_eps=mixer_rms_eps,
                                activation=config.hidden_act,
                                is_lora_enabled=self.is_lora_enabled,
                                model_config=model_config,
                                cache_config=cache_config,
                                prefix=f"{prefix}.mixer")

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        self.mixer(hidden_states, output, mamba_cache_params)
        return output, residual


@support_torch_compile
class MambaModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        is_lora_enabled = bool(lora_config)

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embeddings = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MambaDecoderLayer(config,
                                             model_config=model_config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             is_lora_enabled=is_lora_enabled,
                                             prefix=prefix),
            prefix=f"{prefix}.layers")

        self.norm_f = RMSNorm(config.hidden_size,
                              eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: Optional[MambaCacheParams] = None,
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

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]

            layer_cache_params = None
            if mamba_cache_params is not None:
                layer_cache_params = mamba_cache_params.at_layer_idx(
                    i - self.start_layer)

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=layer_cache_params)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm_f(hidden_states, residual)

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "A_log" in name:
                name = name.replace("A_log", "A")
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MambaForCausalLM(nn.Module, HasInnerState, IsAttentionFree, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        self.scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "Mamba does not support prefix caching"

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.backbone = MambaModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "backbone"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        if config.tie_word_embeddings:
            self.lm_head = self.backbone.embeddings
        else:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.backbone.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):

        mamba_cache_params = None
        if not envs.VLLM_USE_V1:
            if self.mamba_cache is None:
                num_layers = self.model_config.get_num_layers_by_block_type(
                    self.vllm_config.parallel_config, LayerBlockType.mamba)
                state_shape = self.get_mamba_state_shape_from_config(
                    self.vllm_config)
                state_dtype = self.get_mamba_state_dtype_from_config(
                    self.vllm_config)
                self.mamba_cache = MambaCacheManager(self.vllm_config,
                                                     num_layers, *state_shape,
                                                     *state_dtype)

            mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.backbone(input_ids, positions, mamba_cache_params,
                                      intermediate_tensors, inputs_embeds)

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:

        return MambaStateDtypeCalculator.mamba1_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config

        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            intermediate_size=hf_config.intermediate_size,
            state_size=hf_config.state_size,
            conv_kernel=hf_config.conv_kernel,
            use_v1=envs.VLLM_USE_V1)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
