"""PyTorch MAMBA model."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import MambaConfig

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import _BATCH_SIZES_TO_CAPTURE, CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (HasInnerState,
                                                   IsAttentionFree)
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix

KVCache = Tuple[torch.Tensor, torch.Tensor]


class MambaDecoderLayer(nn.Module):

    def __init__(self,
                 config: MambaConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.config = config
        self.is_falcon_mamba = config.model_type == "falcon_mamba"
        mixer_rms_eps = config.mixer_rms_eps if self.is_falcon_mamba else None
        self.mixer = MambaMixer(hidden_size=config.hidden_size,
                                ssm_state_size=config.state_size,
                                conv_kernel_size=config.conv_kernel,
                                intermediate_size=config.intermediate_size,
                                time_step_rank=config.time_step_rank,
                                use_conv_bias=config.use_conv_bias,
                                use_bias=config.use_bias,
                                use_rms_norm=self.is_falcon_mamba,
                                rms_norm_eps=mixer_rms_eps,
                                activation=config.hidden_act)

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states, attn_metadata,
                                   mamba_cache_params)
        return hidden_states, residual


class MambaModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embeddings = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        decoder_layers = []
        for i in range(config.num_hidden_layers):
            decoder_layers.append(
                MambaDecoderLayer(config,
                                  cache_config=cache_config,
                                  quant_config=quant_config))
        self.layers = nn.ModuleList(decoder_layers)
        self.norm_f = RMSNorm(config.hidden_size,
                              eps=config.layer_norm_epsilon)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                mamba_cache_params=mamba_cache_params.at_layer_idx(i))
        hidden_states, _ = self.norm_f(hidden_states, residual)

        return hidden_states


class MambaForCausalLM(nn.Module, HasInnerState, IsAttentionFree):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, \
            "Mamba does not support prefix caching"

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
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
            )

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = get_sampler()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):
        if self.mamba_cache is None:
            max_batch_size = (VllmConfig.get_graph_batch_size(
                self.scheduler_config.max_num_seqs) if self.scheduler_config
                              else max(_BATCH_SIZES_TO_CAPTURE) + 2)
            self.mamba_cache = MambaCacheManager(
                self.lm_head.weight.dtype, self.config.num_hidden_layers,
                max_batch_size, *self._get_mamba_cache_shape())

        (
            mamba_cache_tensors,
            state_indices_tensor,
        ) = self.mamba_cache.current_run_tensors(input_ids, attn_metadata,
                                                 **kwargs)

        mamba_cache_params = MambaCacheParams(mamba_cache_tensors[0],
                                              mamba_cache_tensors[1],
                                              state_indices_tensor)

        hidden_states = self.backbone(input_ids, positions, attn_metadata,
                                      mamba_cache_params, inputs_embeds)

        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        conv_state_shape = (
            self.config.intermediate_size // world_size,
            self.config.conv_kernel - 1,
        )
        temporal_state_shape = (
            self.config.intermediate_size // world_size,
            self.config.state_size,
        )
        return conv_state_shape, temporal_state_shape

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "A_log" in name:
                name = name.replace("A_log", "A")
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
