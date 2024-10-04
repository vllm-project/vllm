from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.eagle import EAGLEConfig


class EAGLE(nn.Module):
    """This class implements the EAGLE draft model from the paper: https://arxiv.org/pdf/2401.15077
    Reference implementation: https://github.com/SafeAILab/EAGLE
    
    Differences from reference implementation:
    1. In reference, LlamaDecoderLayer implementation doesn't have 
       input_layernorm for 1st decoder layer (https://github.com/SafeAILab/EAGLE/blob/7d065d084443fbfd386f88839efd7193c12be869/eagle/model/cnets.py#L427) 
       but we do as HF implementation also does.
    2. We allow any decoder layer to be used in EAGLE whereas in reference 
       decoder layer is fixed to be LlamaDecoderLayer.
    3. We have an optional token_map which reduces draft vocab to most 
       frequently used tokens to give some additional speed-up by reducing 
       sampling overhead. This is disabled unless the checkpoint file has 
       explicit token_map tensor and config has an optional attribute 
       truncated_vocab_size < vocab_size. To use this technique, one has to find
       the top-k most frequent tokens in target dataset and add that as a tensor
       in the draft checkpoint (using key token_map). Also, the draft config
       needs to have truncated_vocab_size (=k) as an attribute."""

    def __init__(self, config: EAGLEConfig, *args, **kwargs) -> None:
        super().__init__()
        self.config = config

        architectures = getattr(self.config.model, "architectures", [])
        model_cls, _ = ModelRegistry.resolve_model_cls(architectures)

        self.model = model_cls(self.config.model, *args, **kwargs)
        self.fc = nn.Linear(config.model.hidden_size * 2,
                            config.model.hidden_size,
                            bias=getattr(self.config, "bias", False))

        self.orig_vocab_size = config.vocab_size
        self.truncated_vocab_size = config.truncated_vocab_size
        self.unpadded_vocab_size = self.truncated_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=self.truncated_vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.truncated_vocab_size,
                                                logit_scale)

        # Token map is a idx to token mapping to reduce the vocab size for
        # the draft model. Using smaller vocab size for draft, containing
        # only most frequent tokens reduces the speculation overhead. This
        # doesn't affect the acceptance rate much and thus gives more speed
        # -up. By default, this is disabled and is only used if the EAGLE
        # checkpoint file has token_map tensor.
        self.token_map = None

    @property
    def sampler(self):
        return self.model.sampler

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:

        tok_embeds = self.model.model.embed_tokens(input_ids)
        inputs_embeds = self.fc(
            torch.cat([tok_embeds, previous_hidden_states], dim=-1))

        inputs_embeds[positions == 0] = 0  # masking inputs at position=0

        hidden_states = self.model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        if self.token_map is not None:
            _logits = logits
            logits = -torch.inf * torch.ones(
                size=(*_logits.shape[:-1], self.orig_vocab_size),
                device=_logits.device,
                dtype=_logits.dtype)

            logits[..., self.token_map] = _logits

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # This implementation is incompitable with https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-8B
        # due to missing lm_head weights and its config being that of a
        # Llama model. Here's a compatible version with the same weights:
        # https://huggingface.co/abhigoyal/EAGLE-LLaMA3-Instruct-8B-vllm
        # Also, here's an example script for converting trained EAGLE
        # checkpoint to vLLM compatible version: https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d
        model_weights = {}
        for name, loaded_weight in weights:
            if name == "token_map":
                if self.config.truncated_vocab_size < self.config.vocab_size:
                    self.token_map = nn.Parameter(loaded_weight,
                                                  requires_grad=False)
            elif name.startswith("fc.weight"):
                weight_loader = getattr(self.fc.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.fc.weight, loaded_weight)
            elif name.startswith("fc.bias"):
                if self.fc.bias is not None:
                    weight_loader = getattr(self.fc.bias, "weight_loader",
                                            default_weight_loader)
                    weight_loader(self.fc.bias, loaded_weight)
                else:
                    raise ValueError("Found bias in the loaded weights "
                                     "but the model config doesn't have bias")
            elif name.startswith("model.lm_head.") or name.startswith(
                    "model.model."):
                model_weights[name.split("model.", 1)[-1]] = loaded_weight
            elif name.startswith("lm_head.") or name.startswith("model."):
                model_weights[name] = loaded_weight
            else:
                model_weights[f"model.{name}"] = loaded_weight

        lm_head_weight = model_weights.pop("lm_head.weight")

        if self.token_map is not None and\
            lm_head_weight.shape[0] > self.token_map.shape[0]:

            lm_head_weight = lm_head_weight[self.token_map]

        weight_loader = getattr(self.lm_head.weight, "weight_loader",
                                default_weight_loader)
        weight_loader(self.lm_head.weight, lm_head_weight)

        self.model.load_weights(model_weights.items())
