from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.transformers_utils.configs.eagle import EAGLEConfig


class EAGLE(nn.Module):

    def __init__(self, config: EAGLEConfig, *args, **kwargs) -> None:
        super().__init__()
        self.config = config

        architectures = getattr(self.config.model, "architectures", [])
        for arch in architectures:
            model_cls = ModelRegistry.load_model_cls(arch)
            if model_cls is not None:
                break

        self.model = model_cls(self.config.model, *args, **kwargs)
        self.fc = nn.Linear(config.model.hidden_size * 2,
                            config.model.hidden_size,
                            bias=False)

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
        return self.model.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        model_weights = []
        for name, loaded_weight in weights:
            if name.startswith("fc."):
                weight_loader = getattr(self.fc.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.fc.weight, loaded_weight)
            elif name.startswith("lm_head.") or name.startswith("model."):
                model_weights.append((name, loaded_weight))
            else:
                model_weights.append((f"model.{name}", loaded_weight))

        self.model.load_weights(model_weights)
