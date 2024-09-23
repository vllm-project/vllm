from typing import List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.config import CacheConfig
from typing import Iterable, List, Optional, Tuple
from .llama import LlamaModel
from vllm.sequence import IntermediateTensors


class LlamaForSequenceClassification(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.model = LlamaModel(config, **kwargs)
        self.score = RowParallelLinear(
            config.hidden_size,
            self.num_labels,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors=intermediate_tensors)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: Optional["SamplingMetadata"] = None) -> torch.Tensor:
        # Use the last token hidden state for classification
        # last_hidden_state = hidden_states[:, -1]
        return hidden_states


    def load_weights(self, weights: List[Tuple[str, torch.Tensor]]):
        model_weights = []
        score_weights = []
        
        for name, param in weights:
            if name.startswith('model.'):
                model_weights.append((name[6:], param))
            elif name.startswith('score.'):
                score_weights.append((name[6:], param))
        
        params_dict = dict(self.model.named_parameters())
        for name, loaded_weight in model_weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        
        score_state_dict = dict(score_weights)
        self.score.load_state_dict(score_state_dict, strict=False)


    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: "SamplingMetadata"
    ) -> Optional[SamplerOutput]:
        return SamplerOutput(outputs=logits, logprobs=None)