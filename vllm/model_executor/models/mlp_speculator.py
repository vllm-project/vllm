import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs import MLPSpeculatorConfig


class MLPSpeculatorLayerNorm(nn.Module):
    """
    A L2 normalization implementation
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value
         fits in the range of your encoding scheme
         (i.e. fp16 requires eps >= 6e-8).
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
    ):
        super(MLPSpeculatorLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps

    def forward(self, x):
        xf = x
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        x = self.weight * x
        x = x + self.bias
        return x


class MLPSpeculator(nn.Module):

    def __init__(self, config: MLPSpeculatorConfig, **kwargs) -> None:
        super().__init__()
        self.n_predict = config.n_predict
        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.inner_dim = config.inner_dim if config.inner_dim != 0 \
            else config.emb_dim

        self.max_speculative_tokens = config.num_lookahead_tokens

        self.emb = nn.ModuleList([
            VocabParallelEmbedding(config.vocab_size,
                                   self.inner_dim,
                                   org_num_embeddings=config.vocab_size)
            for _ in range(self.max_speculative_tokens)
        ])

        self.proj = nn.ModuleList([
            nn.Linear((self.emb_dim if i == 0 else self.inner_dim),
                      self.inner_dim,
                      bias=False) for i in range(self.max_speculative_tokens)
        ])

        self.head = nn.ModuleList([
            nn.Linear(self.inner_dim, self.vocab_size, bias=False)
            for _ in range(self.max_speculative_tokens)
        ])
        self.ln = nn.ModuleList([
            MLPSpeculatorLayerNorm(self.inner_dim)
            for _ in range(self.max_speculative_tokens)
        ])

        self.state_weight = 0.5**(0.5 / config.n_predict)
        self.emb_weight = math.sqrt(
            (1 - self.state_weight**2) * (self.inner_dim / 2))
        self.activation = nn.GELU()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                config.vocab_size, 1.0)
        self.sampler = Sampler()

    def generate_proposals(
        self,
        input_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
        sampling_metadata: SamplingMetadata,
    ) -> List[SamplerOutput]:
        if num_predict_tokens > self.max_speculative_tokens:
            raise ValueError(f"Max speculative tokens for model is "
                             f"{self.max_speculative_tokens}, but "
                             f"{num_predict_tokens} were requested")

        # b x 1 x d
        previous_hidden_states = previous_hidden_states.unsqueeze(1)

        # b x 1
        last_tokens = input_ids.unsqueeze(1)

        next_tokens = []

        for head_index in range(num_predict_tokens):

            # Project and predict
            z = self.emb[head_index](last_tokens)  # b k d
            states = self.proj[head_index](previous_hidden_states)

            # Weighted add of state_weight*state and emb_weight*z
            # Let subsequent LN take care of denominator
            # state_weight is close to 1, so shouldn't be any precision issues
            states.add_(z, alpha=self.emb_weight / self.state_weight)

            states = self.activation(self.ln[head_index](states))  # b k d
            # TODO: not yet supporting top_k_tokens_per_head
            previous_hidden_states = states

            logits = self.logits_processor(self.head[head_index].weight,
                                           states, sampling_metadata)

            output = self.sampler(logits.flatten(0, 1), sampling_metadata)
            last_tokens = output.sampled_token_ids
            next_tokens.append(output)

        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            param = params_dict.get(name.replace("speculator.", ""))
            if param is not None:
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
