from typing import Optional, List, Iterable, Tuple

import torch.nn as nn
import torch
import math
from vllm.attention import AttentionMetadata
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sequence import SamplerOutput


class MLPSpeculatorLayerNorm(nn.Module):
    """
    A L2 normalization implementation
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    elementwise_scale_weight : torch.Tensor
        learned scaling term after normalization?
    elementwise_shift_bias : torch.Tensor
        learned bias term after normalization?
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value fits in the range of your encoding scheme (i.e. fp16 requires eps >= 6e-8).
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
    def __init__(
        self,
        config,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_predict = config.n_predict
        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.inner_dim = config.inner_dim if config.inner_dim != 0 else config.emb_dim
        self.emb = nn.ModuleList([
                VocabParallelEmbedding(config.vocab_size, self.inner_dim, org_num_embeddings=config.vocab_size)
                for _ in range(config.n_predict)
        ])

        self.proj = nn.ModuleList([
            nn.Linear((self.emb_dim if i == 0 else self.inner_dim), self.inner_dim, bias=False)
            for i in range(config.n_predict)
        ])

        self.head = nn.ModuleList([nn.Linear(self.inner_dim, self.vocab_size, bias=False) for _ in range(config.n_predict)])
        self.ln = nn.ModuleList([MLPSpeculatorLayerNorm(self.inner_dim) for _ in range(config.n_predict)])

        self.state_weight = 0.5 ** (0.5 / config.n_predict)
        self.emb_weight = math.sqrt((1 - self.state_weight ** 2) * (self.inner_dim / 2))
        self.activation = nn.GELU()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size, config.vocab_size, 1.0)
        self.sampler = Sampler()

    def generate_proposals(
        self,
        input_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:

        if self.first_decode_step:
            self.first_decode_step = False
        else:
            self.previous_hidden_state = self.previous_hidden_state.reshape(-1, self.n_predict + 1, self.previous_hidden_state.size(1))
            self.previous_hidden_state = self.previous_hidden_state.gather(
                1,
                (self.accepted_token_lengths - 1)[:, None, None].expand(-1, 1, self.previous_hidden_state.size(2))
            ).squeeze(1) # b x d

        # b x 1 x d
        self.previous_hidden_state = self.previous_hidden_state.reshape(self.previous_hidden_state.size(0), 1, self.previous_hidden_state.size(1))

        # b x 1
        last_tokens = input_ids.reshape(-1, 1)

        next_tokens = []

        for head_index in range(self.n_predict):

            # Project and predict
            z = self.emb[head_index](last_tokens)  # b k d
            state = self.proj[head_index](self.previous_hidden_state)

            # Weighted add of state_weight*state and emb_weight*z
            # Let subsequent LN take care of denominator
            # state_weight is close to 1, so shouldn't be any precision issues
            state = torch.add(state, z, alpha=self.emb_weight / self.state_weight)

            state = self.activation(self.ln[head_index](state))  # b k d
            # todo: not yet supporting top_k_tokens_per_head
            self.previous_hidden_state = state

            logits = self.logits_processor(self.head[head_index].weight, state,
                                           sampling_metadata)

            tmp = logits.reshape(-1,logits.size(2))
            output = self.sampler(tmp, sampling_metadata)
            last_tokens = output.sampled_token_ids
            next_tokens.append(output)

        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            param = params_dict[name.replace("speculator.", "")]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
