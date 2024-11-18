import math
from typing import Iterable, List, Set, Tuple

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

SQRT2 = 2**0.5


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
    elementwise_scale_and_shift : bool
        Include a learned scaling and shift term after normalization.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale_and_shift=True,
    ):
        super().__init__()
        self.elementwise_scale_and_shift = elementwise_scale_and_shift
        if self.elementwise_scale_and_shift:
            self.weight = nn.Parameter(torch.empty(normalized_shape))
            self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps

    def forward(self, x):
        xf = x
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale_and_shift:
            x = self.weight * x
            x = x + self.bias
        return x


class MLPSpeculator(nn.Module):
    """
    An implementation of the speculative models introduced in
    "Accelerating Production LLMs with Combined Token/Embedding
    Speculators"
    https://arxiv.org/pdf/2404.19124

    Trained speculators of this type are available on HF hub at:
    https://huggingface.co/ibm-fms and https://huggingface.co/ibm-granite
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.n_predict = config.n_predict
        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.inner_dim = config.inner_dim if config.inner_dim != 0 \
            else config.emb_dim

        self.max_speculative_tokens = config.num_lookahead_tokens

        self.tie_weights = config.tie_weights
        self.scale_input = config.scale_input

        if self.tie_weights:
            assert (
                self.n_predict >
                1), "You cannot tie weights between stages when only 1 exists"
            embedding = VocabParallelEmbedding(
                config.vocab_size,
                self.inner_dim,
                org_num_embeddings=config.vocab_size)
            self.emb = nn.ModuleList([embedding] * self.max_speculative_tokens)

            # the initial projection from the base model may
            # have a different size, so that stays separate.
            proj_first = nn.Linear(self.emb_dim, self.inner_dim, bias=False)
            proj_tied = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
            self.proj = nn.ModuleList([proj_first] + [proj_tied] *
                                      (self.max_speculative_tokens - 1))

            head = ParallelLMHead(self.vocab_size, self.inner_dim, bias=False)
            self.head = nn.ModuleList([head] * self.max_speculative_tokens)

            ln = MLPSpeculatorLayerNorm(self.inner_dim,
                                        elementwise_scale_and_shift=True)
            self.ln = nn.ModuleList([ln] * self.max_speculative_tokens)

        else:
            self.emb = nn.ModuleList([
                VocabParallelEmbedding(config.vocab_size,
                                       self.inner_dim,
                                       org_num_embeddings=config.vocab_size)
                for _ in range(self.max_speculative_tokens)
            ])

            self.proj = nn.ModuleList([
                nn.Linear((self.emb_dim if i == 0 else self.inner_dim),
                          self.inner_dim,
                          bias=False)
                for i in range(self.max_speculative_tokens)
            ])

            self.head = nn.ModuleList([
                ParallelLMHead(self.vocab_size, self.inner_dim, bias=False)
                for _ in range(self.max_speculative_tokens)
            ])
            self.ln = nn.ModuleList([
                MLPSpeculatorLayerNorm(self.inner_dim,
                                       elementwise_scale_and_shift=True)
                for _ in range(self.max_speculative_tokens)
            ])
        if self.scale_input:
            self.ln0 = MLPSpeculatorLayerNorm(
                self.emb_dim, elementwise_scale_and_shift=False)

        self.state_weight = 0.5**(0.5 / config.n_predict)
        self.emb_weight = math.sqrt(
            (1 - self.state_weight**2) * (self.inner_dim / 2))
        self.activation = nn.GELU()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                config.vocab_size, 1.0)
        self.sampler = get_sampler()

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

        if self.scale_input:
            previous_hidden_states = self.ln0(previous_hidden_states) / SQRT2

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
            previous_hidden_states = states
            # TODO: not yet supporting top_k_tokens_per_head
            states = states.flatten(0, 1)

            logits = self.logits_processor(self.head[head_index], states,
                                           sampling_metadata)

            output = self.sampler(logits, sampling_metadata)
            last_tokens = output.sampled_token_ids
            next_tokens.append(output)

        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            param = params_dict.get(name.replace("speculator.", ""))
            if param is not None:
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params
