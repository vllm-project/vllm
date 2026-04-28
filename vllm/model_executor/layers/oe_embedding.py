# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class OEEmbedding(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.base_vocab_size = model_config.get_vocab_size()
        self.oe_vocab_size = model_config.get_oe_vocab_size()

        self.n_embed_per_ngram = model_config.get_n_embed_per_ngram()
        self.n_head_per_ngram = model_config.get_n_head_per_ngram()
        self.max_ngram_size = model_config.get_max_ngram_size()

        self.oe_total_heads = self.n_head_per_ngram * (self.max_ngram_size - 1)

        self.vocab_size_for_head: list[int] = []
        self.vocab_mods: list[torch.Tensor] = []

        # init oe embedding
        self.hidden_size = model_config.get_hidden_size()
        # compute vocab size and vocab mods
        self._initialize_vocab_sizes_for_oe()
        self.cusum_vocab_size_for_heads = np.concatenate(
            [[0], np.cumsum(self.vocab_size_for_head)]
        )

        self._init_ngram_embeddings()

        # scale-related parameters
        self.oe_base_scale = model_config.get_oe_base_scale()
        self.oe_output_scale = model_config.get_oe_output_scale()

    def _initialize_vocab_sizes_for_oe(self):
        """Get vocab sizes for each OE head as prime numbers.

        For each n-gram type(2-gram, 3-gram, ..., max_ngram_size-gram), we
        have `n_head_per_ngram` heads.

        Each head gets a unique prime vocab size starting from oe_vocab_size
        """

        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            return all(n % i != 0 for i in range(3, int(n**0.5) + 1, 2))

        current = max(2, self.oe_vocab_size)
        while True:
            if is_prime(current):
                self.vocab_size_for_head.append(current)
                if len(self.vocab_size_for_head) == self.oe_total_heads:
                    return
            current += 1

    def _init_ngram_embeddings(self) -> None:
        self.oe_embeder = VocabParallelEmbedding(
            sum(self.vocab_size_for_head),
            self.n_embed_per_ngram,
        )
        self.oe_embeder.weight.weight_loader = self.weight_loader
        self.proj = ReplicatedLinear(
            self.n_embed_per_ngram * self.oe_total_heads,
            self.hidden_size,
            bias=False,
        )

    def get_oe_total_heads(self) -> int:
        return self.oe_total_heads

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int
    ):
        # NOTE(yxing): need for tp mode
        assert shard_id < self.oe_total_heads, (
            f"expected {self.oe_total_heads} embedders, now the shard id is {shard_id}"
        )

        # Global range of this head in the concatenated embedding table
        global_start = int(self.cusum_vocab_size_for_heads[shard_id])
        global_end = int(self.cusum_vocab_size_for_heads[shard_id + 1])

        # This TP rank's range in the concatenated embedding table
        tp_start = self.oe_embeder.shard_indices.org_vocab_start_index
        tp_end = self.oe_embeder.shard_indices.org_vocab_end_index

        # Intersect the head's global range with this rank's TP shard range
        intersect_start = max(global_start, tp_start)
        intersect_end = min(global_end, tp_end)

        if intersect_start >= intersect_end:
            return  # This head has no rows on this TP rank

        # Offsets within loaded_weight (source)
        src_start = intersect_start - global_start
        src_end = intersect_end - global_start

        # Offsets within param (local TP shard, destination)
        dst_start = intersect_start - tp_start
        dst_end = intersect_end - tp_start

        param.data[dst_start:dst_end].copy_(loaded_weight[src_start:src_end])

    def forward(
        self,
        base_embedding: torch.Tensor,
        oe_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        The input_ids is flatten. The input_ids include prefill and decode tokens

        The shape of oe_input_ids is: [tokens * total_oe_heads]
        """
        oe_embed_tokens = self.oe_embeder(
            oe_input_ids
        )  # shape: [total_oe_heads * tokens, oe_embed_dim]
        oe_embed_tokens = oe_embed_tokens.view(
            self.oe_total_heads, -1, self.n_embed_per_ngram
        )
        oe_embed_tokens = oe_embed_tokens.transpose(0, 1)
        oe_embed_tokens = oe_embed_tokens.contiguous().view(
            -1, self.oe_total_heads * self.n_embed_per_ngram
        )  # [tokens, oe_total_heads * n_embed_per_ngram]
        oe_proj, _ = self.proj(oe_embed_tokens)  # shape: [tokens, hidden_size]
        return (
            base_embedding * self.oe_base_scale + oe_proj * self.oe_output_scale
        ) / (
            (
                self.oe_base_scale * self.oe_base_scale
                + self.oe_output_scale * self.oe_output_scale
            )
            ** 0.5
        )
