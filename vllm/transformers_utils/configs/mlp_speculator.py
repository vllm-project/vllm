# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from transformers import PretrainedConfig


class MLPSpeculatorConfig(PretrainedConfig):
    model_type = "mlp_speculator"

    attribute_map = {
        "hidden_size": "emb_dim",
    }

    def __init__(self,
                 vocab_size: int = 32000,
                 emb_dim: int = 4096,
                 inner_dim: int = 0,
                 n_predict: int = 3,
                 top_k_tokens_per_head: Optional[List[int]] = None,
                 n_candidates: int = 5,
                 tie_weights: bool = False,
                 scale_input: bool = False,
                 **kwargs):
        """
        Initialize an MLPSpeculatorConfig

        Args:
            vocab_size: int
                the model vocab size
            emb_dim: int
                the model embedding dimension
            inner_dim: int
                the inner dimension of the model. If 0, will be the emb_dim.
            n_predict: int
                the number of lookaheads for the speculator
            top_k_tokens_per_head: List[int]
                Number of tokens to consider from each head when forming the
                candidate tree.
                For each candidate branch in the tree, head n produces topk[n]
                additional sub-branches.
                NOTE: This parameter is currently unused.
            n_candidates: int
                number of child candidates to create per sequence
            tie_weights: bool
                If true, use a single set of weights for every model
                head/stage after the first. The initial projection
                from the base model may have a different size, so that
                stays separate.
            scale_input: bool
                if True, will scale the initial hidden states from
                the base model.
        """
        if top_k_tokens_per_head is None:
            top_k_tokens_per_head = [5, 4, 3]
        assert len(top_k_tokens_per_head) == n_predict
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.inner_dim = inner_dim
        self.n_predict = n_predict
        self.top_k_tokens_per_head = top_k_tokens_per_head
        self.n_candidates = n_candidates
        self.num_lookahead_tokens = n_predict
        self.tie_weights = tie_weights
        self.scale_input = scale_input

        super().__init__(**kwargs)
