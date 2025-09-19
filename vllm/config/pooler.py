# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Any, Optional

from pydantic.dataclasses import dataclass

from vllm.config.utils import config


@config
@dataclass
class PoolerConfig:
    """Controls the behavior of output pooling in pooling models."""

    pooling_type: Optional[str] = None
    """
    The pooling method of the pooling model. This should be a key in
    [`vllm.model_executor.layers.pooler.PoolingType`][].
    """

    ## for embeddings models
    normalize: Optional[bool] = None
    """
    Whether to normalize the embeddings outputs. Defaults to True.
    """
    dimensions: Optional[int] = None
    """
    Reduce the dimensions of embeddings if model
    support matryoshka representation. Defaults to None.
    """
    enable_chunked_processing: Optional[bool] = None
    """
    Whether to enable chunked processing for long inputs that exceed the model's
    maximum position embeddings. When enabled, long inputs will be split into
    chunks, processed separately, and then aggregated using weighted averaging.
    This allows embedding models to handle arbitrarily long text without CUDA
    errors. Defaults to False.
    """
    max_embed_len: Optional[int] = None
    """
    Maximum input length allowed for embedding generation. When set, allows
    inputs longer than max_embed_len to be accepted for embedding models.
    When an input exceeds max_embed_len, it will be handled according to 
    the original max_model_len validation logic. 
    Defaults to None (i.e. set to max_model_len).
    """

    ## for classification models
    activation: Optional[bool] = None
    """
    Whether to apply activation function to the classification outputs.
    Defaults to True.
    """
    logit_bias: Optional[float] = None
    """
    If provided, apply classification logit biases. Defaults to None.
    """

    ## for reward models
    softmax: Optional[bool] = None
    """
    Whether to apply softmax to the reward outputs.
    Defaults to True.
    """
    step_tag_id: Optional[int] = None
    """
    If set, only the score corresponding to the ``step_tag_id`` in the
    generated sentence should be returned. Otherwise, the scores for all tokens
    are returned.
    """
    returned_token_ids: Optional[list[int]] = None
    """
    A list of indices for the vocabulary dimensions to be extracted,
    such as the token IDs of ``good_token`` and ``bad_token`` in the
    ``math-shepherd-mistral-7b-prm`` model.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str
