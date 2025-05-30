# SPDX-License-Identifier: Apache-2.0
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

    normalize: Optional[bool] = None
    """
    Whether to normalize the pooled outputs. Usually, this should be set to
    ``True`` for embedding outputs.
    """

    softmax: Optional[bool] = None
    """
    Whether to apply softmax to the pooled outputs. Usually, this should be set
    to ``True`` for classification outputs.
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
