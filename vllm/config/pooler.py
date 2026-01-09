# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

logger = init_logger(__name__)

PoolingTypeStr = Literal["LAST", "ALL", "CLS", "STEP", "MEAN"]


@config
@dataclass
class PoolerConfig:
    """Controls the behavior of output pooling in pooling models."""

    pooling_type: PoolingTypeStr | None = None
    """
    The pooling method of the pooling model.
    """

    ## for embeddings models
    normalize: bool | None = None
    """
    Whether to normalize the embeddings outputs. Defaults to True.
    """
    dimensions: int | None = None
    """
    Reduce the dimensions of embeddings if model
    support matryoshka representation. Defaults to None.
    """
    enable_chunked_processing: bool | None = None
    """
    Whether to enable chunked processing for long inputs that exceed the model's
    maximum position embeddings. When enabled, long inputs will be split into
    chunks, processed separately, and then aggregated using weighted averaging.
    This allows embedding models to handle arbitrarily long text without CUDA
    errors. Defaults to False.
    """
    max_embed_len: int | None = None
    """
    Maximum input length allowed for embedding generation. When set, allows
    inputs longer than max_embed_len to be accepted for embedding models.
    When an input exceeds max_embed_len, it will be handled according to 
    the original max_model_len validation logic. 
    Defaults to None (i.e. set to max_model_len).
    """

    ## for classification models
    softmax: float | None = None
    """
    softmax will be deprecated, please use use_activation instead.
    """
    activation: float | None = None
    """
    activation will be deprecated, please use use_activation instead.
    """
    use_activation: bool | None = None
    """
    Whether to apply activation function to the classification outputs.
    Defaults to True.
    """
    logit_bias: float | None = None
    """
    If provided, apply classification logit biases. Defaults to None.
    """

    ## for reward models
    step_tag_id: int | None = None
    """
    If set, only the score corresponding to the `step_tag_id` in the
    generated sentence should be returned. Otherwise, the scores for all tokens
    are returned.
    """
    returned_token_ids: list[int] | None = None
    """
    A list of indices for the vocabulary dimensions to be extracted,
    such as the token IDs of `good_token` and `bad_token` in the
    `math-shepherd-mistral-7b-prm` model.
    """

    def __post_init__(self):
        # raise deprecated warning for softmax and activation
        self.use_activation = get_use_activation(self)

    def get_pooling_type(self) -> PoolingTypeStr:
        assert self.pooling_type is not None, "Should be resolved by ModelConfig"
        return self.pooling_type

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
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str


def get_use_activation(o: object):
    if softmax := getattr(o, "softmax", None) is not None:
        logger.warning_once(
            "softmax will be deprecated and will be removed in v0.15. "
            "Please use use_activation instead."
        )
        return softmax

    if activation := getattr(o, "activation", None) is not None:
        logger.warning_once(
            "activation will be deprecated and will be removed in v0.15. "
            "Please use use_activation instead."
        )
        return activation

    return getattr(o, "use_activation", None)
