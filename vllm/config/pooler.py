# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal, get_args

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.tasks import PoolingTask
from vllm.utils.hashing import safe_hash

logger = init_logger(__name__)

SequencePoolingType = Literal["CLS", "LAST", "MEAN"]
SEQ_POOLING_TYPES: tuple[SequencePoolingType, ...] = get_args(SequencePoolingType)

TokenPoolingType = Literal["ALL", "STEP"]
TOK_POOLING_TYPES: tuple[TokenPoolingType, ...] = get_args(TokenPoolingType)


@config
class PoolerConfig:
    """Controls the behavior of output pooling in pooling models."""

    task: PoolingTask | None = None
    """
    The task used for pooling.
    """

    pooling_type: SequencePoolingType | TokenPoolingType | None = None
    """
    The pooling method used for pooling.

    If set, `seq_pooling_type` or `tok_pooling_type` are automatically populated
    with this field. Alternatively, users can set `seq_pooling_type` and
    `tok_pooling_type` explicitly.

    This field is mainly for user convenience. Internal code should always use
    `seq_pooling_type` or `tok_pooling_type` instead of `pooling_type`.
    """

    seq_pooling_type: SequencePoolingType | None = None
    """
    The pooling method used for sequence pooling.
    """

    tok_pooling_type: TokenPoolingType | None = None
    """
    The pooling method used for tokenwise pooling.
    """

    use_activation: bool | None = None
    """
    Whether to apply activation function to the pooler outputs.
    `None` uses the pooler's default, which is `True` in most cases.
    """

    ## for embedding models
    dimensions: int | None = None
    """
    Reduce the dimensions of embeddings if model
    support matryoshka representation. Defaults to None.
    """
    enable_chunked_processing: bool = False
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

    ## for classification models — affine score calibration
    logit_mean: float | None = None
    """
    If provided, subtract this value from classification logits before
    activation. Used for affine score calibration (Platt scaling):
    activation((logit - logit_mean) / logit_sigma). Defaults to None.
    """

    logit_sigma: float | None = None
    """
    If provided, divide the classification logits by this value after
    mean subtraction. Used for affine score calibration (Platt scaling):
    activation((logit - logit_mean) / logit_sigma). Defaults to None.
    """

    # Deprecated aliases — will be removed in v0.21
    logit_bias: float | None = None
    """
    Deprecated: Use logit_mean instead. Will be removed in v0.21.
    """

    logit_scale: float | None = None
    """
    Deprecated: Use logit_sigma instead (note: logit_sigma = 1/logit_scale).
    Will be removed in v0.21.
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

    def __post_init__(self) -> None:
        # Handle deprecated logit_bias → logit_mean
        if self.logit_bias is not None:
            if self.logit_mean is not None:
                raise ValueError(
                    "Cannot set both `logit_bias` and `logit_mean`. "
                    "`logit_bias` is deprecated, use `logit_mean` instead."
                )
            logger.warning(
                "`logit_bias` is deprecated and will be removed in v0.21. "
                "Use `logit_mean` instead."
            )
            self.logit_mean = self.logit_bias
            self.logit_bias = None

        # Handle deprecated logit_scale → logit_sigma
        if self.logit_scale is not None:
            if self.logit_sigma is not None:
                raise ValueError(
                    "Cannot set both `logit_scale` and `logit_sigma`. "
                    "`logit_scale` is deprecated, use `logit_sigma` instead."
                )
            logger.warning(
                "`logit_scale` is deprecated and will be removed in v0.21. "
                "Use `logit_sigma` instead (logit_sigma = 1/logit_scale)."
            )
            if self.logit_scale == 0:
                raise ValueError("logit_scale cannot be 0 (division by zero)")
            self.logit_sigma = 1.0 / self.logit_scale
            self.logit_scale = None

        if self.logit_sigma is not None and self.logit_sigma == 0:
            raise ValueError("logit_sigma cannot be 0 (division by zero)")

        if pooling_type := self.pooling_type:
            if self.seq_pooling_type is not None:
                raise ValueError(
                    "Cannot set both `pooling_type` and `seq_pooling_type`"
                )
            if self.tok_pooling_type is not None:
                raise ValueError(
                    "Cannot set both `pooling_type` and `tok_pooling_type`"
                )

            if pooling_type in SEQ_POOLING_TYPES:
                logger.debug(
                    "Resolved `pooling_type=%r` to `seq_pooling_type=%r`.",
                    pooling_type,
                    pooling_type,
                )
                self.seq_pooling_type = pooling_type  # type: ignore[assignment]
            elif pooling_type in TOK_POOLING_TYPES:
                logger.debug(
                    "Resolved `pooling_type=%r` to `tok_pooling_type=%r`.",
                    pooling_type,
                    pooling_type,
                )
                self.tok_pooling_type = pooling_type  # type: ignore[assignment]
            else:
                raise NotImplementedError(pooling_type)

    def get_seq_pooling_type(self) -> SequencePoolingType:
        assert self.seq_pooling_type is not None, "Should be resolved by ModelConfig"
        return self.seq_pooling_type

    def get_tok_pooling_type(self) -> TokenPoolingType:
        assert self.tok_pooling_type is not None, "Should be resolved by ModelConfig"
        return self.tok_pooling_type

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
