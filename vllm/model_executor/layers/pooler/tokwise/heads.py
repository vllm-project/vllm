# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.model_executor.layers.pooler import ActivationFn, ClassifierFn, ProjectorFn
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .methods import TokenPoolingMethodOutputItem

TokenPoolerHeadOutputItem: TypeAlias = torch.Tensor | None


class TokenPoolerHead(nn.Module, ABC):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        raise NotImplementedError

    @abstractmethod
    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        raise NotImplementedError

    def forward(
        self,
        pooled_data: list[TokenPoolingMethodOutputItem],
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolerHeadOutputItem]:
        pooling_params = pooling_metadata.pooling_params
        if len(pooled_data) != len(pooling_params):
            raise ValueError(
                f"pooled_data length ({len(pooled_data)}) does not match "
                f"pooling_params length ({len(pooling_params)})"
            )

        return [self.forward_chunk(d, p) for d, p in zip(pooled_data, pooling_params)]


class TokenEmbeddingPoolerHead(TokenPoolerHead):
    def __init__(
        self,
        head_dtype: torch.dtype | str | None = None,
        projector: ProjectorFn | None = None,
        activation: ActivationFn | None = None,
    ) -> None:
        super().__init__()

        self.head_dtype = head_dtype
        self.projector = projector
        self.activation = activation

    def extra_repr(self) -> str:
        attrs = []
        if self.head_dtype is not None:
            attrs.append(f"head_dtype={self.head_dtype}")
        if self.projector is not None:
            attrs.append("projector=True")
        if self.activation is not None:
            attrs.append(f"activation={self.activation.__class__.__name__}")
        return ", ".join(attrs)

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed"}

    def project_batch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project entire batch tensor at once for zero-copy scoring.

        Applies projector and activation (same pipeline as forward_chunk
        but without per-request matryoshka truncation).
        Returns [total_tokens, embed_dim].

        The dtype cast is deferred until after projection so that the
        large [N, hidden_dim] tensor stays in its native dtype.  Only
        the smaller [N, embed_dim] output is cast — avoids a temporary
        allocation that can be 4-8x larger than the final result.
        """
        # Project in the input's native dtype (fp16) to avoid casting
        # the large [N, hidden_dim] tensor to fp32.  We temporarily cast
        # the small weight [embed, hidden] instead.  Only the small
        # [N, embed_dim] output is cast to head_dtype afterwards.
        if self.projector is not None:
            if (
                isinstance(self.projector, nn.Linear)
                and self.projector.weight.dtype != hidden_states.dtype
            ):
                import torch.nn.functional as F

                w = self.projector.weight.to(hidden_states.dtype)
                b = (
                    self.projector.bias.to(hidden_states.dtype)
                    if self.projector.bias is not None
                    else None
                )
                hidden_states = F.linear(hidden_states, w, b)
            else:
                # nn.Sequential or other non-Linear projector: we can't
                # downcast the projector weights, so cast the input up
                # to head_dtype to match the projector's parameter
                # dtype.  Mirrors forward_chunk's behavior.
                if (
                    self.head_dtype is not None
                    and hidden_states.dtype != self.head_dtype
                ):
                    hidden_states = hidden_states.to(self.head_dtype)
                hidden_states = self.projector(hidden_states)
        # Cast the small [N, embed_dim] result, not the big [N, hidden].
        if self.head_dtype is not None:
            hidden_states = hidden_states.to(self.head_dtype)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        return hidden_states

    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        # for unfinished chunked prefill
        if pooled_data is None:
            return None

        if self.head_dtype is not None:
            pooled_data = pooled_data.to(self.head_dtype)
        # pooled_data shape: [n_tokens, hidden_size]

        # Apply ST projector
        if self.projector is not None:
            embeddings = self.projector(pooled_data)
        else:
            embeddings = pooled_data
        # embeddings shape: [n_tokens, embedding_size]

        # for matryoshka representation
        if pooling_param.dimensions is not None:
            embeddings = embeddings[..., : pooling_param.dimensions]

        # for normalize
        if self.activation is not None and pooling_param.use_activation:
            embeddings = self.activation(embeddings)

        # embeddings shape: [n_tokens, embedding_size]
        return embeddings


class TokenClassifierPoolerHead(TokenPoolerHead):
    def __init__(
        self,
        classifier: ClassifierFn | None = None,
        logit_mean: float | None = None,
        logit_sigma: float | None = None,
        head_dtype: torch.dtype | str | None = None,
        activation: ActivationFn | None = None,
    ) -> None:
        super().__init__()

        self.classifier = classifier
        self.logit_mean = logit_mean
        self.logit_sigma = logit_sigma
        self.head_dtype = head_dtype
        self.activation = activation

    def extra_repr(self) -> str:
        attrs = []
        if self.head_dtype is not None:
            attrs.append(f"head_dtype={self.head_dtype}")
        if self.classifier is not None:
            attrs.append("classifier=True")
        if self.logit_mean is not None:
            attrs.append(f"logit_mean={self.logit_mean}")
        if self.logit_sigma is not None:
            attrs.append(f"logit_sigma={self.logit_sigma}")
        if self.activation is not None:
            attrs.append(f"activation={self.activation.__class__.__name__}")
        return ", ".join(attrs)

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_classify"}

    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        # for unfinished chunked prefill
        if pooled_data is None:
            return None

        if self.head_dtype is not None:
            pooled_data = pooled_data.to(self.head_dtype)
        # hidden_states shape: [n_token, hidden_size]

        if self.classifier is not None:
            logits = self.classifier(pooled_data)
        else:
            logits = pooled_data
        # logits shape: [n_token, num_labels]

        # Affine score calibration: activation((logit - mean) / sigma)
        if self.logit_mean is not None:
            logits = logits - self.logit_mean
        if self.logit_sigma is not None:
            logits = logits / self.logit_sigma

        if self.activation is not None and pooling_param.use_activation:
            logits = self.activation(logits)

        # logits shape: [n_token, num_labels]
        return logits
