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

    # Rows processed per chunk in project_batch when a head_dtype upcast is
    # needed. Bounds the [chunk, hidden_dim] head_dtype transient (e.g.
    # 16384 x 2048 fp32 = 128 MiB) instead of materialising the full
    # [N, hidden_dim] batch at head_dtype — see PR #40337 review.
    _PROJECT_BATCH_CHUNK = 16384

    def _project_rows(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Upcast + project + activate, exactly like forward_chunk."""
        if self.head_dtype is not None and hidden_states.dtype != self.head_dtype:
            hidden_states = hidden_states.to(self.head_dtype)
        if self.projector is not None:
            hidden_states = self.projector(hidden_states)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        return hidden_states

    def project_batch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project entire batch tensor for zero-copy scoring.

        Applies upcast, projector and activation — the same pipeline as
        forward_chunk (without per-request matryoshka truncation), so
        queries and documents are projected at identical precision.
        Returns [total_tokens, embed_dim].

        The head_dtype upcast is applied in row chunks: the projector and
        activation act row-wise, so chunking preserves the upcast-then-
        project semantics (any deviation is BLAS kernel selection vs row
        count, <= ~1 ulp of fp32) while the head_dtype transient stays
        bounded at [_PROJECT_BATCH_CHUNK, hidden_dim] instead of the full
        [N, hidden_dim] batch (which peaked at GiB scale for ColPali-sized
        batches with an fp32 head).
        """
        n = hidden_states.shape[0]
        needs_cast = (
            self.head_dtype is not None and hidden_states.dtype != self.head_dtype
        )
        if not needs_cast or n <= self._PROJECT_BATCH_CHUNK:
            return self._project_rows(hidden_states)

        out: torch.Tensor | None = None
        for start in range(0, n, self._PROJECT_BATCH_CHUNK):
            chunk = self._project_rows(
                hidden_states[start : start + self._PROJECT_BATCH_CHUNK]
            )
            if out is None:
                out = torch.empty(
                    (n, *chunk.shape[1:]), dtype=chunk.dtype, device=chunk.device
                )
            out[start : start + chunk.shape[0]] = chunk
        assert out is not None
        return out

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
