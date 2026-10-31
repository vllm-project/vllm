# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable, Set
from typing import TypeAlias

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.model_executor.layers.pooler import ActivationFn, ClassifierFn, ProjectorFn
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .methods import TokenPoolingMethodOutputItem

TokenPoolerHeadOutputItem: TypeAlias = torch.Tensor | None


def _mean_pool_chunks(data: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Mean-pool complete contextual states, accumulating in FP32."""
    chunk_size = min(chunk_size, len(data))
    num_full, remainder = divmod(len(data), chunk_size)
    full_end = num_full * chunk_size
    full_chunks = data[:full_end].reshape(num_full, chunk_size, data.shape[-1])
    if not remainder:
        return full_chunks.mean(dim=1, dtype=torch.float32)

    return torch.cat(
        [
            full_chunks.mean(dim=1, dtype=torch.float32),
            data[full_end:].mean(dim=0, keepdim=True, dtype=torch.float32),
        ]
    )


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

    def forward_batched(
        self,
        hidden_states: torch.Tensor,
        pooling_params: list[PoolingParams],
        split: Callable[[torch.Tensor], list[TokenPoolingMethodOutputItem]],
    ) -> list[TokenPoolerHeadOutputItem] | None:
        """Batch projection, splitting only when finishing needs per-request options.

        Keep eligibility checks in sync with PoolingParams fields read by forward_chunk.
        """
        if envs.VLLM_BATCH_INVARIANT or any(
            p.late_chunking_params is not None for p in pooling_params
        ):
            return None
        # [total_tokens, hidden_size] -> [total_tokens, embedding_size]
        embeddings = self._project(hidden_states)
        first = pooling_params[0]
        if all(
            p.dimensions == first.dimensions
            and p.use_activation == first.use_activation
            for p in pooling_params
        ):
            embeddings = self._finish(embeddings, first)
            # Each output has shape [request_tokens, output_size].
            return split(embeddings)
        # Each part has shape [request_tokens, embedding_size].
        pooled_data = split(embeddings)
        del embeddings
        return [
            self._finish(part, param) if part is not None else None
            for part, param in zip(pooled_data, pooling_params, strict=True)
        ]

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project [num_rows, hidden_size] to [num_rows, embedding_size]."""
        if self.head_dtype is not None:
            hidden_states = hidden_states.to(self.head_dtype)
        if self.projector is not None:
            return self.projector(hidden_states)
        return hidden_states

    def _finish(
        self, embeddings: torch.Tensor, pooling_param: PoolingParams
    ) -> torch.Tensor:
        """Slice/normalize [num_rows, embedding_size] to [num_rows, output_size]."""
        if pooling_param.dimensions is not None:
            embeddings = embeddings[..., : pooling_param.dimensions]
        if self.activation is not None and pooling_param.use_activation:
            embeddings = self.activation(embeddings)
        return embeddings

    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        # for unfinished chunked prefill
        if pooled_data is None:
            return None

        # pooled_data: [num_tokens, hidden_size]
        if (late_chunking := pooling_param.late_chunking_params) is not None:
            means = _mean_pool_chunks(pooled_data, late_chunking.chunk_size)
            pooled_data = (
                means.to(pooled_data.dtype) if self.head_dtype is None else means
            )
            # pooled_data: [num_chunks, hidden_size]

        # Output: [num_rows, output_size], one row per token or late chunk.
        return self._finish(self._project(pooled_data), pooling_param)


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
