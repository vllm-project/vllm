# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .config import (
    SING_PROBE_ATTN_MODEL_TYPE,
    SING_PROBE_IDENTITY_MODEL_TYPE,
    SING_PROBE_MLP_MODEL_TYPE,
    ProbeConfig,
)
from .probe_kernels import (
    ACT_GELU,
    ACT_RELU,
    attention_tail,
    classify_tail,
)


class ProbeHead(nn.Module, ABC):
    def __init__(
        self,
        *,
        hidden_size: int | None,
        layer_ids: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_indices = layer_ids
        self.label_names = labels

    @abstractmethod
    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityProbeHead(ProbeHead):
    @classmethod
    def from_config(cls, config: ProbeConfig, dtype: torch.dtype) -> "ProbeHead":
        if not config.base_model_layer_ids:
            raise ValueError(
                "identity token probe config must list base_model_layer_ids"
            )
        return cls(
            hidden_size=config.hidden_size,
            layer_ids=config.base_model_layer_ids,
            labels=(),
        )

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        return features.float()


class SingProbeMlpModel(ProbeHead):
    def __init__(self, config: ProbeConfig, dtype: torch.dtype) -> None:
        if config.input_size is None:
            raise ValueError("MLP token probe requires hidden_size")
        super().__init__(
            hidden_size=config.hidden_size,
            layer_ids=config.base_model_layer_ids,
            labels=config.labels,
        )
        self.fc1 = nn.Linear(config.input_size, config.intermediate_size, dtype=dtype)
        self.fc2 = nn.Linear(config.intermediate_size, config.num_labels, dtype=dtype)
        if config.hidden_act not in ("gelu", "relu"):
            raise ValueError(
                f"unsupported token probe activation {config.hidden_act!r}"
            )
        self.activation = ACT_GELU if config.hidden_act == "gelu" else ACT_RELU

    @classmethod
    def from_config(cls, config: ProbeConfig, dtype: torch.dtype) -> "ProbeHead":
        return cls(config, dtype)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(features.to(self.fc1.weight.dtype))
        return classify_tail(
            hidden,
            self.fc2.weight,
            self.fc2.bias,
            self.activation,
        )


class SingProbeAttnModel(ProbeHead):
    def __init__(self, config: ProbeConfig, dtype: torch.dtype) -> None:
        if config.input_size is None:
            raise ValueError("attention token probe requires hidden_size")
        if config.sliding_window is not None and config.sliding_window <= 0:
            raise ValueError("token probe sliding_window must be positive")
        super().__init__(
            hidden_size=config.hidden_size,
            layer_ids=config.base_model_layer_ids,
            labels=config.labels,
        )
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        projection_dim = self.num_attention_heads * self.head_dim
        self.proj_qkv = nn.Linear(
            config.input_size,
            projection_dim + 2 * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.norm = nn.RMSNorm(projection_dim, eps=1e-6, dtype=dtype)
        self.o_proj = nn.Linear(projection_dim, projection_dim, bias=False, dtype=dtype)
        self.classifier = nn.Linear(projection_dim, config.num_labels, dtype=dtype)

    @classmethod
    def from_config(cls, config: ProbeConfig, dtype: torch.dtype) -> "ProbeHead":
        return cls(config, dtype)

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.head_dim

    def project(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qkv = self.proj_qkv(features.to(self.proj_qkv.weight.dtype))
        return qkv.split((self.q_dim, 2 * self.kv_dim), dim=-1)

    def classify(self, attention_output: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        projected = self.o_proj(attention_output.to(self.o_proj.weight.dtype))
        return attention_tail(
            projected,
            q,
            self.norm.weight,
            self.classifier.weight,
            self.classifier.bias,
            self.norm.eps,
        )

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("attention probe requires paged attention metadata")


PROBE_MODELS: dict[str, type[ProbeHead]] = {
    SING_PROBE_IDENTITY_MODEL_TYPE: IdentityProbeHead,
    SING_PROBE_MLP_MODEL_TYPE: SingProbeMlpModel,
    SING_PROBE_ATTN_MODEL_TYPE: SingProbeAttnModel,
}
