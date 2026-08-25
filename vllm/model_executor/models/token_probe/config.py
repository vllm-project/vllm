# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

SING_PROBE_IDENTITY_MODEL_TYPE = "sing_probe_identity"
SING_PROBE_MLP_MODEL_TYPE = "sing_probe_mlp"
SING_PROBE_ATTN_MODEL_TYPE = "sing_probe_attn"


def resolve_labels(config: dict) -> tuple[str, ...]:
    labels = config.get("labels")
    if labels:
        return tuple(str(label) for label in labels)

    count = config.get("num_labels")
    if count is None:
        raise ValueError("token probe config must specify 'labels' or 'num_labels'")
    count = int(count)
    if count <= 0:
        raise ValueError(f"token probe num_labels must be positive, got {count}")
    return tuple(f"label_{i}" for i in range(count))


@dataclass(frozen=True)
class ProbeConfig:
    model_type: str
    hidden_size: int | None
    base_model_layer_ids: tuple[int, ...]
    labels: tuple[str, ...]
    intermediate_size: int
    hidden_act: str
    num_attention_heads: int
    head_dim: int
    sliding_window: int | None

    @property
    def input_size(self) -> int | None:
        if self.hidden_size is None:
            return None
        return self.hidden_size * len(self.base_model_layer_ids)

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @classmethod
    def from_dict(cls, config: dict) -> "ProbeConfig":
        model_type = config.get("model_type")
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("token probe config must specify 'model_type'")
        model_type = model_type.lower()
        labels = (
            ()
            if model_type == SING_PROBE_IDENTITY_MODEL_TYPE
            else resolve_labels(config)
        )
        layer_ids = config.get("base_model_layer_ids") or ()
        hidden_size = config.get("hidden_size")
        return cls(
            model_type=model_type,
            hidden_size=None if hidden_size is None else int(hidden_size),
            base_model_layer_ids=tuple(int(layer_id) for layer_id in layer_ids),
            labels=labels,
            intermediate_size=int(config.get("intermediate_size", 1024)),
            hidden_act=str(config.get("hidden_act", "gelu")),
            num_attention_heads=int(config.get("num_attention_heads", 4)),
            head_dim=int(config.get("head_dim", 64)),
            sliding_window=config.get("sliding_window"),
        )
