# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

LocalLoRASourceLayout = Literal[
    "row", "column", "replicated", "merged", "moe", "moe_3d"
]


@dataclass(frozen=True)
class LocalLoRAModulePlan:
    """Runtime packed order and source ownership for one local module."""

    module_name: str
    layer_type: str
    source_layout: LocalLoRASourceLayout
    source_names: tuple[tuple[str, ...], ...]
    factor_shapes: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    dtype: str
    tp_rank: int
    tp_size: int
    global_input_size: int
    global_output_sizes: tuple[int, ...]
    output_shard_ids: tuple[int, ...]
    expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class LocalLoRAPlan:
    """A value-independent binding to the receiver's actual wrapped modules."""

    rank: int
    lora_alpha: int
    target_modules: tuple[str, ...]
    modules: tuple[LocalLoRAModulePlan, ...]

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
