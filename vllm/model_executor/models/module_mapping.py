# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
#  https://github.com/modelscope/ms-swift/blob/v2.4.2/swift/utils/module_mapping.py

from dataclasses import dataclass, field
from typing import Union


@dataclass
class ModelKeys:
    model_type: str = None

    module_list: str = None

    embedding: str = None

    mlp: str = None

    down_proj: str = None

    attention: str = None

    o_proj: str = None

    q_proj: str = None

    k_proj: str = None

    v_proj: str = None

    qkv_proj: str = None

    qk_proj: str = None

    qa_proj: str = None

    qb_proj: str = None

    kva_proj: str = None

    kvb_proj: str = None

    output: str = None


@dataclass
class MultiModelKeys(ModelKeys):
    language_model: list[str] = field(default_factory=list)
    connector: list[str] = field(default_factory=list)
    # vision tower and audio tower
    tower_model: list[str] = field(default_factory=list)
    generator: list[str] = field(default_factory=list)

    @staticmethod
    def from_string_field(language_model: Union[str, list[str]] = None,
                          connector: Union[str, list[str]] = None,
                          tower_model: Union[str, list[str]] = None,
                          generator: Union[str, list[str]] = None,
                          **kwargs) -> 'MultiModelKeys':

        def to_list(value):
            if value is None:
                return []
            return [value] if isinstance(value, str) else list(value)

        return MultiModelKeys(language_model=to_list(language_model),
                              connector=to_list(connector),
                              tower_model=to_list(tower_model),
                              generator=to_list(generator),
                              **kwargs)
