# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
#  https://github.com/modelscope/ms-swift/blob/v2.4.2/swift/utils/module_mapping.py

from dataclasses import dataclass, field


@dataclass
class MultiModelKeys:
    language_model: list[str] = field(default_factory=list)
    connector: list[str] = field(default_factory=list)
    # vision tower and audio tower
    tower_model: list[str] = field(default_factory=list)
    generator: list[str] = field(default_factory=list)

    @staticmethod
    def from_string_field(
        language_model: str | list[str] = None,
        connector: str | list[str] = None,
        tower_model: str | list[str] = None,
        generator: str | list[str] = None,
        **kwargs,
    ) -> "MultiModelKeys":
        def to_list(value):
            if value is None:
                return []
            return [value] if isinstance(value, str) else list(value)

        return MultiModelKeys(
            language_model=to_list(language_model),
            connector=to_list(connector),
            tower_model=to_list(tower_model),
            generator=to_list(generator),
            **kwargs,
        )
