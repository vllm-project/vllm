# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig
    from .inc_scheme import INCScheme


def resolve_scheme(layer_config: "INCLayerConfig") -> "INCScheme":
    from .inc_mxfp4_scheme import INCMxfp4Scheme
    from .inc_wna16_scheme import INCWna16Scheme

    scheme_list: list[type[INCScheme]] = [
        INCWna16Scheme,
        INCMxfp4Scheme,
    ]

    for scheme_cls in scheme_list:
        if scheme_cls.can_handle(layer_config):
            return scheme_cls()

    raise NotImplementedError(f"No INC scheme found for layer config: {layer_config}")
