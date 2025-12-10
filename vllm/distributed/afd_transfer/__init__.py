# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AFD (Attention-FFN Disaggregation) transfer components.

This module provides the distributed infrastructure for AFD, enabling
disaggregated FFN computation across different machines while keeping
attention computation local.
"""

from .afd_connector import AFDConnectorBase, AFDConnectorFactory, AFDConnectorMetadata

__all__ = ["AFDConnectorBase", "AFDConnectorMetadata", "AFDConnectorFactory"]
