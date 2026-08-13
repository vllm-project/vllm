# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class EplbMetricsSnapshot:
    rebalancing: bool = False
    rebalance_events: int = 0
