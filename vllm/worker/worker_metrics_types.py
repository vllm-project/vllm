# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List


@dataclass
class VllmWorkerStats:
    summary_total_prefill_token: List[int]
