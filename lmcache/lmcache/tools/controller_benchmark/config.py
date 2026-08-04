"""Configuration for LMCache Controller ZMQ Benchmark"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ZMQBenchmarkConfig:
    """Configuration for ZMQ benchmark parameters"""

    controller_pull_url: str
    controller_reply_url: Optional[str]
    duration: int
    batch_size: int
    num_instances: int
    num_workers: int
    num_locations: int
    num_keys: int
    controller_heartbeat_url: Optional[str] = None
    num_hashes: int = 100
    operations: Dict[str, float] = field(default_factory=dict)
    heartbeat_interval: float = 1.0
    register_first: bool = True
    # Multi-process settings
    num_processes: int = 1
    process_id: int = 0

    def __post_init__(self):
        if not self.operations:
            # Default: 70% admit, 25% evict, 5% heartbeat
            self.operations = {
                "admit": 70.0,
                "evict": 25.0,
                "heartbeat": 5.0,
            }
        # Validate operation percentages sum to 100
        total_percentage = sum(self.operations.values())
        if abs(total_percentage - 100.0) > 0.01:
            raise ValueError(
                "Operation percentages must sum to 100, got: %s" % total_percentage
            )
