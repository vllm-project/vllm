from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    request_rate: float = float("inf")
    ignore_request_timestamp: bool = False
