import time
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Dict

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

@dataclass
class EvictionStats:
    total_evictions: int = 0
    total_fetches: int = 0
    eviction_latency_ms: List[float] = field(default_factory=list)
    fetch_latency_ms: List[float] = field(default_factory=list)
    
    def record_eviction(self, block_id: int, latency_ms: float = 0.0):
        self.total_evictions += 1
        self.eviction_latency_ms.append(latency_ms)
        
    def record_fetch(self, block_id: int, latency_ms: float = 0.0):
        self.total_fetches += 1
        self.fetch_latency_ms.append(latency_ms)
    
    def summary(self) -> Dict:
        avg_evict = 0.0
        avg_fetch = 0.0
        if np is not None:
            avg_evict = float(np.mean(self.eviction_latency_ms)) if self.eviction_latency_ms else 0.0
            avg_fetch = float(np.mean(self.fetch_latency_ms)) if self.fetch_latency_ms else 0.0
        else:
            avg_evict = sum(self.eviction_latency_ms) / len(self.eviction_latency_ms) if self.eviction_latency_ms else 0.0
            avg_fetch = sum(self.fetch_latency_ms) / len(self.fetch_latency_ms) if self.fetch_latency_ms else 0.0
            
        return {
            "evictions": self.total_evictions,
            "fetches": self.total_fetches,
            "avg_eviction_latency_ms": avg_evict,
            "avg_fetch_latency_ms": avg_fetch,
            "eviction_fetch_ratio": self.total_evictions / max(1, self.total_fetches)
        }

@dataclass
class AccessTrace:
    timestamp: float
    request_id: str
    block_id: int
    operation: Literal["allocate", "access", "evict", "fetch"]
    location: Literal["gpu", "cpu", "in_transit"]
    attention_score: Optional[float] = None

class AccessTracer:
    def __init__(self, output_path: str):
        self.traces: List[AccessTrace] = []
        self.output_path = output_path
        
    def record(self, request_id: str, block_id: int, operation: Literal["allocate", "access", "evict", "fetch"], location: Literal["gpu", "cpu", "in_transit"], attention_score: Optional[float] = None):
        trace = AccessTrace(
            timestamp=time.time(),
            request_id=request_id,
            block_id=block_id,
            operation=operation,
            location=location,
            attention_score=attention_score
        )
        self.traces.append(trace)
        
    def save(self):
        if pd is not None:
            df = pd.DataFrame([asdict(t) for t in self.traces])
            df.to_csv(self.output_path, index=False)
        else:
            # Fallback to simple CSV export if pandas is not available
            import csv
            if not self.traces: return
            keys = self.traces[0].__dict__.keys()
            with open(self.output_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows([asdict(t) for t in self.traces])
