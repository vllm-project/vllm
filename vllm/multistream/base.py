from dataclasses import dataclass
from enum import Enum

class MSEventKey(Enum):
    ATTN_COM_FINISH = 0
    ATTN_AR_FINISH = 1
    FFN_COM_FINISH = 2
    FFN_AR_FINISH = 3


@dataclass
class MSAttentionMetadataSplitConfig:
    """
    micro batch split config for split attention metadata
    """
    # micro batch num
    num_micro_batches: int = 2
    # split micro batches only when total tokens >= min_total_tokens_to_split
    min_total_tokens_to_split: int = 256,
    # split micro batches only when prefill tokens >= min_prefill_tokens_to_split
    min_prefill_tokens_to_split: int = 64,
    # token imbalance_ratio between micro batches
    imbalance_ratio: float = 0.1,
    # enable to split one query into two micro batches
    enable_request_split: bool = True,