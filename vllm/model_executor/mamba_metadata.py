from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch

@dataclass
class MambaCacheParams:
    seqlen_offset: int = 0
    conv_state: torch.Tensor = torch.Tensor()
    ssm_state: torch.Tensor = torch.Tensor()


@dataclass
class RequestInfo:
    request_id: str = ''
    n: int = 1


class MambaCache:
    def __init__(
        self,
        request_info: RequestInfo,
        layer_idx2mamba_cache: Optional[Dict[int, MambaCacheParams]] = None
    ) -> None:
        self.request_info = request_info
        if layer_idx2mamba_cache is None:
            self.layer_idx2mamba_cache = defaultdict(MambaCacheParams)
        else:
            self.layer_idx2mamba_cache = layer_idx2mamba_cache

