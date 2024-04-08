from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch

@dataclass
class MambaCacheParams:
    is_prompt: bool = False
    conv_state: torch.Tensor = torch.Tensor()
    ssm_state: torch.Tensor = torch.Tensor()


@dataclass
class RequestInfo:
    request_id: str = ''
    n: int = 1


