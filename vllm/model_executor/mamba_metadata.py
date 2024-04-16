from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class MambaCacheParams:
    is_prompt: bool = False
    conv_state: torch.Tensor = torch.Tensor()
    ssm_state: torch.Tensor = torch.Tensor()


@dataclass
class RequestInfo:
    request_id: str = ''
    seqs_id: List[int] = field(default_factory=list)
