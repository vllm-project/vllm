from typing import Optional, List, Union
import msgspec

from dataclasses import dataclass

@dataclass
class EngineCoreOutput:

    request_id: str
    new_token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None

class EngineCoreOutputs(msgspec.Struct):

    # [num_reqs]
    outputs: List[EngineCoreOutput]