from dataclasses import dataclass, fields
from typing import List, Optional, Union

from vllm.logger import init_logger

logger = init_logger(__name__)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str
    served_model_name: Optional[Union[List[str]]] = None
    tokenizer: Optional[str] = None
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    seed: int = 0

    def to_dict(self):
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))
