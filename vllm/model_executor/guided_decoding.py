from enum import Enum
import time
from typing import List, Union
try:
    from outlines.serve.vllm import JSONLogitsProcessor, RegexLogitsProcessor
except ImportError:
    raise ValueError("Please install 'outlines' (pip install outlines) to use guided generation.")
import torch

from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import LogitsProcessor

class GuidedDecodingEngine(Enum):
    OUTLINES = "outlines"

class GuidedDecodingMode(Enum):
    REGEX = "regex"
    JSON_SCHEMA = "schema"

class OutlinesJSONLogitsProcessor(JSONLogitsProcessor):

    def __init__(self, json_schema: dict, llm: LLM):
        super().__init__(json_schema, llm)

    def __call__(
        self,
        input_ids: List[int],
        scores: torch.Tensor,
        seq_id: int,
    ) -> torch.Tensor:
        return super().__call__(seq_id, input_ids, scores)


class OulinesRegexLogitsProcessor(RegexLogitsProcessor):

    def __init__(self, regex: str, llm: LLM):
        super().__init__(regex, llm)

    def __call__(
        self,
        input_ids: List[int],
        scores: torch.Tensor,
        seq_id: int,
    ) -> torch.Tensor:
        return super().__call__(seq_id, input_ids, scores)

    
def get_logits_processor(specification: Union[str, dict], mode: GuidedDecodingMode, engine: GuidedDecodingEngine, llm_engine: LLMEngine):
    if engine == GuidedDecodingEngine.OUTLINES:
        if mode == GuidedDecodingMode.JSON_SCHEMA:
            return OutlinesJSONLogitsProcessor(specification, llm_engine)
        elif mode == GuidedDecodingMode.REGEX:
            return OulinesRegexLogitsProcessor(specification, llm_engine)
        else:
            raise ValueError(f"Unknown mode: {mode}")