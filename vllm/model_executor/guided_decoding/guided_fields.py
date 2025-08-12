from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, Union

from pydantic import BaseModel


# These classes are deprecated, see SamplingParams
class LLMGuidedOptions(TypedDict, total=False):
    guided_json: Union[Dict, BaseModel, str]
    guided_regex: str
    guided_choice: List[str]
    guided_grammar: str
    guided_decoding_backend: str
    guided_whitespace_pattern: str
    guided_json_object: bool


@dataclass
class GuidedDecodingRequest:
    """One of the fields will be used to retrieve the logit processor."""
    guided_json: Optional[Union[Dict, BaseModel, str]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None
    guided_grammar: Optional[str] = None
    guided_decoding_backend: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None
    guided_json_object: Optional[bool] = None

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        guide_count = sum([
            self.guided_json is not None, self.guided_regex is not None,
            self.guided_choice is not None, self.guided_grammar is not None,
            self.guided_json_object is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding but multiple are "
                f"specified: {self.__dict__}")
