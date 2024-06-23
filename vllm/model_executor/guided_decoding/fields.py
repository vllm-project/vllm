from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest)


@dataclass
class GuidedDecodingFields:
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
            self.guided_json is not None,
            self.guided_regex is not None,
            self.guided_choice is not None,
            self.guided_grammar is not None,
            self.guided_json_object is not None,
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding but multiple is "
                f"specified: {self.__dict__}")

    @classmethod
    def from_openai_request(cls, request: Union[CompletionRequest,
                                                ChatCompletionRequest]):
        is_json_object = (request.response_format is not None
                          and request.response_format.type == "json_object")
        return cls(
            guided_json=request.guided_json,
            guided_regex=request.guided_regex,
            guided_choice=request.guided_choice,
            guided_grammar=request.guided_grammar,
            guided_decoding_backend=request.guided_decoding_backend,
            guided_whitespace_pattern=request.guided_whitespace_pattern or " ",
            guided_json_object=is_json_object or None,
        )
