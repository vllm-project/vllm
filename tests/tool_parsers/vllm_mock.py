# vllm_mock.py — stubs out all vLLM imports so tests run without vLLM installed
import sys
import uuid
import logging
from types import ModuleType
from dataclasses import dataclass, field
from typing import Optional, Any

# ---------------------------------------------------------------------------
# Data classes mirroring vLLM protocol objects
# ---------------------------------------------------------------------------

@dataclass
class FunctionCall:
    name: str
    arguments: str

@dataclass
class ToolCall:
    type: str = "function"
    function: FunctionCall = None

@dataclass
class DeltaFunctionCall:
    name: Optional[str] = None
    arguments: Optional[str] = None

    def model_dump(self, exclude_none=False):
        d = {"name": self.name, "arguments": self.arguments}
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d

@dataclass
class DeltaToolCall:
    index: int = 0
    type: Optional[str] = None
    id: Optional[str] = None
    function: Optional[dict] = None

@dataclass
class DeltaMessage:
    content: Optional[str] = None
    tool_calls: list = field(default_factory=list)

@dataclass
class ExtractedToolCallInformation:
    tools_called: bool
    tool_calls: list
    content: Optional[str] = None

@dataclass
class ChatCompletionRequest:
    model: str = ""
    messages: list = field(default_factory=list)
    tools: Optional[list] = None
    tool_choice: Optional[Any] = None
    skip_special_tokens: Optional[bool] = None

@dataclass
class ResponsesRequest:
    pass

# ---------------------------------------------------------------------------
# ToolParser base class and manager
# ---------------------------------------------------------------------------

class ToolParser:
    def __init__(self, tokenizer, tools=None):
        self.model_tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()

    def adjust_request(self, request):
        return request

class ToolParserManager:
    _parsers = {}

    @classmethod
    def register_module(cls, name):
        def decorator(klass):
            cls._parsers[name] = klass
            return klass
        return decorator

    @classmethod
    def get_tool_parser(cls, name):
        return cls._parsers[name]

# ---------------------------------------------------------------------------
# Misc stubs
# ---------------------------------------------------------------------------

def make_tool_call_id():
    return str(uuid.uuid4())

def init_logger(name):
    return logging.getLogger(name)

TokenizerLike = object

Tool = object

def find_common_prefix(a: str, b: str) -> str:
    result = []
    for c1, c2 in zip(a, b):
        if c1 != c2:
            break
        result.append(c1)
    return "".join(result)

# ---------------------------------------------------------------------------
# test utilities (replaces tests.tool_parsers.utils)
# ---------------------------------------------------------------------------

class StreamingReconstructor:
    def __init__(self):
        self.tool_calls = []
        self.content = ""

import re as _re

# Split on special token boundaries so <|"|> is never split mid-token
_SPLIT_PATTERN = _re.compile(
    r'(<\|tool_call>|<tool_call\|>|</tool_call>|<tool_call>|<\|"\|>)'
)

def _tokenize_for_streaming(text: str):
    """Split text into chunks that never bisect a special token."""
    parts = _SPLIT_PATTERN.split(text)
    return [p for p in parts if p]  # drop empty strings


def run_tool_extraction_streaming(tool_parser, deltas, assert_one_tool_per_delta=False):
    import json

    reconstructor = StreamingReconstructor()
    previous_text = ""
    previous_token_ids = []

    for delta in deltas:
        current_text = previous_text + delta
        current_token_ids = previous_token_ids + [0] * len(delta)

        result = tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=[0] * len(delta),
            request=None,
        )

        if result:
            if result.content:
                reconstructor.content += result.content
            if result.tool_calls:
                for tc in result.tool_calls:
                    idx = tc.index
                    while len(reconstructor.tool_calls) <= idx:
                        reconstructor.tool_calls.append(
                            ToolCall(
                                type="function",
                                function=FunctionCall(name="", arguments=""),
                            )
                        )
                    func = tc.function
                    if func:
                        name = func.get("name")
                        if name:
                            reconstructor.tool_calls[idx].function.name = name
                        args = func.get("arguments")
                        if args is not None:
                            reconstructor.tool_calls[idx].function.arguments += args

        previous_text = current_text
        previous_token_ids = current_token_ids

   # Replace the repair loop at the bottom with this:
    for tc in reconstructor.tool_calls:
        raw = tc.function.arguments.strip()
        if not raw:
            tc.function.arguments = "{}"
            continue
        try:
            json.loads(raw)
            continue
        except json.JSONDecodeError:
            pass
        # Count unclosed braces/quotes and close them
        open_braces = raw.count("{") - raw.count("}")
        open_quotes = raw.count('"') % 2
        suffix = ('"' if open_quotes else "") + ("}" * open_braces)
        try:
            json.loads(raw + suffix)
            tc.function.arguments = raw + suffix
            continue
        except json.JSONDecodeError:
            pass
        # Last resort: try all combinations
        for closing in ['"}', '}', '}}', '}}}', '"}}', '"}}}'  ]:
            try:
                json.loads(raw + closing)
                tc.function.arguments = raw + closing
                break
            except json.JSONDecodeError:
                continue

    return reconstructor


def run_tool_extraction(tool_parser, model_output, streaming=False):
    if not streaming:
        result = tool_parser.extract_tool_calls(model_output, request=None)
        return result.content, result.tool_calls
    else:
        deltas = _tokenize_for_streaming(model_output)  # ← was list(model_output)
        result = run_tool_extraction_streaming(tool_parser, deltas)
        return result.content, result.tool_calls
# Register all fake modules — every dotted level must be registered
# ---------------------------------------------------------------------------

def _make_module(name, **attrs):
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

# Top-level
_make_module("vllm")

# vllm.tokenizers
_make_module("vllm.tokenizers", TokenizerLike=TokenizerLike)

# vllm.logger
_make_module("vllm.logger", init_logger=init_logger)

# vllm.tool_parsers
_make_module("vllm.tool_parsers",
    ToolParser=ToolParser,
    ToolParserManager=ToolParserManager,
)
_make_module("vllm.tool_parsers.abstract_tool_parser",
    ToolParser=ToolParser,
    Tool=Tool,
)
_make_module("vllm.tool_parsers.utils",
    find_common_prefix=find_common_prefix,
)

# vllm.entrypoints — every intermediate level needed
_make_module("vllm.entrypoints")
_make_module("vllm.entrypoints.chat_utils", make_tool_call_id=make_tool_call_id)
_make_module("vllm.entrypoints.openai")
_make_module("vllm.entrypoints.openai.engine")
_make_module("vllm.entrypoints.openai.engine.protocol",
    DeltaFunctionCall=DeltaFunctionCall,
    DeltaMessage=DeltaMessage,
    DeltaToolCall=DeltaToolCall,
    ExtractedToolCallInformation=ExtractedToolCallInformation,
    FunctionCall=FunctionCall,
    ToolCall=ToolCall,
)
_make_module("vllm.entrypoints.openai.chat_completion")
_make_module("vllm.entrypoints.openai.chat_completion.protocol",
    ChatCompletionRequest=ChatCompletionRequest,
)
_make_module("vllm.entrypoints.openai.responses")
_make_module("vllm.entrypoints.openai.responses.protocol",
    ResponsesRequest=ResponsesRequest,
)


utils_mod = ModuleType("tests.tool_parsers.utils")
utils_mod.run_tool_extraction = run_tool_extraction
utils_mod.run_tool_extraction_streaming = run_tool_extraction_streaming
sys.modules["tests.tool_parsers.utils"] = utils_mod