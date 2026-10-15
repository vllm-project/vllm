# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)

# Legacy guided-decoding fields are treated like any other extra field since
# their removal, so requests still using them get HTTP 200 with unconstrained
# output and no visible signal (see #53975). Warn instead of burying the
# only breadcrumb at DEBUG level.
_REMOVED_GUIDED_FIELDS = frozenset(
    {
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_grammar",
        "guided_decoding_backend",
        "guided_whitespace_pattern",
    }
)


class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # Compare against both field names and aliases
        extra_keys = data.keys() - field_names
        if extra_keys:
            removed = extra_keys & _REMOVED_GUIDED_FIELDS
            if removed:
                logger.warning_once(
                    "Request contains the removed guided-decoding field(s) "
                    "%s, which are ignored; output will NOT be constrained. "
                    "Use `structured_outputs` (or `response_format`) "
                    "instead; see docs/features/structured_outputs.md.",
                    str(sorted(removed)),
                )
            logger.debug(
                "The following fields were present in the request but ignored: %s",
                extra_keys,
            )
        return result


class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo


class ModelPermission(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: int | None = None
    created_cache_tokens: int | None = None
    multimodal_tokens: dict[str, int] | None = None
    """Prompt tokens contributed by each input modality, keyed by modality name
    (e.g. `image`, `audio`, `video`). A breakdown of the multimodal
    placeholder tokens already counted in `prompt_tokens`; `None` when the
    request has no multimodal input."""


class CompletionTokenUsageInfo(OpenAIBaseModel):
    reasoning_tokens: int = 0


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None
    completion_tokens_details: CompletionTokenUsageInfo | None = None
