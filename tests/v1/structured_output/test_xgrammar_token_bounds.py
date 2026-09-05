# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm import SamplingParams
from vllm.config import StructuredOutputsConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.structured_output import backend_xgrammar
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

pytestmark = pytest.mark.cpu_test


class _StubModelConfig:
    is_diffusion = False

    @staticmethod
    def get_vocab_size() -> int:
        return 5


class _OneTokenModelConfig:
    is_diffusion = False

    @staticmethod
    def get_vocab_size() -> int:
        return 1


class _TwoTokenModelConfig:
    is_diffusion = False

    @staticmethod
    def get_vocab_size() -> int:
        return 2


class _StubTokenizer:
    pass


class _FailIfCompiled:
    def compile_grammar(self, grammar_spec: str) -> None:
        pytest.fail(f"native grammar compile reached for {grammar_spec!r}")

    def compile_structural_tag(self, grammar_spec: str) -> None:
        pytest.fail(f"native structural-tag compile reached for {grammar_spec!r}")


class _SerializedGrammar:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def serialize_json(self) -> str:
        return json.dumps(self.payload)


class _CountingDecodedVocab(list[str]):
    def __init__(self, values: list[str], owner: "_CountingTokenizerInfo") -> None:
        super().__init__(values)
        self.owner = owner

    def __iter__(self):
        self.owner.decoded_vocab_iteration_count += 1
        return super().__iter__()


class _CountingTokenizerInfo:
    def __init__(self, decoded_vocab: list[str], *, vocab_size: int) -> None:
        self._decoded_vocab = decoded_vocab
        self.decoded_vocab_access_count = 0
        self.decoded_vocab_iteration_count = 0
        self.vocab_size = vocab_size

    @property
    def decoded_vocab(self) -> _CountingDecodedVocab:
        self.decoded_vocab_access_count += 1
        return _CountingDecodedVocab(self._decoded_vocab, self)


def _token_dispatch_payload(token_id: int | float) -> str:
    return json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "token_dispatch",
                "rules": [[token_id, {"type": "token", "token": 0}]],
                "exclude_tokens": [],
            },
        }
    )


def _typeless_token_dispatch_payload(token_id: int | float) -> str:
    return _structural_tag_payload(
        {
            "rules": [[token_id, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        }
    )


def _repeat_token_dispatch_payload(trigger: int | str = 0) -> str:
    return _structural_tag_payload(
        _repeat_format_payload(
            {
                "type": "token_dispatch",
                "rules": [[trigger, {"type": "token", "token": 0}]],
                "exclude_tokens": [],
            }
        )
    )


def _repeat_format_payload(content: dict[str, object]) -> dict[str, object]:
    return {
        "type": "repeat",
        "min": 0,
        "max": 1,
        "content": content,
    }


def _repeat_nested_grammar_payload(token_id: int = 2) -> str:
    return _structural_tag_payload(
        _repeat_format_payload(
            {"type": "grammar", "grammar": f"root ::= Token({token_id})\n"}
        )
    )


def _many_repeat_shadowed_token_dispatch_payload(count: int = 8) -> str:
    return _structural_tag_payload(
        {
            "type": "sequence",
            "elements": [
                _repeat_format_payload(
                    {
                        "type": "token",
                        "token": 0,
                        "rules": [[0, {"type": "token", "token": 0}]],
                        "exclude_tokens": [],
                    }
                )
                for _ in range(count)
            ],
        }
    )


def _structural_tag_payload(format_payload: dict[str, object]) -> str:
    return json.dumps({"type": "structural_tag", "format": format_payload})


def _deep_structural_tag_payload(
    depth: int,
    leaf: dict[str, object],
) -> str:
    format_payload = leaf
    for _ in range(depth):
        format_payload = {"content": format_payload}
    return _structural_tag_payload(format_payload)


def _string_token_format_payloads() -> list[dict[str, object]]:
    return [
        {"type": "token", "token": "<tool>"},
        {
            "type": "token",
            "token": "<tool>",
            "content": {"type": "grammar", "grammar": "this is not ebnf"},
        },
        {
            "elements": [{"type": "token", "token": "<tool>"}],
            "content": {"type": "grammar", "grammar": "this is not ebnf"},
        },
        {"type": "exclude_token", "exclude_tokens": ["<tool>"]},
        {"type": "any_tokens", "exclude_tokens": ["<tool>"]},
        {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>"],
            "exclude_tokens": [],
            "tags": [_token_triggered_tag("<tool>")],
        },
        {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>"],
            "exclude_tokens": [],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"token": "<tool>"},
                    "content": {"type": "any_text"},
                    "end": "</tool>",
                }
            ],
        },
        {
            "type": "token_dispatch",
            "rules": [["<tool>", {"type": "token", "token": "<tool>"}]],
            "exclude_tokens": ["<tool>"],
        },
        {
            "rules": [[0, {"type": "token", "token": "<tool>"}]],
            "exclude_tokens": ["<tool>"],
        },
        {
            "rules": [
                [0, {"type": "token", "token": 0}],
                ["<tool>", {"type": "token", "token": "<tool>"}],
            ],
            "exclude_tokens": [],
        },
        {
            "rules": [["<dispatch>", {"type": "token", "token": "<tool>"}]],
            "exclude_tokens": [],
        },
        {"elements": [{"type": "token", "token": "<tool>"}]},
        {
            "begin": {"type": "token", "token": "<tool>"},
            "content": {"type": "any_text"},
            "end": "</tool>",
        },
        {
            "type": "tag",
            "begin": {"type": "bogus", "token": "<tool>"},
            "content": {"type": "any_text"},
            "end": "</tool>",
        },
        {
            "begin": {"type": "bogus", "token": "<tool>"},
            "content": {"type": "any_text"},
            "end": "</tool>",
        },
        {
            "begin": "<tool>",
            "content": {"type": "any_text"},
            "end": {"token": "<tool>"},
        },
        {
            "type": "repeat",
            "min": 0,
            "max": 1,
            "content": {"type": "token", "token": "<tool>"},
        },
        {
            "type": "tag",
            "begin": "<tool>",
            "content": {"type": "any_text"},
            "end": {"type": "bogus", "token": "<tool>"},
        },
    ]


def _distinct_string_triggered_tags_payload() -> dict[str, object]:
    return {
        "type": "token_triggered_tags",
        "trigger_tokens": ["<a>", "<b>"],
        "exclude_tokens": [],
        "tags": [
            {
                "type": "tag",
                "begin": {"type": "token", "token": "<a>"},
                "content": {"type": "any_text"},
                "end": "</a>",
            },
            {
                "type": "tag",
                "begin": {"type": "token", "token": "<b>"},
                "content": {"type": "any_text"},
                "end": "</b>",
            },
        ],
    }


def _mixed_numeric_string_alias_triggered_tags_payload() -> dict[str, object]:
    return {
        "type": "token_triggered_tags",
        "trigger_tokens": [0],
        "exclude_tokens": [],
        "tags": [
            {
                "type": "tag",
                "begin": {"type": "token", "token": "<tool>"},
                "content": {"type": "any_text"},
                "end": "</tool>",
            }
        ],
    }


def _multi_trigger_mixed_numeric_string_alias_triggered_tags_payload() -> dict[
    str, object
]:
    return {
        "type": "token_triggered_tags",
        "trigger_tokens": [0, 1],
        "exclude_tokens": [],
        "tags": [
            _token_triggered_tag(0),
            _token_triggered_tag("<a>"),
        ],
    }


def _build_hf_tokenizer(vocab: dict[str, int]) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token=None)),
        eos_token=next(iter(vocab)),
    )


def _build_structural_tag_tokenizer() -> PreTrainedTokenizerFast:
    return _build_hf_tokenizer({"<tool>": 0, "<dispatch>": 1})


def _run_structural_tag_subprocess(
    script: str,
    structural_tag: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    worktree_root = str(Path(__file__).resolve().parents[3])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        worktree_root
        if not existing_pythonpath
        else f"{worktree_root}{os.pathsep}{existing_pythonpath}"
    )
    return subprocess.run(
        [sys.executable, "-c", script, structural_tag],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _nested_grammar_placeholder_collision_payload(
    token_id: int = 1,
) -> dict[str, object]:
    return {
        "type": "sequence",
        "elements": [
            {"type": "token", "token": "<tool>"},
            {
                "type": "json_schema",
                "json_schema": {"type": "integer", "minimum": 0},
            },
            {"type": "grammar", "grammar": f"root ::= Token({token_id})\n"},
        ],
    }


def _malformed_token_dependent_sequence_payload() -> str:
    return _structural_tag_payload(
        {
            "type": "sequence",
            "elements": [
                {"type": "token", "token": "<tool>"},
                {"type": "grammar", "grammar": "this is not ebnf"},
            ],
        }
    )


def _hidden_token_dispatch_after_invalid_string_branches_payload() -> dict[str, object]:
    return {
        "begin": {"type": "bogus", "token": ""},
        "content": {"type": "token", "token": ""},
        "end": "</tool>",
        "rules": [[5, {"type": "token", "token": "<tool>"}]],
        "exclude_tokens": [],
    }


def _empty_token_dispatch_trigger_payload() -> dict[str, object]:
    return {
        "type": "token_dispatch",
        "rules": [["", {"type": "token", "token": "<tool>"}]],
        "exclude_tokens": [],
    }


def _token_triggered_tag(token_id: int | float | str) -> dict[str, object]:
    return {
        "type": "tag",
        "begin": {"type": "token", "token": token_id},
        "content": {"type": "any_text"},
        "end": "</tool>",
    }


def _shadowed_typeless_token_dispatch_payload(
    token_id: int | float,
) -> dict[str, object]:
    return {
        "begin": "<tool>",
        "content": {
            "type": "token_triggered_tags",
            "trigger_tokens": [0],
            "tags": [_token_triggered_tag(0)],
            "at_least_one": 1,
        },
        "end": "</tool>",
        "rules": [[token_id, {"type": "token", "token": 0}]],
        "exclude_tokens": [],
    }


def _wrong_typed_tag_shadowed_typeless_token_dispatch_payload(
    token_id: int | float,
) -> dict[str, object]:
    return {
        "triggers": ["<tool>"],
        "tags": [
            {
                "type": "token",
                "begin": "<tool>",
                "content": {"type": "any_text"},
                "end": "</tool>",
            }
        ],
        "rules": [[token_id, {"type": "token", "token": 0}]],
        "exclude_tokens": [],
    }


def _typeless_token_dispatch_shadow(
    token_id: int | float,
    **shadow_fields: object,
) -> dict[str, object]:
    return {
        **shadow_fields,
        "rules": [[token_id, {"type": "token", "token": 0}]],
        "exclude_tokens": [],
    }


def _legacy_structural_tag_payload() -> str:
    return json.dumps(
        {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<tool>",
                    "schema": {"type": "object"},
                    "end": "</tool>",
                }
            ],
            "triggers": ["<tool>"],
        }
    )


def _token_tag_dispatch_grammar(token_id: int) -> str:
    return f'body ::= "ok"\nroot ::= TokenTagDispatch(({token_id}, body))\n'


def _token_tag_dispatch_excludes_grammar(token_id: int) -> str:
    return f"root ::= TokenTagDispatch(excludes=({token_id},))\n"


def _build_chat_request(structural_tag: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        structured_outputs={"structural_tag": structural_tag},
    )


def _build_completion_request(structural_tag: str) -> CompletionRequest:
    return CompletionRequest(
        model="model",
        prompt="hello",
        max_tokens=1,
        structured_outputs={"structural_tag": structural_tag},
    )


def _build_chat_response_format_request(structural_tag: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        response_format=json.loads(structural_tag),
    )


def _build_completion_response_format_request(
    structural_tag: str,
) -> CompletionRequest:
    return CompletionRequest(
        model="model",
        prompt="hello",
        max_tokens=1,
        response_format=json.loads(structural_tag),
    )


@pytest.mark.parametrize(
    "structured_outputs",
    [
        StructuredOutputsParams(grammar="root ::= Token(5)\n"),
        StructuredOutputsParams(grammar=_token_tag_dispatch_grammar(5)),
        StructuredOutputsParams(grammar=_token_tag_dispatch_excludes_grammar(5)),
        StructuredOutputsParams(structural_tag=_token_dispatch_payload(5)),
    ],
)
def test_request_validation_rejects_out_of_range_tokens_before_native_compile(
    structured_outputs: StructuredOutputsParams,
) -> None:
    params = SamplingParams(structured_outputs=structured_outputs)

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        params._validate_structured_outputs(
            _StubModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _StubTokenizer(),
        )


@pytest.mark.parametrize(
    "format_payload",
    [
        {"type": "token", "token": 5},
        {"type": "exclude_token", "exclude_tokens": [5]},
        {"type": "any_tokens", "exclude_tokens": [5]},
        {
            "type": "token_triggered_tags",
            "trigger_tokens": [5],
            "exclude_tokens": [],
            "tags": [_token_triggered_tag(5)],
        },
        {
            "type": "token_triggered_tags",
            "trigger_tokens": [0],
            "exclude_tokens": [5],
            "tags": [_token_triggered_tag(0)],
        },
        {
            "type": "token_dispatch",
            "rules": [[5, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "type": "token_dispatch",
            "rules": [[0, {"type": "token", "token": 0}]],
            "exclude_tokens": [5],
        },
        {
            "rules": [[5, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {"type": "grammar", "grammar": "root ::= Token(5)\n"},
    ],
)
def test_request_validation_rejects_structural_tag_token_fields_before_native_compile(
    format_payload: dict[str, object],
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(format_payload)
        )
    )

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        params._validate_structured_outputs(
            _StubModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _build_structural_tag_tokenizer(),
        )


def test_request_validation_rejects_negative_repeat_token_before_native_compile() -> (
    None
):
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                {
                    "type": "repeat",
                    "min": 0,
                    "max": 1,
                    "content": {"type": "token", "token": -1},
                }
            )
        )
    )

    with pytest.raises(
        (ValueError, VLLMValidationError), match="Invalid structural tag specification"
    ):
        params._validate_structured_outputs(
            _StubModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _build_structural_tag_tokenizer(),
        )


@pytest.mark.parametrize(
    "structural_tag",
    [
        _repeat_token_dispatch_payload("<tool>"),
        _structural_tag_payload(
            _repeat_format_payload(
                {
                    "rules": [[0, {"type": "token", "token": 0}]],
                    "exclude_tokens": [],
                }
            )
        ),
        _structural_tag_payload(
            _repeat_format_payload(
                {
                    "type": "sequence",
                    "elements": [
                        {
                            "type": "token_dispatch",
                            "rules": [[0, {"type": "token", "token": 0}]],
                            "exclude_tokens": [],
                        }
                    ],
                }
            )
        ),
        _structural_tag_payload(
            {
                "type": "sequence",
                "elements": [
                    _repeat_format_payload(
                        {
                            "type": "token_dispatch",
                            "rules": [[0, {"type": "token", "token": 0}]],
                            "exclude_tokens": [],
                        }
                    ),
                    {
                        "type": "token_dispatch",
                        "rules": [[0, {"type": "token", "token": 0}]],
                        "exclude_tokens": [],
                    },
                ],
            }
        ),
    ],
)
def test_openai_request_rejects_repeat_wrapped_token_dispatch_before_native_parse(
    structural_tag: str,
) -> None:
    result = _run_structural_tag_subprocess(
        """
import sys
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

try:
    ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        structured_outputs={"structural_tag": sys.argv[1]},
    )
except Exception as exc:
    if "Invalid structured_outputs structural_tag specification" not in str(exc):
        raise
else:
    raise AssertionError("repeat-wrapped token_dispatch should be rejected")
""",
        structural_tag,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "wrapper_type",
    ["optional", "plus", "star"],
    ids=["zero-or-one", "one-or-more", "zero-or-more"],
)
def test_request_validation_preserves_non_repeat_token_dispatch_wrappers(
    wrapper_type: str,
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                {
                    "type": wrapper_type,
                    "content": {
                        "type": "token_dispatch",
                        "rules": [[0, {"type": "token", "token": 0}]],
                        "exclude_tokens": [],
                    },
                }
            )
        )
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _StubTokenizer(),
    )


def test_request_validation_preserves_token_dispatch_outside_repeat_subtree() -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                {
                    "type": "sequence",
                    "elements": [
                        _repeat_format_payload({"type": "token", "token": 0}),
                        {
                            "type": "token_dispatch",
                            "rules": [[0, {"type": "token", "token": 0}]],
                            "exclude_tokens": [],
                        },
                    ],
                }
            )
        )
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _StubTokenizer(),
    )


def test_openai_request_preserves_repeat_nested_grammar_token_id() -> None:
    request = _build_chat_request(_repeat_nested_grammar_payload())

    assert request.model == "model"


def test_repeat_probe_uses_bounded_native_parses_for_candidate_shaped_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(_many_repeat_shadowed_token_dispatch_payload())
    original = backend_xgrammar._parse_xgrammar_structural_tag_probe
    parse_count = 0

    def count_probe_calls(*args: Any, **kwargs: Any):
        nonlocal parse_count
        parse_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        backend_xgrammar,
        "_parse_xgrammar_structural_tag_probe",
        count_probe_calls,
    )

    assert not backend_xgrammar._xgrammar_has_repeat_token_dispatch(payload)
    assert parse_count == 2


def test_repeat_probe_uses_bounded_native_parses_for_active_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(_repeat_token_dispatch_payload())
    original = backend_xgrammar._parse_xgrammar_structural_tag_probe
    parse_count = 0

    def count_probe_calls(*args: Any, **kwargs: Any):
        nonlocal parse_count
        parse_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        backend_xgrammar,
        "_parse_xgrammar_structural_tag_probe",
        count_probe_calls,
    )

    assert backend_xgrammar._xgrammar_has_repeat_token_dispatch(payload)
    assert parse_count == 2


@pytest.mark.parametrize(
    "format_payload",
    [
        {"type": "token", "token": 5.0},
        {"type": "exclude_token", "exclude_tokens": [5.0]},
        {"type": "any_tokens", "exclude_tokens": [5.0]},
        {
            "type": "token_triggered_tags",
            "trigger_tokens": [5.0],
            "exclude_tokens": [],
            "tags": [_token_triggered_tag(5.0)],
        },
        {
            "type": "token_triggered_tags",
            "trigger_tokens": [0],
            "exclude_tokens": [5.0],
            "tags": [_token_triggered_tag(0)],
        },
        {
            "type": "token_dispatch",
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "type": "token_dispatch",
            "rules": [[0, {"type": "token", "token": 0}]],
            "exclude_tokens": [5.0],
        },
        {
            "type": "token_dispatch",
            "rules": [[0, {"type": "token", "token": 5.0}]],
            "exclude_tokens": [],
        },
        {
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "value": 1,
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "excludes": 1,
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "begin": 1,
            "content": {},
            "end": "</tool>",
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "tags": [
                {
                    "begin": "<tool>",
                    "content": {"type": "any_text"},
                    "end": "</tool>",
                }
            ],
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        _shadowed_typeless_token_dispatch_payload(5.0),
        _wrong_typed_tag_shadowed_typeless_token_dispatch_payload(5.0),
        _typeless_token_dispatch_shadow(5.0, json_schema={}, style="bogus"),
        _typeless_token_dispatch_shadow(
            5.0,
            triggers=[""],
            tags=[
                {
                    "begin": "<tool>",
                    "content": {"type": "any_text"},
                    "end": "</tool>",
                }
            ],
        ),
    ],
)
def test_request_rejects_coerced_structural_tag_tokens_before_native_compile(
    format_payload: dict[str, object],
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(format_payload)
        )
    )

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        params._validate_structured_outputs(
            _StubModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _build_structural_tag_tokenizer(),
        )


def test_request_validation_preserves_in_range_coerced_structural_tag_token_ids() -> (
    None
):
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload({"type": "token", "token": 4.0})
        )
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _StubTokenizer(),
    )


@pytest.mark.parametrize("format_payload", _string_token_format_payloads())
def test_request_validation_preserves_tokenizer_resolved_structural_tag_tokens(
    format_payload: dict[str, object],
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(format_payload)
        )
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _build_structural_tag_tokenizer(),
    )


def test_request_validation_preserves_distinct_string_trigger_identities() -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                _distinct_string_triggered_tags_payload()
            )
        )
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _build_hf_tokenizer({"<a>": 0, "<b>": 1}),
    )


def test_request_validation_preserves_mixed_numeric_string_token_aliases() -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                _mixed_numeric_string_alias_triggered_tags_payload()
            )
        )
    )

    params._validate_structured_outputs(
        _OneTokenModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _build_hf_tokenizer({"<tool>": 0}),
    )


def test_request_validation_preserves_multi_trigger_mixed_numeric_string_aliases() -> (
    None
):
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                _multi_trigger_mixed_numeric_string_alias_triggered_tags_payload()
            )
        )
    )

    params._validate_structured_outputs(
        _TwoTokenModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _build_hf_tokenizer({"<zero>": 0, "<a>": 1}),
    )


@pytest.mark.parametrize(
    "format_payload",
    [
        {"type": "token", "token": "<missing>"},
        {
            "type": "repeat",
            "min": 0,
            "max": 1,
            "content": {"type": "token", "token": "<missing>"},
        },
    ],
)
def test_request_validation_rejects_unknown_string_tokens_with_padded_vocab(
    format_payload: dict[str, object],
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(format_payload)
        )
    )

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        params._validate_structured_outputs(
            _TwoTokenModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _build_hf_tokenizer({"<tool>": 0}),
        )


def test_request_validation_caches_tokenizer_info_for_string_structural_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _build_hf_tokenizer({"<tool>": 0})
    original = backend_xgrammar._build_xgrammar_tokenizer_info
    build_count = 0

    def count_builds(*args: object, **kwargs: object) -> Any:
        nonlocal build_count
        build_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        backend_xgrammar,
        "_build_xgrammar_tokenizer_info",
        count_builds,
    )

    for _ in range(2):
        params = SamplingParams(
            structured_outputs=StructuredOutputsParams(
                structural_tag=_structural_tag_payload(
                    {"type": "token", "token": "<tool>"}
                )
            )
        )
        params._validate_structured_outputs(
            _OneTokenModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            tokenizer,
        )

    assert build_count == 1


def test_request_validation_caches_tokenizer_reverse_vocab_for_string_structural_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_info = _CountingTokenizerInfo(["<tool>"], vocab_size=1)
    monkeypatch.setattr(
        backend_xgrammar,
        "_get_cached_xgrammar_tokenizer_info",
        lambda tokenizer: tokenizer_info,
    )

    for _ in range(2):
        params = SamplingParams(
            structured_outputs=StructuredOutputsParams(
                structural_tag=_structural_tag_payload(
                    {"type": "token", "token": "<tool>"}
                )
            )
        )
        params._validate_structured_outputs(
            _OneTokenModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _build_hf_tokenizer({"<tool>": 0}),
        )

    assert tokenizer_info.decoded_vocab_access_count == 1
    assert tokenizer_info.decoded_vocab_iteration_count == 1


def test_request_validation_rejects_string_tags_without_tokenizer_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_build(*args: object, **kwargs: object) -> Any:
        raise ValueError("unsupported tokenizer")

    monkeypatch.setattr(
        backend_xgrammar,
        "_build_xgrammar_tokenizer_info",
        fail_build,
    )
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload({"type": "token", "token": "<tool>"})
        )
    )

    with pytest.raises(
        (ValueError, VLLMValidationError), match="Invalid structural tag specification"
    ):
        params._validate_structured_outputs(
            _OneTokenModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _build_hf_tokenizer({"<tool>": 0}),
        )


@pytest.mark.parametrize("token_id", [1, 2])
def test_request_validation_rejects_nested_grammar_token_placeholder_collisions(
    token_id: int,
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                _nested_grammar_placeholder_collision_payload(token_id)
            )
        )
    )

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        params._validate_structured_outputs(
            _OneTokenModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _build_hf_tokenizer({"<tool>": 0}),
        )


def test_request_validation_rejects_hidden_token_dispatch_after_invalid_branches() -> (
    None
):
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                _hidden_token_dispatch_after_invalid_string_branches_payload()
            )
        )
    )

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        params._validate_structured_outputs(
            _StubModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _build_structural_tag_tokenizer(),
        )


def test_request_validation_preserves_empty_token_dispatch_trigger() -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                _empty_token_dispatch_trigger_payload()
            )
        )
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _build_hf_tokenizer({"": 0, "<tool>": 1}),
    )


@pytest.mark.parametrize(
    "format_payload",
    [
        {
            "rules": [[4.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {"elements": [{"type": "token", "token": 4.0}]},
        {
            "begin": {"token": 4.0},
            "content": {"type": "any_text"},
            "end": "</tool>",
        },
        {
            "value": "ok",
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "begin": "<tool>",
            "content": {"type": "any_text"},
            "end": "</tool>",
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
    ],
)
def test_request_validation_preserves_native_typeless_structural_tag_formats(
    format_payload: dict[str, object],
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(format_payload)
        )
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _build_structural_tag_tokenizer(),
    )


@pytest.mark.parametrize(
    "format_payload",
    [
        {
            "type": "token",
            "token": 0,
            "content": {
                "type": "repeat",
                "min": 0,
                "max": 1,
                "content": {
                    "type": "token_dispatch",
                    "rules": [[0, {"type": "token", "token": 0}]],
                    "exclude_tokens": [],
                },
            },
        },
        {
            "elements": [{"type": "token", "token": 0}],
            "content": {
                "type": "repeat",
                "min": 0,
                "max": 1,
                "content": {
                    "type": "token_dispatch",
                    "rules": [[0, {"type": "token", "token": 0}]],
                    "exclude_tokens": [],
                },
            },
        },
    ],
)
def test_request_validation_preserves_shadowed_repeat_token_dispatch(
    format_payload: dict[str, object],
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(format_payload)
        )
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _StubTokenizer(),
    )


def test_request_validation_preserves_native_typed_empty_const_string() -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            structural_tag=_structural_tag_payload(
                {"type": "const_string", "value": ""}
            )
        )
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _StubTokenizer(),
    )


def test_request_validation_rejects_non_string_structural_tag_type() -> None:
    structural_tag = json.dumps(
        {
            "type": None,
            "format": {"type": "const_string", "value": "ok"},
        }
    )

    with pytest.raises(
        (ValueError, VLLMValidationError), match="Invalid structural tag specification"
    ):
        backend_xgrammar.validate_xgrammar_structural_tag_syntax(structural_tag)


@pytest.mark.parametrize(
    "grammar",
    [
        'root ::= "Token(5)"\n',
        'root ::= # Token(5)\n "ok"\n',
        'root ::= "TokenTagDispatch((5, body))"\n',
        'root ::= # TokenTagDispatch((5, body))\n "ok"\n',
    ],
)
def test_request_validation_ignores_token_macros_in_literals_and_comments(
    grammar: str,
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(grammar=grammar),
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _StubTokenizer(),
    )


@pytest.mark.parametrize(
    "grammar",
    [
        "root ::= [#] Token(5)\n",
        "root ::= # comment ends at CR\rToken(5)\r",
    ],
)
def test_request_validation_rejects_tokens_after_native_ebnf_comment_boundaries(
    grammar: str,
) -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(grammar=grammar),
    )

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        params._validate_structured_outputs(
            _StubModelConfig(),
            StructuredOutputsConfig(backend="xgrammar"),
            _StubTokenizer(),
        )


def test_request_validation_preserves_native_token_dispatch_rule_names() -> None:
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            grammar='body.5 ::= "x"\nroot ::= TokenTagDispatch((0, body.5))\n',
        ),
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _StubTokenizer(),
    )


def test_request_validation_preserves_native_deep_structural_tag() -> None:
    structural_tag = _deep_structural_tag_payload(600, {"value": "ok"})
    backend_xgrammar.validate_xgrammar_structural_tag_syntax(structural_tag)
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(structural_tag=structural_tag)
    )

    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend="xgrammar"),
        _StubTokenizer(),
    )


@pytest.mark.parametrize("version", ["v13", "v14"])
def test_token_bounds_accept_supported_xgrammar_serialization_versions(
    version: str,
) -> None:
    grammar = _SerializedGrammar(
        {
            "__VERSION__": version,
            "grammar_expr_data": [0],
            "grammar_expr_indptr": [9, 1, 4],
        }
    )

    backend_xgrammar._validate_xgrammar_grammar_token_ids(
        cast(Any, grammar),
        vocab_size=5,
    )


def test_token_bounds_fail_closed_for_unknown_xgrammar_serialization_version() -> None:
    grammar = _SerializedGrammar(
        {
            "__VERSION__": "v999",
            "grammar_expr_data": [0],
            "grammar_expr_indptr": [9, 1, 4],
        }
    )

    with pytest.raises(
        ValueError,
        match="Unsupported xgrammar grammar serialization version",
    ):
        backend_xgrammar._validate_xgrammar_grammar_token_ids(
            cast(Any, grammar),
            vocab_size=5,
        )


def test_public_xgrammar_validator_translates_token_bounds_to_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grammar = _SerializedGrammar(
        {
            "__VERSION__": "v14",
            "grammar_expr_data": [0],
            "grammar_expr_indptr": [9, 1, 5],
        }
    )
    grammar_type = SimpleNamespace(from_ebnf=lambda _: grammar)
    monkeypatch.setattr(
        backend_xgrammar,
        "xgr",
        SimpleNamespace(Grammar=grammar_type),
    )
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(grammar='root ::= "ok"')
    )

    with pytest.raises(VLLMValidationError, match="outside the tokenizer"):
        backend_xgrammar.validate_xgrammar_grammar(params, vocab_size=5)


@pytest.mark.parametrize(
    "request_builder",
    [
        _build_chat_request,
        _build_completion_request,
        _build_chat_response_format_request,
        _build_completion_response_format_request,
    ],
)
@pytest.mark.parametrize(
    "structural_tag",
    [
        _token_dispatch_payload(5),
        _typeless_token_dispatch_payload(4.0),
        _legacy_structural_tag_payload(),
    ],
)
def test_openai_request_validation_accepts_parse_only_structural_tags_without_vocab(
    request_builder: Callable[[str], Any],
    structural_tag: str,
) -> None:
    request = request_builder(structural_tag)

    assert request.model == "model"


@pytest.mark.parametrize(
    "request_builder",
    [
        _build_chat_request,
        _build_completion_request,
        _build_chat_response_format_request,
        _build_completion_response_format_request,
    ],
)
@pytest.mark.parametrize("format_payload", _string_token_format_payloads())
def test_openai_request_validation_accepts_string_tokens_without_vocab(
    request_builder: Callable[[str], Any],
    format_payload: dict[str, object],
) -> None:
    request = request_builder(_structural_tag_payload(format_payload))

    assert request.model == "model"


@pytest.mark.parametrize(
    "request_builder",
    [
        _build_chat_request,
        _build_completion_request,
        _build_chat_response_format_request,
        _build_completion_response_format_request,
    ],
)
def test_openai_request_validation_accepts_distinct_string_trigger_identities(
    request_builder: Callable[[str], Any],
) -> None:
    request = request_builder(
        _structural_tag_payload(_distinct_string_triggered_tags_payload())
    )

    assert request.model == "model"


@pytest.mark.parametrize(
    "request_builder",
    [
        _build_chat_request,
        _build_completion_request,
        _build_chat_response_format_request,
        _build_completion_response_format_request,
    ],
)
def test_openai_request_validation_accepts_mixed_numeric_string_token_aliases(
    request_builder: Callable[[str], Any],
) -> None:
    request = request_builder(
        _structural_tag_payload(_mixed_numeric_string_alias_triggered_tags_payload())
    )

    assert request.model == "model"


@pytest.mark.parametrize(
    "request_builder",
    [
        _build_chat_request,
        _build_completion_request,
        _build_chat_response_format_request,
        _build_completion_response_format_request,
    ],
)
def test_openai_request_validation_accepts_multi_trigger_mixed_numeric_string_aliases(
    request_builder: Callable[[str], Any],
) -> None:
    request = request_builder(
        _structural_tag_payload(
            _multi_trigger_mixed_numeric_string_alias_triggered_tags_payload()
        )
    )

    assert request.model == "model"


@pytest.mark.parametrize(
    "request_builder",
    [
        _build_chat_request,
        _build_completion_request,
        _build_chat_response_format_request,
        _build_completion_response_format_request,
    ],
)
def test_openai_request_validation_rejects_malformed_token_dependent_sequence(
    request_builder: Callable[[str], Any],
) -> None:
    with pytest.raises(
        (ValueError, VLLMValidationError),
        match="Invalid .* structural_tag specification",
    ):
        request_builder(_malformed_token_dependent_sequence_payload())


@pytest.mark.parametrize(
    ("request_type", "grammar_spec"),
    [
        (StructuredOutputOptions.GRAMMAR, "root ::= Token(5)\n"),
        (
            StructuredOutputOptions.GRAMMAR,
            _token_tag_dispatch_grammar(5),
        ),
        (
            StructuredOutputOptions.GRAMMAR,
            _token_tag_dispatch_excludes_grammar(5),
        ),
        (
            StructuredOutputOptions.STRUCTURAL_TAG,
            _token_dispatch_payload(5),
        ),
    ],
)
def test_backend_rejects_out_of_range_tokens_before_native_compile(
    request_type: StructuredOutputOptions,
    grammar_spec: str,
) -> None:
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    backend.compiler = _FailIfCompiled()
    backend.vocab_size = 5
    backend.num_speculative_tokens = 0

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        backend.compile_grammar(request_type, grammar_spec)


def test_backend_rejects_negative_repeat_token_before_native_compile() -> None:
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    backend.compiler = _FailIfCompiled()
    backend.vocab_size = 5
    backend.num_speculative_tokens = 0

    with pytest.raises(
        (ValueError, VLLMValidationError), match="Invalid structural tag specification"
    ):
        backend.compile_grammar(
            StructuredOutputOptions.STRUCTURAL_TAG,
            _structural_tag_payload(
                {
                    "type": "repeat",
                    "min": 0,
                    "max": 1,
                    "content": {"type": "token", "token": -1},
                }
            ),
        )


@pytest.mark.parametrize(
    "structural_tag",
    [
        _repeat_token_dispatch_payload(0),
        _structural_tag_payload(
            _repeat_format_payload(
                {
                    "rules": [[0, {"type": "token", "token": 0}]],
                    "exclude_tokens": [],
                }
            )
        ),
        _structural_tag_payload(
            _repeat_format_payload(
                {
                    "type": "sequence",
                    "elements": [
                        {
                            "type": "token_dispatch",
                            "rules": [[0, {"type": "token", "token": 0}]],
                            "exclude_tokens": [],
                        }
                    ],
                }
            )
        ),
        _structural_tag_payload(
            {
                "type": "sequence",
                "elements": [
                    _repeat_format_payload(
                        {
                            "type": "token_dispatch",
                            "rules": [[0, {"type": "token", "token": 0}]],
                            "exclude_tokens": [],
                        }
                    ),
                    {
                        "type": "token_dispatch",
                        "rules": [[0, {"type": "token", "token": 0}]],
                        "exclude_tokens": [],
                    },
                ],
            }
        ),
    ],
)
def test_backend_rejects_repeat_wrapped_token_dispatch_before_native_parse(
    structural_tag: str,
) -> None:
    result = _run_structural_tag_subprocess(
        """
import sys
from vllm.v1.structured_output import backend_xgrammar
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

class FailIfCompiled:
    def compile_grammar(self, grammar_spec):
        raise AssertionError(f"native grammar compile reached for {grammar_spec!r}")

    def compile_structural_tag(self, grammar_spec):
        raise AssertionError(
            f"native structural-tag compile reached for {grammar_spec!r}"
        )

backend = object.__new__(backend_xgrammar.XgrammarBackend)
backend.compiler = FailIfCompiled()
backend.vocab_size = 5
backend.num_speculative_tokens = 0

try:
    backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, sys.argv[1])
except ValueError as exc:
    if "Invalid structural tag specification" not in str(exc):
        raise
else:
    raise AssertionError("repeat-wrapped token_dispatch should be rejected")
""",
        structural_tag,
    )

    assert result.returncode == 0, result.stderr


def test_backend_preserves_repeat_wrapped_nested_grammar_token_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_xgrammar.xgr, "GrammarMatcher", lambda *args, **kwargs: object()
    )
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    tokenizer_info = backend_xgrammar.xgr.TokenizerInfo(
        ["a", "b", "c"],
        backend_xgrammar.xgr.VocabType.RAW,
        vocab_size=3,
    )
    backend.compiler = backend_xgrammar.xgr.GrammarCompiler(tokenizer_info)
    backend._structural_tag_validation_tokenizer_info = tokenizer_info
    backend.vocab_size = 3
    backend.num_speculative_tokens = 0

    backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        _repeat_nested_grammar_payload(),
    )


def test_backend_preserves_tokenizer_resolved_deep_structural_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_xgrammar.xgr, "GrammarMatcher", lambda *args, **kwargs: object()
    )
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    tokenizer_info = backend_xgrammar.xgr.TokenizerInfo(
        ["<tool>"],
        backend_xgrammar.xgr.VocabType.RAW,
        vocab_size=1,
    )
    backend.compiler = backend_xgrammar.xgr.GrammarCompiler(tokenizer_info)
    backend._structural_tag_validation_compiler = backend.compiler
    backend._structural_tag_validation_tokenizer_info = tokenizer_info
    backend.vocab_size = 1
    backend.num_speculative_tokens = 0

    backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        _deep_structural_tag_payload(600, {"type": "token", "token": "<tool>"}),
    )


@pytest.mark.parametrize(
    "format_payload",
    [
        {"type": "token", "token": 5.0},
        {"type": "exclude_token", "exclude_tokens": [5.0]},
        {"type": "any_tokens", "exclude_tokens": [5.0]},
        {
            "type": "token_triggered_tags",
            "trigger_tokens": [5.0],
            "exclude_tokens": [],
            "tags": [_token_triggered_tag(5.0)],
        },
        {
            "type": "token_triggered_tags",
            "trigger_tokens": [0],
            "exclude_tokens": [5.0],
            "tags": [_token_triggered_tag(0)],
        },
        {
            "type": "token_dispatch",
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "type": "token_dispatch",
            "rules": [[0, {"type": "token", "token": 0}]],
            "exclude_tokens": [5.0],
        },
        {
            "type": "token_dispatch",
            "rules": [[0, {"type": "token", "token": 5.0}]],
            "exclude_tokens": [],
        },
        {
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "value": 1,
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "excludes": 1,
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "begin": 1,
            "content": {},
            "end": "</tool>",
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        {
            "tags": [
                {
                    "begin": "<tool>",
                    "content": {"type": "any_text"},
                    "end": "</tool>",
                }
            ],
            "rules": [[5.0, {"type": "token", "token": 0}]],
            "exclude_tokens": [],
        },
        _shadowed_typeless_token_dispatch_payload(5.0),
        _wrong_typed_tag_shadowed_typeless_token_dispatch_payload(5.0),
        _typeless_token_dispatch_shadow(5.0, json_schema={}, style="bogus"),
        _typeless_token_dispatch_shadow(
            5.0,
            triggers=[""],
            tags=[
                {
                    "begin": "<tool>",
                    "content": {"type": "any_text"},
                    "end": "</tool>",
                }
            ],
        ),
    ],
)
def test_backend_rejects_coerced_structural_tag_token_ids_before_native_compile(
    format_payload: dict[str, object],
) -> None:
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    backend.compiler = _FailIfCompiled()
    backend.vocab_size = 5

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        backend.compile_grammar(
            StructuredOutputOptions.STRUCTURAL_TAG,
            _structural_tag_payload(format_payload),
        )


@pytest.mark.parametrize("format_payload", _string_token_format_payloads())
def test_backend_preserves_tokenizer_resolved_structural_tag_tokens(
    format_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_xgrammar.xgr, "GrammarMatcher", lambda *args, **kwargs: object()
    )
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    tokenizer_info = backend_xgrammar.xgr.TokenizerInfo(
        ["x", "<tool>"],
        backend_xgrammar.xgr.VocabType.RAW,
        vocab_size=2,
    )
    backend.compiler = backend_xgrammar.xgr.GrammarCompiler(tokenizer_info)
    backend._structural_tag_validation_compiler = backend.compiler
    backend._structural_tag_validation_tokenizer_info = tokenizer_info
    backend.vocab_size = 2
    backend.num_speculative_tokens = 0

    grammar = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        _structural_tag_payload(format_payload),
    )

    assert 1 in list(backend_xgrammar._iter_xgrammar_token_ids(grammar.ctx.grammar))


def test_backend_preserves_distinct_string_trigger_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_xgrammar.xgr, "GrammarMatcher", lambda *args, **kwargs: object()
    )
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    tokenizer_info = backend_xgrammar.xgr.TokenizerInfo(
        ["x", "<a>", "<b>"],
        backend_xgrammar.xgr.VocabType.RAW,
        vocab_size=3,
    )
    backend.compiler = backend_xgrammar.xgr.GrammarCompiler(tokenizer_info)
    backend._structural_tag_validation_compiler = backend.compiler
    backend._structural_tag_validation_tokenizer_info = tokenizer_info
    backend.vocab_size = 3
    backend.num_speculative_tokens = 0

    grammar = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        _structural_tag_payload(_distinct_string_triggered_tags_payload()),
    )

    token_ids = set(backend_xgrammar._iter_xgrammar_token_ids(grammar.ctx.grammar))

    assert {1, 2}.issubset(token_ids)


def test_backend_preserves_mixed_numeric_string_token_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_xgrammar.xgr, "GrammarMatcher", lambda *args, **kwargs: object()
    )
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    tokenizer_info = backend_xgrammar.xgr.TokenizerInfo(
        ["<tool>"],
        backend_xgrammar.xgr.VocabType.RAW,
        vocab_size=1,
    )
    backend.compiler = backend_xgrammar.xgr.GrammarCompiler(tokenizer_info)
    backend._structural_tag_validation_compiler = backend.compiler
    backend._structural_tag_validation_tokenizer_info = tokenizer_info
    backend.vocab_size = 1
    backend.num_speculative_tokens = 0

    grammar = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        _structural_tag_payload(_mixed_numeric_string_alias_triggered_tags_payload()),
    )

    assert list(backend_xgrammar._iter_xgrammar_token_ids(grammar.ctx.grammar)) == [0]


def test_backend_preserves_multi_trigger_mixed_numeric_string_token_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_xgrammar.xgr, "GrammarMatcher", lambda *args, **kwargs: object()
    )
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    tokenizer_info = backend_xgrammar.xgr.TokenizerInfo(
        ["<zero>", "<a>"],
        backend_xgrammar.xgr.VocabType.RAW,
        vocab_size=2,
    )
    backend.compiler = backend_xgrammar.xgr.GrammarCompiler(tokenizer_info)
    backend._structural_tag_validation_compiler = backend.compiler
    backend._structural_tag_validation_tokenizer_info = tokenizer_info
    backend.vocab_size = 2
    backend.num_speculative_tokens = 0

    grammar = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        _structural_tag_payload(
            _multi_trigger_mixed_numeric_string_alias_triggered_tags_payload()
        ),
    )

    assert set(backend_xgrammar._iter_xgrammar_token_ids(grammar.ctx.grammar)) == {
        0,
        1,
    }


@pytest.mark.parametrize("token_id", [1, 2])
def test_backend_rejects_nested_grammar_token_placeholder_collisions(
    token_id: int,
) -> None:
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    backend.compiler = _FailIfCompiled()
    backend._structural_tag_validation_compiler = _FailIfCompiled()
    backend.vocab_size = 1
    backend.num_speculative_tokens = 0

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        backend.compile_grammar(
            StructuredOutputOptions.STRUCTURAL_TAG,
            _structural_tag_payload(
                _nested_grammar_placeholder_collision_payload(token_id)
            ),
        )


def test_backend_rejects_hidden_token_dispatch_after_invalid_string_branches() -> None:
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    backend.compiler = _FailIfCompiled()
    backend._structural_tag_validation_compiler = _FailIfCompiled()
    backend.vocab_size = 5
    backend.num_speculative_tokens = 0

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        backend.compile_grammar(
            StructuredOutputOptions.STRUCTURAL_TAG,
            _structural_tag_payload(
                _hidden_token_dispatch_after_invalid_string_branches_payload()
            ),
        )


def test_backend_preserves_empty_token_dispatch_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_xgrammar.xgr, "GrammarMatcher", lambda *args, **kwargs: object()
    )
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    tokenizer_info = backend_xgrammar.xgr.TokenizerInfo(
        ["", "<tool>"],
        backend_xgrammar.xgr.VocabType.RAW,
        vocab_size=2,
    )
    backend.compiler = backend_xgrammar.xgr.GrammarCompiler(tokenizer_info)
    backend._structural_tag_validation_compiler = backend.compiler
    backend._structural_tag_validation_tokenizer_info = tokenizer_info
    backend.vocab_size = 2
    backend.num_speculative_tokens = 0

    grammar = backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG,
        _structural_tag_payload(_empty_token_dispatch_trigger_payload()),
    )

    token_ids = set(backend_xgrammar._iter_xgrammar_token_ids(grammar.ctx.grammar))

    assert {0, 1}.issubset(token_ids)


def test_backend_rejects_mixed_string_and_oob_token_before_compile() -> None:
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    backend.compiler = _FailIfCompiled()
    backend.vocab_size = 5

    with pytest.raises((ValueError, VLLMValidationError), match="token ID"):
        backend.compile_grammar(
            StructuredOutputOptions.STRUCTURAL_TAG,
            _structural_tag_payload(
                {
                    "type": "token_dispatch",
                    "rules": [["<tool>", {"type": "token", "token": 2147483647}]],
                    "exclude_tokens": [],
                }
            ),
        )


@pytest.mark.parametrize(
    "format_payload",
    [
        {"type": "token", "token": "<tool>"},
        {
            "type": "repeat",
            "min": 0,
            "max": 1,
            "content": {"type": "token", "token": "<tool>"},
        },
    ],
)
def test_backend_rejects_tokenizer_resolved_out_of_range_token_ids(
    format_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_xgrammar.xgr, "GrammarMatcher", lambda *args, **kwargs: object()
    )
    backend = object.__new__(backend_xgrammar.XgrammarBackend)
    backend.compiler = _FailIfCompiled()
    tokenizer_info = backend_xgrammar.xgr.TokenizerInfo(
        ["x", "<tool>"],
        backend_xgrammar.xgr.VocabType.RAW,
        vocab_size=1,
    )
    backend._structural_tag_validation_compiler = backend_xgrammar.xgr.GrammarCompiler(
        tokenizer_info
    )
    backend._structural_tag_validation_tokenizer_info = tokenizer_info
    backend.vocab_size = 1
    backend.num_speculative_tokens = 0

    with pytest.raises((ValueError, VLLMValidationError), match="token ID 1"):
        backend.compile_grammar(
            StructuredOutputOptions.STRUCTURAL_TAG,
            _structural_tag_payload(format_payload),
        )
