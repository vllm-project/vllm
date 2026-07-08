# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import BFCLDataset, get_samples


def _patch_hf_api(side_effect):
    """Return a patch context that swaps `hf_api()` to a stub whose
    `.hf_hub_download` attribute uses `side_effect`."""
    fake_api = MagicMock()
    fake_api.hf_hub_download.side_effect = side_effect
    return patch("vllm.benchmarks.datasets.datasets.hf_api", return_value=fake_api)


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("gpt2")


_FAKE_ROWS = {
    "simple": [
        {
            "id": "simple_0",
            "question": [
                [
                    {
                        "role": "user",
                        "content": "What is 2+2?",
                    }
                ]
            ],
            "function": [
                {
                    "name": "add",
                    "description": "Add two numbers.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "a": {"type": "integer", "description": "first"},
                            "b": {"type": "float", "description": "second"},
                        },
                        "required": ["a", "b"],
                    },
                }
            ],
        },
    ],
    "live_simple": [
        {
            "id": "live_simple_0",
            "question": [[{"role": "user", "content": "Tell me the weather."}]],
            "function": [
                {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "city": {"type": "any", "description": "city"},
                            "coords": {"type": "tuple", "description": "coords"},
                        },
                        "required": ["city"],
                    },
                }
            ],
        },
    ],
}


def _write_fake_files(tmp_path: Path) -> dict[str, Path]:
    """Write fake BFCL JSONL files mimicking the HF repo layout."""
    paths = {}
    for category, rows in _FAKE_ROWS.items():
        p = tmp_path / f"BFCL_v3_{category}.json"
        with p.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        paths[category] = p
    return paths


def _args_for_bfcl(categories: list[str] | None) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="hf",
        dataset_path="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        hf_name=None,
        hf_subset=None,
        hf_split=None,
        hf_output_len=64,
        disable_shuffle=True,
        num_prompts=2,
        no_oversample=False,
        no_stream=True,
        seed=0,
        request_id_prefix="",
        trust_remote_code=False,
        skip_chat_template=False,
        enable_multimodal_chat=False,
        backend="openai-chat",
        bfcl_categories=categories,
    )


@pytest.mark.benchmark
def test_bfcl_dataset_translates_schema_and_attaches_tools(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """BFCLDataset should translate schemas to OpenAI tool format, set
    `messages` directly on SampleRequest, and attach tools/tool_choice via
    request_overrides."""
    paths = _write_fake_files(tmp_path)

    def fake_download(_repo, filename, **_kwargs):
        category = filename.removeprefix("BFCL_v3_").removesuffix(".json")
        return str(paths[category])

    args = _args_for_bfcl(categories=["simple", "live_simple"])

    with _patch_hf_api(fake_download):
        samples = get_samples(args, hf_tokenizer)

    assert len(samples) == 2
    for s in samples:
        assert s.chat_messages is not None
        assert isinstance(s.chat_messages, list)
        assert s.chat_messages[0]["role"] == "user"
        assert s.request_overrides is not None
        assert "tools" in s.request_overrides
        assert s.request_overrides["tool_choice"] == "auto"
        # messages must NOT leak into request_overrides — it has its own
        # typed field on SampleRequest.
        assert "messages" not in s.request_overrides
        tools = s.request_overrides["tools"]
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        # Translated schema: dict -> object, float -> number,
        # any -> string, tuple -> array.
        params = tool["function"]["parameters"]
        assert params["type"] == "object"
        for prop in params["properties"].values():
            assert prop["type"] in {"integer", "number", "string", "array"}


@pytest.mark.benchmark
def test_bfcl_dataset_requires_openai_chat_backend(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    args = _args_for_bfcl(categories=["simple"])
    args.backend = "openai"

    with pytest.raises(ValueError, match="openai-chat"):
        get_samples(args, hf_tokenizer)


@pytest.mark.benchmark
def test_bfcl_dataset_missing_category_raises_clear_error(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """A typo'd category should produce an actionable ValueError, not an
    opaque huggingface_hub exception."""
    from huggingface_hub.errors import EntryNotFoundError

    args = _args_for_bfcl(categories=["simpl"])  # typo

    def raise_missing(_repo, filename, **_kwargs):
        raise EntryNotFoundError(f"404 Not Found: {filename}")

    with (
        _patch_hf_api(raise_missing),
        pytest.raises(ValueError, match=r"BFCL category 'simpl' not found"),
    ):
        get_samples(args, hf_tokenizer)


@pytest.mark.benchmark
def test_chat_backend_uses_messages_field_when_set() -> None:
    """When RequestFuncInput.chat_messages is set, the chat backend must use
    it verbatim and skip default content construction from `prompt`."""
    import asyncio

    from vllm.benchmarks.lib.endpoint_request_func import (
        RequestFuncInput,
        async_request_openai_chat_completions,
    )

    captured: dict = {}

    class _FakeResp:
        status = 500
        reason = "stop-after-capture"
        content = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _FakeSession:
        def post(self, url, json, headers):  # noqa: A002
            captured["url"] = url
            captured["payload"] = json
            return _FakeResp()

    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "call add(3, 4)"},
    ]
    req = RequestFuncInput(
        prompt="IGNORED",
        api_url="http://localhost:0/v1/chat/completions",
        prompt_len=10,
        output_len=16,
        model="test-model",
        chat_messages=messages,
        extra_body={"tools": [{"type": "function", "function": {"name": "add"}}]},
    )

    asyncio.run(
        async_request_openai_chat_completions(
            request_func_input=req, session=_FakeSession()
        )
    )

    payload = captured["payload"]
    assert payload["messages"] is messages, (
        "chat backend must forward RequestFuncInput.chat_messages verbatim "
        "instead of constructing a default user message from `prompt`"
    )
    # extra_body still merges in as before (shallow, per-request wins).
    assert payload["tools"][0]["function"]["name"] == "add"


@pytest.mark.benchmark
def test_bfcl_prompt_len_includes_tools(tmp_path: Path) -> None:
    """prompt_len must reflect tokens from both messages *and* tool schemas,
    so percentile buckets and input-distribution summaries aren't biased
    low for tool-heavy traffic."""
    paths = _write_fake_files(tmp_path)

    def fake_download(_repo, filename, **_kwargs):
        category = filename.removeprefix("BFCL_v3_").removesuffix(".json")
        return str(paths[category])

    captured: dict = {}

    class _FakeTokenizer:
        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=True
        ):
            captured["tools"] = tools
            base = " ".join(m.get("content", "") for m in messages)
            tool_text = json.dumps(tools) if tools else ""
            return base + " " + tool_text

        def __call__(self, text):
            # 1 "token" per whitespace-separated word.
            return type("Enc", (), {"input_ids": text.split()})()

    fake = _FakeTokenizer()
    args = _args_for_bfcl(categories=["simple"])
    args.num_prompts = 1

    with _patch_hf_api(fake_download):
        samples = get_samples(args, fake)

    assert len(samples) == 1
    assert captured["tools"] is not None, (
        "apply_chat_template must be called with tools= so the schema "
        "contributes to the prompt-length estimate"
    )
    assert len(captured["tools"]) == 1
    assert captured["tools"][0]["function"]["name"] == "add"

    # Sanity: prompt_len exceeds a messages-only estimate. The fake row's
    # user message is "What is 2+2?" (3 whitespace-separated tokens).
    assert samples[0].prompt_len > 3


@pytest.mark.benchmark
def test_bfcl_prompt_len_falls_back_when_tokenizer_rejects_tools(
    tmp_path: Path,
) -> None:
    """Older tokenizers don't accept tools=; fallback must still produce a
    non-zero prompt_len without crashing."""
    paths = _write_fake_files(tmp_path)

    def fake_download(_repo, filename, **_kwargs):
        category = filename.removeprefix("BFCL_v3_").removesuffix(".json")
        return str(paths[category])

    class _LegacyTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if "tools" in kwargs:
                raise TypeError("unexpected keyword argument 'tools'")
            return " ".join(m.get("content", "") for m in messages)

        def __call__(self, text):
            return type("Enc", (), {"input_ids": text.split()})()

    args = _args_for_bfcl(categories=["simple"])
    args.num_prompts = 1

    with _patch_hf_api(fake_download):
        samples = get_samples(args, _LegacyTokenizer())

    assert len(samples) == 1
    assert samples[0].prompt_len > 0


@pytest.mark.benchmark
def test_bfcl_schema_translation_is_recursive() -> None:
    """_translate_schema must recurse into nested properties."""
    input_schema = {
        "type": "dict",
        "properties": {
            "nested": {
                "type": "dict",
                "properties": {
                    "value": {"type": "float"},
                    "tags": {"type": "tuple", "items": {"type": "any"}},
                },
            }
        },
    }
    out = BFCLDataset._translate_schema(input_schema)
    assert out["type"] == "object"
    assert out["properties"]["nested"]["type"] == "object"
    assert out["properties"]["nested"]["properties"]["value"]["type"] == "number"
    assert out["properties"]["nested"]["properties"]["tags"]["type"] == "array"
    nested_props = out["properties"]["nested"]["properties"]
    assert nested_props["tags"]["items"]["type"] == "string"
