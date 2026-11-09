# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from types import SimpleNamespace
from typing import _get_protocol_attrs  # type: ignore
from unittest.mock import patch

import pytest
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TokenizersBackend,
    tokenization_utils_base,
)

from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.hf import HfTokenizer
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.tokenizers.registry import (
    cached_tokenizer_from_config,
    resolve_tokenizer_args,
    tokenizer_args_from_config,
)


def _get_missing_attrs(obj: object, target: type):
    return [k for k in _get_protocol_attrs(target) if not hasattr(obj, k)]


def _assert_tokenizer_like(tokenizer: object):
    missing_attrs = _get_missing_attrs(tokenizer, TokenizerLike)
    assert not missing_attrs, f"Missing attrs: {missing_attrs}"


def test_tokenizer_like_protocol():
    tokenizer = get_tokenizer("openai-community/gpt2")
    assert isinstance(tokenizer, TokenizersBackend)
    _assert_tokenizer_like(tokenizer)

    tokenizer = get_tokenizer(
        "mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_mode="mistral",
    )
    assert isinstance(tokenizer, MistralTokenizer)
    _assert_tokenizer_like(tokenizer)

    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3", tokenizer_mode="deepseek_v32")
    assert isinstance(tokenizer, HfTokenizer)

    # Verify it's a fast tokenizer (required for FastIncrementalDetokenizer)
    assert isinstance(tokenizer, TokenizersBackend)
    assert "DSV32" in tokenizer.__class__.__name__
    _assert_tokenizer_like(tokenizer)


@pytest.mark.parametrize(
    "tokenizer_name", ["facebook/opt-125m", "openai-community/gpt2"]
)
def test_tokenizer_revision(tokenizer_name: str):
    # Assume that "main" branch always exists
    tokenizer = get_tokenizer(tokenizer_name, revision="main")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # Assume that "never" branch always does not exist
    with pytest.raises(OSError, match="not a valid git identifier"):
        get_tokenizer(tokenizer_name, revision="never")


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "code_revision, error_match",
    [
        pytest.param(
            "never",
            "not a valid git identifier|couldn't find them in the cached files",
            id="unknown-revision",
        ),
        pytest.param(
            "73293ac504e43f4a895688b57c65cebe24f01a38",
            r"tokenization_fast\.py",
            id="revision-without-code",
        ),
    ],
)
def test_tokenizer_code_revision_rejects_missing_code(
    code_revision: str, error_match: str
):
    """A data snapshot must not shadow the requested remote-code revision."""
    with pytest.raises(OSError, match=error_match):
        get_tokenizer(
            "hf-internal-testing/test_dynamic_tokenizer",
            tokenizer_mode="hf",
            revision="08a35780b7a78794436d8654aafa26ee0165c149",
            code_revision=code_revision,
            trust_remote_code=True,
        )


@pytest.mark.skip_global_cleanup
def test_tokenizer_code_revision_distinct_from_revision():
    """Distinct code/data revisions: the resolver must keep the Hub ID.

    Dynamic code is read from a local directory as-is, so resolving to a
    data-revision snapshot would silently shadow an independent code
    revision (the same contract already enforced for processors).
    """
    revision = "08a35780b7a78794436d8654aafa26ee0165c149"
    repo = "hf-internal-testing/test_dynamic_tokenizer"
    _, name, _, kwargs = resolve_tokenizer_args(
        repo,
        tokenizer_mode="hf",
        revision=revision,
        code_revision=revision,
        trust_remote_code=True,
    )
    assert Path(name).is_dir()
    assert kwargs["code_revision"] == revision

    _, name, _, kwargs = resolve_tokenizer_args(
        repo,
        tokenizer_mode="hf",
        revision=revision,
        code_revision="73293ac504e43f4a895688b57c65cebe24f01a38",
        trust_remote_code=True,
    )
    assert name == repo
    assert kwargs["code_revision"] == "73293ac504e43f4a895688b57c65cebe24f01a38"


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "code_revision", [None, "08a35780b7a78794436d8654aafa26ee0165c149"]
)
def test_tokenizer_code_revision_default_or_matching(code_revision: str | None):
    revision = "08a35780b7a78794436d8654aafa26ee0165c149"
    tokenizer = get_tokenizer(
        "hf-internal-testing/test_dynamic_tokenizer",
        tokenizer_mode="hf",
        revision=revision,
        code_revision=code_revision,
        trust_remote_code=True,
    )
    reference = AutoTokenizer.from_pretrained(
        "hf-internal-testing/test_dynamic_tokenizer",
        revision=revision,
        code_revision=code_revision,
        trust_remote_code=True,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert tokenizer.encode("hello world") == reference.encode("hello world")
    assert tokenizer.name_or_path == "hf-internal-testing/test_dynamic_tokenizer"
    _assert_tokenizer_like(tokenizer)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "code_revision",
    [
        None,
        "c8ae8a2e5845fc3ad4e5237e4346e5a9ba1e87d2",
        "2efddce40dddaccd37bae208c3c7ca66dbedf68a",
    ],
)
def test_tokenizer_config_code_revision(code_revision):
    """Load real remote config code at its pin, not the data revision."""
    revision = "c8ae8a2e5845fc3ad4e5237e4346e5a9ba1e87d2"

    with patch(
        "vllm.tokenizers.registry.CachedHfTokenizer.from_pretrained",
        return_value=SimpleNamespace(is_fast=True),
    ) as load_tokenizer:
        get_tokenizer(
            "hf-internal-testing/test_dynamic_model",
            tokenizer_mode="hf",
            trust_remote_code=True,
            revision=revision,
            code_revision=code_revision,
        )

    config = load_tokenizer.call_args.kwargs["config"]
    assert (code_revision or revision) in type(config).__module__.split(".")
    assert config.new_hidden_size == 12


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "loader", (cached_tokenizer_from_config, tokenizer_args_from_config)
)
@pytest.mark.parametrize(
    "model, code_revision, kwargs, error_match",
    [
        pytest.param(
            None,
            "never",
            {},
            "not a valid git identifier|couldn't find them in the cached files",
            id="same-repo",
        ),
        pytest.param(
            None,
            "73293ac504e43f4a895688b57c65cebe24f01a38",
            {},
            r"tokenization_fast\.py",
            id="same-repo-missing-code",
        ),
        pytest.param(
            "facebook/opt-125m",
            "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
            {},
            None,
            id="separate-repo",
        ),
        pytest.param(
            None,
            "never",
            {"code_revision": "08a35780b7a78794436d8654aafa26ee0165c149"},
            None,
            id="explicit-revision",
        ),
        pytest.param(None, "never", {"code_revision": None}, None, id="explicit-none"),
    ],
)
def test_config_tokenizer_code_revision(
    loader, model, code_revision, kwargs, error_match
):
    """Only same-repo tokenizers inherit model code revisions; overrides win."""
    revision = "08a35780b7a78794436d8654aafa26ee0165c149"
    repo = "hf-internal-testing/test_dynamic_tokenizer"
    config = SimpleNamespace(
        model=model or repo,
        tokenizer=repo,
        runner_type="generate",
        tokenizer_mode="hf",
        tokenizer_revision=revision,
        code_revision=code_revision,
        trust_remote_code=True,
        skip_tokenizer_init=False,
    )

    def load():
        result = loader(config, **kwargs)
        if loader is tokenizer_args_from_config:
            mode, name, args, resolved_kwargs = result
            return get_tokenizer(name, *args, tokenizer_mode=mode, **resolved_kwargs)
        return result

    if error_match is not None:
        with pytest.raises(OSError, match=error_match):
            load()
    else:
        tokenizer = load()
        reference = AutoTokenizer.from_pretrained(
            repo,
            revision=revision,
            code_revision=kwargs.get("code_revision"),
            trust_remote_code=True,
        )
        assert tokenizer.encode("hello world") == reference.encode("hello world")
        _assert_tokenizer_like(tokenizer)


@pytest.mark.parametrize(
    "loader", (cached_tokenizer_from_config, tokenizer_args_from_config)
)
@pytest.mark.parametrize("tokenizer_mode", ["mistral", "auto"])
@pytest.mark.parametrize("code_revision", [None, "never"])
def test_config_mistral_code_revision(loader, tokenizer_mode, code_revision):
    """Config loading must not pass HF-only revisions to the Mistral backend."""
    repo = "mistralai/Mistral-7B-Instruct-v0.3"
    config = SimpleNamespace(
        model=repo,
        tokenizer=repo,
        runner_type="generate",
        tokenizer_mode=tokenizer_mode,
        tokenizer_revision=None,
        code_revision=code_revision,
        trust_remote_code=False,
        skip_tokenizer_init=False,
    )
    tokenizer = loader(config)
    if loader is tokenizer_args_from_config:
        mode, name, args, kwargs = tokenizer
        tokenizer = get_tokenizer(name, *args, tokenizer_mode=mode, **kwargs)

    reference = get_tokenizer(repo, tokenizer_mode="mistral")
    assert isinstance(tokenizer, MistralTokenizer)
    assert tokenizer.encode("hello world") == reference.encode("hello world")


def test_hub_tokenizer_loads_from_local_snapshot():
    """Transformers makes a Hub request per file when given a repo ID."""
    with patch.object(
        tokenization_utils_base,
        "list_repo_templates",
        wraps=tokenization_utils_base.list_repo_templates,
    ) as list_repo_templates:
        tokenizer = get_tokenizer("openai-community/gpt2")

    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert tokenizer.name_or_path == "openai-community/gpt2"
    list_repo_templates.assert_not_called()


@pytest.mark.parametrize("tokenizer_name", ["BAAI/bge-base-en"])
@pytest.mark.parametrize("n_tokens", [510])
def test_special_tokens(tokenizer_name: str, n_tokens: int):
    tokenizer = get_tokenizer(tokenizer_name, revision="main")

    prompts = "[UNK]" * n_tokens
    prompt_token_ids = tokenizer.encode(prompts)
    assert len(prompt_token_ids) == n_tokens + 2
