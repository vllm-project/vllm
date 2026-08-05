# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level padding defaults must reach frontends that build their own
`TokenizeParams`, such as the OpenAI-compatible pooling endpoints."""

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TextPrompt
from vllm.renderers import TokenizeParams
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers import get_tokenizer

MODEL = "google/siglip-base-patch16-224"
TRAINED_LENGTH = 64


@pytest.fixture(scope="module")
def renderer() -> HfRenderer:
    vllm_config = EngineArgs(
        model=MODEL,
        runner="pooling",
        dtype="float32",
        max_model_len=TRAINED_LENGTH,
        enforce_eager=True,
    ).create_engine_config()

    return HfRenderer(vllm_config, get_tokenizer(MODEL))


def _request_built_params() -> TokenizeParams:
    """What the pooling endpoints build from a request: no padding field."""
    return TokenizeParams(
        max_total_tokens=TRAINED_LENGTH,
        max_output_tokens=0,
        add_special_tokens=True,
    )


def test_model_default_is_applied(renderer: HfRenderer):
    assert renderer.default_cmpl_tok_params.pad_prompt_tokens == -1

    prompt = TextPrompt(prompt="a photo of a stop sign")
    tokenized = renderer._tokenize_singleton_prompt(prompt, _request_built_params())

    assert len(tokenized["prompt_token_ids"]) == TRAINED_LENGTH


def test_explicit_padding_is_not_overridden(renderer: HfRenderer):
    params = TokenizeParams(
        max_total_tokens=TRAINED_LENGTH,
        max_output_tokens=0,
        pad_prompt_tokens=16,
    )
    prompt = TextPrompt(prompt="a photo of a stop sign")

    tokenized = renderer._tokenize_singleton_prompt(prompt, params)

    assert len(tokenized["prompt_token_ids"]) == 16


def test_multimodal_prompts_are_not_padded(renderer: HfRenderer):
    """Image prompts carry placeholder text that processing replaces."""
    prompt = TextPrompt(prompt="", multi_modal_data={"image": []})

    tokenized = renderer._tokenize_singleton_prompt(prompt, _request_built_params())

    assert len(tokenized["prompt_token_ids"]) < TRAINED_LENGTH


def _declare_model_inputs(
    renderer: HfRenderer, monkeypatch: pytest.MonkeyPatch, names: list[str] | None
):
    info = renderer.get_mm_processor().info
    init_kwargs = dict(info.get_tokenizer().init_kwargs)

    if names is None:
        init_kwargs.pop("model_input_names", None)
    else:
        init_kwargs["model_input_names"] = names

    monkeypatch.setattr(info.get_tokenizer(), "init_kwargs", init_kwargs)
    return info


def test_checkpoint_consuming_attention_mask_opts_out(
    renderer: HfRenderer, monkeypatch: pytest.MonkeyPatch
):
    """A checkpoint that declares an attention mask is left unpadded."""
    info = _declare_model_inputs(renderer, monkeypatch, ["input_ids", "attention_mask"])

    assert info.get_default_tok_params().pad_prompt_tokens is None


def test_undeclared_model_inputs_still_pad(
    renderer: HfRenderer, monkeypatch: pytest.MonkeyPatch
):
    """`SiglipTokenizer.model_input_names` defaults to containing
    `attention_mask`, so a checkpoint that declares nothing must still pad."""
    info = _declare_model_inputs(renderer, monkeypatch, None)

    assert info.get_default_tok_params().pad_prompt_tokens == -1
