# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The grammar must guard every token the engine stops on.

Kimi K3's tokenizer eos is ``[EOS]`` while the model ends a turn with
``<|end_of_msg|>`` (``generation_config.json``). Guarding only the tokenizer's
id lets the real terminator be matched as ordinary text, so a grammar that is
not satisfied yet cannot hold generation back.
"""

from types import SimpleNamespace

from vllm.v1.structured_output.backend_xgrammar import engine_stop_token_ids


def _config(hf_eos, generation_eos, raises=False):
    def try_get_generation_config():
        if raises:
            raise RuntimeError("no generation config")
        return {} if generation_eos is None else {"eos_token_id": generation_eos}

    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(eos_token_id=hf_eos),
        hf_config=SimpleNamespace(eos_token_id=hf_eos),
        try_get_generation_config=try_get_generation_config,
    )
    return SimpleNamespace(model_config=model_config)


def test_union_of_tokenizer_and_model_eos():
    # K3 shape: tokenizer says [EOS]=163585, the model ends turns with 163586.
    ids = engine_stop_token_ids(
        _config(163586, 163586), SimpleNamespace(eos_token_id=163585)
    )
    assert ids == [163585, 163586]


def test_generation_config_eos_list_is_included():
    ids = engine_stop_token_ids(_config(None, [7, 9]), SimpleNamespace(eos_token_id=1))
    assert ids == [1, 7, 9]


def test_single_eos_is_unchanged():
    ids = engine_stop_token_ids(_config(2, 2), SimpleNamespace(eos_token_id=2))
    assert ids == [2]


def test_unreadable_generation_config_is_not_fatal():
    ids = engine_stop_token_ids(
        _config(5, None, raises=True), SimpleNamespace(eos_token_id=4)
    )
    assert ids == [4, 5]
