# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import get_samples
from vllm.benchmarks.datasets.datasets import _sharegpt_turns


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("openai-community/gpt2")


def _conv(*pairs: tuple[str, str]) -> list[dict]:
    return [{"from": role, "value": value} for role, value in pairs]


def _write_sharegpt(path: Path, n_convs: int, turns_per_conv: int) -> None:
    data = [
        {
            "id": f"c{c}",
            "conversations": _conv(
                *[
                    ("human" if t % 2 == 0 else "gpt", f"c{c} message {t} padding text")
                    for t in range(turns_per_conv)
                ]
            ),
        }
        for c in range(n_convs)
    ]
    path.write_text(json.dumps(data))


def _args(
    dataset_path: str,
    multi_turn: bool,
    max_turns: int | None = None,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
):
    return argparse.Namespace(
        dataset_name="sharegpt",
        dataset_path=dataset_path,
        disable_shuffle=True,
        num_prompts=50,
        sharegpt_output_len=8,
        enable_multimodal_chat=False,
        no_oversample=True,
        seed=0,
        request_id_prefix="",
        include_multi_turn=multi_turn,
        sharegpt_max_turns=max_turns,
        sharegpt_max_prompt_len=max_prompt_len,
        sharegpt_max_total_len=max_total_len,
    )


def test_default_is_single_turn():
    """Without the flag, only the first exchange is used."""
    c = _conv(("human", "u1"), ("gpt", "a1"), ("human", "u2"), ("gpt", "a2"))
    assert list(_sharegpt_turns(c, max_turns=1)) == [("u1", "a1")]


def test_prompts_accumulate_and_form_a_prefix_chain():
    c = _conv(
        ("human", "u1"), ("gpt", "a1"),
        ("human", "u2"), ("gpt", "a2"),
        ("human", "u3"), ("gpt", "a3"),
    )
    turns = list(_sharegpt_turns(c))
    assert turns == [
        ("u1", "a1"),
        ("u1a1u2", "a2"),
        ("u1a1u2a2u3", "a3"),
    ]
    prompts = [p for p, _ in turns]
    for earlier, later in zip(prompts, prompts[1:]):
        assert later.startswith(earlier) and len(later) > len(earlier)


@pytest.mark.parametrize("assistant", ["gpt", "chatgpt", "bing", "bard"])
def test_assistant_role_variants(assistant: str):
    c = _conv(("human", "u1"), (assistant, "a1"), ("human", "u2"), (assistant, "a2"))
    assert list(_sharegpt_turns(c)) == [("u1", "a1"), ("u1a1u2", "a2")]


def test_malformed_conversations_lose_no_content():
    """System turns and repeated roles join the history; a dangling turn is dropped."""
    system = _conv(("system", "S"), ("human", "u1"), ("gpt", "a1"), ("human", "u2"),
                   ("gpt", "a2"))
    assert list(_sharegpt_turns(system)) == [("Su1", "a1"), ("Su1a1u2", "a2")]

    doubled = _conv(("human", "u1a"), ("human", "u1b"), ("gpt", "a1"))
    assert list(_sharegpt_turns(doubled)) == [("u1au1b", "a1")]

    dangling = _conv(("human", "u1"), ("gpt", "a1"), ("human", "u2"))
    assert list(_sharegpt_turns(dangling)) == [("u1", "a1")]

    assert list(_sharegpt_turns(_conv(("system", "s")))) == []


@pytest.mark.benchmark
def test_include_multi_turn_yields_more_requests(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """End to end through get_samples: 5 conversations of 6 turns each."""
    path = tmp_path / "sharegpt.json"
    _write_sharegpt(path, n_convs=5, turns_per_conv=6)

    single = get_samples(_args(str(path), multi_turn=False), hf_tokenizer)
    multi = get_samples(_args(str(path), multi_turn=True), hf_tokenizer)
    capped = get_samples(_args(str(path), multi_turn=True, max_turns=2), hf_tokenizer)

    assert len(single) == 5           # one request per conversation
    assert len(multi) == 15           # three assistant replies per conversation
    assert len(capped) == 10          # capped at two per conversation

    # Requests from one conversation arrive in order, each prompt extending the last.
    per_conv = [multi[i:i + 3] for i in range(0, 15, 3)]
    for group in per_conv:
        prompts = [s.prompt for s in group]
        for earlier, later in zip(prompts, prompts[1:]):
            assert later.startswith(earlier)
        assert group[0].prompt_len < group[-1].prompt_len


@pytest.mark.benchmark
def test_prompt_len_limits_are_configurable(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """The prompt-length limits gate which multi-turn requests survive.

    Multi-turn prompts grow with conversation depth, so a low limit drops the tail of
    every conversation. Long conversations are used so that later prompts clear the
    limit under test.
    """
    path = tmp_path / "sharegpt.json"
    _write_sharegpt(path, n_convs=4, turns_per_conv=20)

    generous = get_samples(
        _args(str(path), multi_turn=True, max_prompt_len=1024, max_total_len=2048),
        hf_tokenizer,
    )
    strict = get_samples(
        _args(str(path), multi_turn=True, max_prompt_len=32, max_total_len=64),
        hf_tokenizer,
    )

    assert len(strict) < len(generous)
    assert all(s.prompt_len <= 32 for s in strict)
    assert any(s.prompt_len > 32 for s in generous)
