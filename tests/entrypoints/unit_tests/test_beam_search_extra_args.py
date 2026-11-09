# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam search must forward `extra_args` to its per-step `SamplingParams`.

A custom logits processor reads its per-request arguments from
`SamplingParams.extra_args`. That works under ordinary sampling. Under beam search it
did not: `BeamSearchParams` had no `extra_args` field, and the per-step and per-beam
`SamplingParams` built in `beam_search/{offline,online}.py` did not carry one -- so a
processor was silently unreachable whenever beam search was in play. Nothing failed; the
processor simply never saw its arguments.

The end-to-end regression test is
`tests/v1/logits_processors/test_custom_offline.py::test_custom_logitsprocs_beam_search`,
which asserts through `LLM.beam_search` that a processor actually receives the
arguments. The checks here are the cheap complement: the field exists, and no
`SamplingParams` construction on the beam path forgets the keyword. The defect was an
omitted keyword argument, and a new construction site that forgets it is exactly what
the source guard catches and an end-to-end test on the other sites would not.
"""

import ast
import pathlib

import pytest

from vllm.sampling_params import BeamSearchParams

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BEAM_DIR = REPO_ROOT / "vllm" / "entrypoints" / "generate" / "beam_search"


def test_beam_search_params_accepts_and_stores_extra_args():
    """Without the field, a caller cannot pass processor arguments at all."""
    params = BeamSearchParams(
        beam_width=2, max_tokens=4, extra_args={"target_token": 128}
    )
    assert params.extra_args == {"target_token": 128}


def test_beam_search_params_extra_args_defaults_to_none():
    """The field is additive: callers who do not set it are unaffected."""
    assert BeamSearchParams(beam_width=2, max_tokens=4).extra_args is None


@pytest.mark.parametrize("module_name", ["offline.py", "online.py"])
def test_every_sampling_params_in_the_beam_path_forwards_extra_args(module_name):
    """Each construction, not just one.

    The offline path builds two -- a base per step and a per-beam one from the grammar
    -- and forwarding only the base leaves structured-outputs beam search silently
    dropping the arguments.
    """
    tree = ast.parse((BEAM_DIR / module_name).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SamplingParams"
    ]
    assert calls, f"no SamplingParams construction found in {module_name}"

    missing = [
        node.lineno
        for node in calls
        if "extra_args" not in {kw.arg for kw in node.keywords if kw.arg}
    ]
    assert not missing, (
        f"{module_name}: SamplingParams built without extra_args at line(s) "
        f"{missing}; a custom logits processor cannot see its per-request "
        "arguments on this path"
    )


# ---------------------------------------------------------------------------
# Server path. The checks above cover the beam-search internals; they passed
# while the REST path was still broken, because `to_beam_search_params` never
# populated `extra_args` in the first place -- the consumption site was tested,
# the production site was not.
# ---------------------------------------------------------------------------

ENTRYPOINTS = REPO_ROOT / "vllm" / "entrypoints"

PROTOCOLS = [
    "openai/chat_completion/protocol.py",
    "openai/completion/protocol.py",
    "speech_to_text/transcription/protocol.py",
    "speech_to_text/translation/protocol.py",
]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_every_beam_search_params_construction_forwards_extra_args(protocol):
    """Each `to_beam_search_params`, not just the chat one.

    All four request protocols build a `BeamSearchParams`, and one that omits
    `extra_args` leaves `params.extra_args is None` for every REST caller on
    that endpoint -- the per-step `SamplingParams` then carries nothing and the
    processor is as unreachable as it was before this change.
    """
    tree = ast.parse((ENTRYPOINTS / protocol).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "BeamSearchParams"
    ]
    assert calls, f"no BeamSearchParams construction found in {protocol}"

    missing = [
        node.lineno
        for node in calls
        if "extra_args" not in {kw.arg for kw in node.keywords if kw.arg}
    ]
    assert not missing, (
        f"{protocol}: BeamSearchParams built without extra_args at line(s) "
        f"{missing}; a REST caller using beam search cannot reach a custom "
        "logits processor on this endpoint"
    )


def test_chat_request_forwards_vllm_xargs_to_beam_search():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        use_beam_search=True,
        n=2,
        vllm_xargs={"target_token": 128},
    )
    assert req.to_beam_search_params(4, {}).extra_args == {"target_token": 128}


def test_completion_request_forwards_vllm_xargs_to_beam_search():
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    req = CompletionRequest(
        prompt="hi", use_beam_search=True, n=2, vllm_xargs={"target_token": 128}
    )
    assert req.to_beam_search_params(4, {}).extra_args == {"target_token": 128}


@pytest.mark.parametrize("field", ["kv_transfer_params", "ec_transfer_params"])
def test_beam_search_also_forwards_transfer_params(field):
    """`extra_args` is the transport for these too, not just `vllm_xargs`.

    Forwarding only `vllm_xargs` would still drop them under beam search while
    ordinary sampling kept them.
    """
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    req = CompletionRequest(prompt="hi", use_beam_search=True, n=2, **{field: {"a": 1}})
    extra_args = req.to_beam_search_params(4, {}).extra_args
    assert extra_args is not None
    assert extra_args[field] == {"a": 1}


def test_beam_search_extra_args_matches_sampling_path():
    """Parity is the real contract: same request, same `extra_args`.

    Asserted against the ordinary path rather than a literal so the two cannot
    drift apart again.
    """
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    req = CompletionRequest(
        prompt="hi",
        n=2,
        vllm_xargs={"target_token": 128},
        kv_transfer_params={"a": 1},
    )
    beam = req.to_beam_search_params(4, {}).extra_args
    sampling = req.to_sampling_params(4, {}).extra_args
    assert beam == sampling


def test_empty_vllm_xargs_is_not_collapsed_to_none():
    """`{}` means "configured with nothing", which is not "not configured".

    `copy_extra_args` preserves that distinction downstream, so the protocol
    must not destroy it upstream.
    """
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    req = CompletionRequest(prompt="hi", n=2, vllm_xargs={})
    assert req.to_beam_search_params(4, {}).extra_args == {}
