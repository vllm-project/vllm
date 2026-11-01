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
