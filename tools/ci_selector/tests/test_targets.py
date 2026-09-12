# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Command-shape contract, and what a step really invokes, vs. the real checkout."""

import pytest
from ci_selector.codemap.pipeline.buildkite import load_pipeline_configs, load_steps
from ci_selector.codemap.pipeline.invoked_tests import invoked_files
from ci_selector.codemap.pipeline.scripts import scan_script
from ci_selector.codemap.pipeline.step import LoadReport
from ci_selector.codemap.pipeline.targets import map_step

# Aliased: pytest collects any module-level name starting with `test_`,
# and would report the import itself as a broken test.
from ci_selector.codemap.repo import test_file_catalog as _test_file_catalog
from helpers import drift_message


@pytest.fixture(scope="module")
def mapped(vllm_repo):
    report = LoadReport()
    all_steps = []
    for c in load_pipeline_configs(vllm_repo):
        all_steps.extend(load_steps(vllm_repo, c, report))
    targets = [map_step(vllm_repo, s, script_scanner=scan_script) for s in all_steps]
    invoked = invoked_files(_test_file_catalog(vllm_repo), targets)
    return all_steps, invoked, {t.step_id: t for t in targets}


@pytest.mark.drift
def test_unparsable_empty_at_head(mapped):
    """BENIGN absorbs setup noise; anything left means a command shape
    the parser does not understand."""
    _, _, by_id = mapped
    leftovers = {sid: t.unparsable for sid, t in by_id.items() if t.unparsable}
    assert leftovers == {}, drift_message(
        "These step commands use a shape the target parser does not "
        f"understand: {leftovers}",
        "We cannot tell which tests those commands run, so the step is selected "
        "by weaker signals than it should be.",
        "it is setup noise that runs no test: add the command to BENIGN_CMDS "
        "in ci_selector/handwritten.py",
        "it wraps another command (like `uv run pytest`): add it to UNWRAP_CMDS "
        "in ci_selector/handwritten.py",
        "it is a genuinely new way of invoking tests: teach "
        "ci_selector/codemap/pipeline/targets.py to read it",
    )


def test_dangling_is_bounded_and_explained(mapped):
    _, _, by_id = mapped
    dangling = {d for t in by_id.values() for d in t.dangling}
    # run-npu-test.sh references a path that only exists in the external
    # vllm-ascend CI checkout; anything else is parser skew or YAML rot.
    assert dangling <= {"tests/e2e/vllm_interface/"}, dangling


def test_lora_shard_ignores_recorded_not_applied(mapped):
    """--ignore narrows the shard but never shrinks the target set."""
    _, _, by_id = mapped
    st = by_id["vllm_ci:lora"]
    assert any(t.path == "tests/lora" for t in st.targets)
    assert "tests/lora/test_llama_tp.py" in st.ignored


def test_find_xargs_shape(mapped):
    """pytorch.yaml `find compile/ ... | xargs pytest {}`."""
    _, _, by_id = mapped
    finds = [
        t
        for st in by_id.values()
        for t in st.targets
        if st.step_id.startswith("vllm_ci:")
        and t.path.startswith("tests/compile")
        and any(n.startswith("find-exclude:") for n in t.narrowing)
    ]
    assert finds, "expected find|xargs-derived tests/compile targets"


def test_wrapped_block_extraction(mapped):
    """run-cpu-test.sh receives its pytest block as an argv string."""
    _, _, by_id = mapped
    cpu_kernel = next(
        st
        for sid, st in by_id.items()
        if st.step_id.startswith("vllm_ci:")
        and any(t.path == "tests/kernels/test_onednn.py" for t in st.targets)
    )
    assert any("run-cpu-test.sh" in s for s in cpu_kernel.scripts_seen)


def test_script_recursion_reaches_nixl_tests(mapped):
    """Disaggregated jobs invoke tests/**/*.sh whose pytest lines carry
    ${GIT_ROOT}-prefixed paths."""
    _, _, by_id = mapped
    nixl = "tests/v1/kv_connector/nixl_integration/test_accuracy.py"
    assert any(t.path == nixl for st in by_id.values() for t in st.targets)


def test_config_list_files_become_data_edges(mapped):
    """lm_eval coverage is the .txt list, resolved against the test
    file's directory."""
    _, _, by_id = mapped
    data = {d for st in by_id.values() for d in st.data_files}
    assert any(
        d.startswith("tests/evals/") and d.endswith("models-small.txt") for d in data
    ), sorted(data)


def test_pipe_then_and_chain_not_lost(mapped):
    """`torchrun x | grep ok && pytest y` keeps y (multi-node block shape)."""
    _, invoked, _ = mapped
    assert "tests/distributed/test_multi_node_assignment.py" in invoked
    assert "tests/distributed/test_node_count.py" in invoked


def _wrap_step(commands):
    from ci_selector.codemap.pipeline.step import Step

    return Step(
        pipeline="t",
        source_file="x.yaml",
        label="wrap",
        key="wrap",
        group=None,
        commands=commands,
        source_file_dependencies=None,
    )


def _map(vllm_repo, command):
    from ci_selector.codemap.pipeline.targets import map_step

    return map_step(vllm_repo, _wrap_step([command]))


def test_uv_run_wrapper_unwraps(vllm_repo):
    st = _map(vllm_repo, "uv run pytest -v lora/test_llama_tp.py")
    assert [t.path for t in st.targets] == ["tests/lora/test_llama_tp.py"]
    assert not st.unparsable


def test_uv_non_run_subcommand_not_unwrapped(vllm_repo):
    """uv is also a benign command; only `run` unwraps, so a missing `run`
    must not phantom-target the argv."""
    st = _map(vllm_repo, "uv pytest lora/test_llama_tp.py")
    assert not st.targets and not st.unparsable


def test_sudo_wrapped_pytest(vllm_repo):
    st = _map(vllm_repo, "sudo -E pytest lora/test_llama_tp.py")
    assert [t.path for t in st.targets] == ["tests/lora/test_llama_tp.py"]
    assert not st.unparsable


def test_docker_wrapped_test_is_unparsable(vllm_repo):
    st = _map(vllm_repo, "docker exec ci pytest tests/lora/test_llama_tp.py")
    assert st.unparsable and not st.targets


def test_zero_match_glob_is_dangling(vllm_repo):
    """A glob matching nothing is the same stale hole as a rename; recording
    neither a target nor a dangling hid it from the preflight escalation."""
    st = _map(vllm_repo, "pytest -v lora/test_no_such_thing_*.py")
    assert not st.targets
    assert st.dangling == ["lora/test_no_such_thing_*.py"]


def test_matching_glob_still_targets(vllm_repo):
    st = _map(vllm_repo, "pytest -v lora/test_llama_*.py")
    assert "tests/lora/test_llama_tp.py" in [t.path for t in st.targets]
    assert not st.dangling


def test_foreign_absolute_path_is_dangling(vllm_repo):
    """A container path outside the workspace root used to resolve to '' (the
    vllm_repo root), which matched nothing downstream while silencing both the
    dangling escalation and the zero-target warning."""
    st = _map(vllm_repo, "pytest -v /workspace/tests/e2e/singlecard/test_offline.py")
    assert not st.targets
    assert st.dangling


def test_uv_global_flags_before_run_unwrap(vllm_repo):
    for cmd in (
        "uv -q run pytest lora/test_llama_tp.py",
        "uv --directory /vllm-workspace run pytest lora/test_llama_tp.py",
    ):
        st = _map(vllm_repo, cmd)
        assert [t.path for t in st.targets] == ["tests/lora/test_llama_tp.py"], cmd
        assert not st.unparsable


def test_docker_compose_wrapped_test_unparsable(vllm_repo):
    for cmd in (
        "docker compose run svc pytest tests/lora/test_llama_tp.py",
        "docker-compose exec svc pytest tests/lora/test_llama_tp.py",
        "docker --context prod run img pytest tests/lora/test_llama_tp.py",
    ):
        st = _map(vllm_repo, cmd)
        assert st.unparsable and not st.targets, cmd
