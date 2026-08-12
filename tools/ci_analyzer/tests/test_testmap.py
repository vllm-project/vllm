# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Command-shape contract + uninvoked derivation vs. the real checkout."""

import pytest
from ci_analyzer.jobs.buildkite import load_pipeline_configs, load_steps
from ci_analyzer.jobs.model import LoadReport
from ci_analyzer.validate.uninvoked import uninvoked_report


@pytest.fixture(scope="module")
def mapped(repo):
    report = LoadReport()
    all_steps = []
    for c in load_pipeline_configs(repo):
        all_steps.extend(load_steps(repo, c, report))
    ur, targets = uninvoked_report(repo, all_steps)
    return all_steps, ur, {t.step_id: t for t in targets}


def test_unparsable_empty_at_head(mapped):
    """BENIGN absorbs setup noise; anything left means a command shape
    the parser does not understand."""
    _, _, by_id = mapped
    leftovers = {sid: t.unparsable for sid, t in by_id.items() if t.unparsable}
    assert leftovers == {}


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
    _, ur, by_id = mapped
    nixl = "tests/v1/kv_connector/nixl_integration/test_accuracy.py"
    assert nixl not in ur.orphans
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
    _, ur, _ = mapped
    assert "tests/distributed/test_multi_node_assignment.py" not in ur.orphans
    assert "tests/distributed/test_node_count.py" not in ur.orphans


def test_uninvoked_orphans_derived(mapped):
    """A test invoked only via the orphaned legacy test-amd.yaml is classified
    legacy-only, never an orphan. Both derived from HEAD, no filenames pinned."""
    _, ur, _ = mapped
    assert ur.orphans, "expected orphan test files at HEAD"
    assert ur.legacy_only, "expected legacy-test-amd-only test files at HEAD"
    assert not (set(ur.orphans) & set(ur.legacy_only))


def _wrap_step(commands):
    from ci_analyzer.jobs.model import Step

    return Step(
        pipeline="t",
        source_file="x.yaml",
        label="wrap",
        key="wrap",
        group=None,
        commands=commands,
        source_file_dependencies=None,
    )


def _map(repo, command):
    from ci_analyzer.jobs.testmap import map_step

    return map_step(repo, _wrap_step([command]))


def test_uv_run_wrapper_unwraps(repo):
    st = _map(repo, "uv run pytest -v lora/test_llama_tp.py")
    assert [t.path for t in st.targets] == ["tests/lora/test_llama_tp.py"]
    assert not st.unparsable


def test_uv_non_run_subcommand_not_unwrapped(repo):
    """uv is also a benign command; only `run` unwraps, so a missing `run`
    must not phantom-target the argv."""
    st = _map(repo, "uv pytest lora/test_llama_tp.py")
    assert not st.targets and not st.unparsable


def test_sudo_wrapped_pytest(repo):
    st = _map(repo, "sudo -E pytest lora/test_llama_tp.py")
    assert [t.path for t in st.targets] == ["tests/lora/test_llama_tp.py"]
    assert not st.unparsable


def test_docker_wrapped_test_is_unparsable(repo):
    st = _map(repo, "docker exec ci pytest tests/lora/test_llama_tp.py")
    assert st.unparsable and not st.targets


def test_zero_match_glob_is_dangling(repo):
    """A glob matching nothing is the same stale hole as a rename; recording
    neither a target nor a dangling hid it from the preflight escalation."""
    st = _map(repo, "pytest -v lora/test_no_such_thing_*.py")
    assert not st.targets
    assert st.dangling == ["lora/test_no_such_thing_*.py"]


def test_matching_glob_still_targets(repo):
    st = _map(repo, "pytest -v lora/test_llama_*.py")
    assert "tests/lora/test_llama_tp.py" in [t.path for t in st.targets]
    assert not st.dangling


def test_foreign_absolute_path_is_dangling(repo):
    """A container path outside the workspace root used to resolve to '' (the
    repo root), which matched nothing downstream while silencing both the
    dangling escalation and the zero-target warning."""
    st = _map(repo, "pytest -v /workspace/tests/e2e/singlecard/test_offline.py")
    assert not st.targets
    assert st.dangling


def test_uv_global_flags_before_run_unwrap(repo):
    for cmd in (
        "uv -q run pytest lora/test_llama_tp.py",
        "uv --directory /vllm-workspace run pytest lora/test_llama_tp.py",
    ):
        st = _map(repo, cmd)
        assert [t.path for t in st.targets] == ["tests/lora/test_llama_tp.py"], cmd
        assert not st.unparsable


def test_docker_compose_wrapped_test_unparsable(repo):
    for cmd in (
        "docker compose run svc pytest tests/lora/test_llama_tp.py",
        "docker-compose exec svc pytest tests/lora/test_llama_tp.py",
        "docker --context prod run img pytest tests/lora/test_llama_tp.py",
    ):
        st = _map(repo, cmd)
        assert st.unparsable and not st.targets, cmd
