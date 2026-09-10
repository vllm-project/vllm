# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.check_area_test_coverage import (  # noqa: E402
    check_coverage,
    find_pytest_invocations,
    parse_invocation,
)

FIXTURES = Path(__file__).parent / "fixtures"
PRE_SHARDING_YAML = FIXTURES / "models_language_pre_sharding.yaml"
POST_SHARDING_YAML = FIXTURES / "area_post_sharding.yaml"
CPU_LANE_YAML = FIXTURES / "cpu_lane_language.yaml"


def _make_tree(root: Path, *files: str) -> Path:
    """Create a synthetic area tree; returns the --tree path."""
    tree = root / "tests/models/language"
    for rel in files:
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_x():\n    pass\n")
    return tree


def _write_yaml(root: Path, commands: list[str]) -> Path:
    yaml_path = root / "area.yaml"
    steps = "\n".join(
        f"- label: job{i}\n  key: job{i}\n  commands:\n    - {cmd!r}"
        for i, cmd in enumerate(commands)
    )
    yaml_path.write_text(f"steps:\n{steps}\n")
    return yaml_path


def test_unclaimed_file_reported_with_nearest_job(tmp_path):
    tree = _make_tree(
        tmp_path,
        "generation/core/test_common.py",
        "generation/rogue/test_orphan.py",
    )
    yaml_path = _write_yaml(tmp_path, ["pytest -v -s models/language/generation/core"])

    violations, _ = check_coverage(tree, [yaml_path])

    assert len(violations) == 1
    assert "generation/rogue/test_orphan.py" in violations[0]
    assert "job0" in violations[0]  # the nearest job that would need to claim it


def test_partition_filters_detected(tmp_path):
    tree = _make_tree(tmp_path, "generation/hybrid/01/test_a.py")
    yaml_path = _write_yaml(
        tmp_path,
        [
            # -k expression and shard flags (env-var style)
            "pytest -v -s models/language/generation "
            "-k 'not granite-4.0-tiny-preview' "
            "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT "
            "--shard-id=$$BUILDKITE_PARALLEL_JOB",
            # per-file target inside the tree
            "pytest -v -s models/language/generation/hybrid/01/test_a.py",
            # compound -m partition (splits by test content, not hardware)
            "pytest -v -s models/language -m 'core_model and slow_test'",
            # single marker that is not a lane-capability marker
            "pytest -v -s models/language/generation -m hybrid_model",
        ],
    )

    violations, _ = check_coverage(tree, [yaml_path])

    text = "\n".join(violations)
    assert "-k expression" in text
    assert "pytest-shard flags" in text
    assert "per-file target" in text
    assert "partitioning -m expression: 'core_model and slow_test'" in text
    assert "partitioning -m expression: 'hybrid_model'" in text


def test_cpu_recursive_command_accepted_as_lane_filter(tmp_path):
    tree = _make_tree(
        tmp_path,
        "generation/hybrid/01/test_a.py",
        "generation/hybrid_granite/test_g.py",
        "generation/core/test_c.py",
        "pooling/test_p.py",
    )

    violations, notes = check_coverage(tree, [POST_SHARDING_YAML, CPU_LANE_YAML])

    assert violations == []
    assert any("lane filter -m cpu_model" in note for note in notes)


def test_pre_sharding_language_yaml_fails(tmp_path):
    # The pre-change language YAML partitions directories with -k, compound
    # -m expressions and shard flags: the guard must detect them.
    tree = _make_tree(tmp_path, "generation/core/test_common.py")

    violations, _ = check_coverage(tree, [PRE_SHARDING_YAML])

    text = "\n".join(violations)
    assert "-k expression: 'not granite-4.0-tiny-preview'" in text
    assert "-k expression: 'granite-4.0-tiny-preview'" in text
    assert "partitioning -m expression: 'core_model and slow_test'" in text
    assert "pytest-shard flags" in text


def test_post_sharding_language_yaml_passes(tmp_path):
    tree = _make_tree(
        tmp_path,
        "generation/hybrid/01/test_a.py",
        "generation/hybrid_granite/test_g.py",
        "generation/core/test_c.py",
        "pooling/test_p.py",
    )

    violations, _ = check_coverage(tree, [POST_SHARDING_YAML, CPU_LANE_YAML])

    assert violations == []


def test_out_of_tree_commands_ignored(tmp_path):
    tree = _make_tree(tmp_path, "generation/core/test_c.py")
    yaml_path = _write_yaml(tmp_path, ["pytest -v -s models/language/generation/core"])
    # cpu_kernel command targets files outside the language tree entirely.
    violations, _ = check_coverage(tree, [yaml_path, CPU_LANE_YAML])

    assert violations == []


def test_find_pytest_invocations_skips_comments_and_wrappers():
    cmds = find_pytest_invocations(
        "pip freeze | grep -E 'torch'\n"
        "# pytest -v -s models/language  (commented out)\n"
        'bash run.sh 25m "\n'
        'pytest -x -v -s tests/models/language/generation -m cpu_model"\n'
    )
    assert cmds == ['pytest -x -v -s tests/models/language/generation -m cpu_model"']


def test_parse_invocation_handles_env_var_shard_flags_and_quotes():
    cmd = parse_invocation(
        "pytest -v -s models/language/generation -m hybrid_model "
        "-k 'not granite-4.0-tiny-preview' "
        "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT "
        '--shard-id=$$BUILDKITE_PARALLEL_JOB"',
        lane="base",
        step="job",
        tree_rel="models/language",
    )
    assert cmd.targets == ["models/language/generation"]
    assert cmd.m_expr == "hybrid_model"
    assert cmd.k_expr == "not granite-4.0-tiny-preview"
    assert cmd.sharded
    assert not cmd.is_whole_dir
