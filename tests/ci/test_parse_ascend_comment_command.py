# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/parse_ascend_comment_command.py"
)


def load_parser():
    spec = importlib.util.spec_from_file_location(
        "parse_ascend_comment_command",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_help_command_does_not_request_benchmark():
    parser = load_parser()

    command = parser.parse_comment_command("/ascend help")

    assert command is not None
    assert command.action == "help"
    assert command.to_env()["ASCEND_COMMENT_COMMAND"] == "0"
    assert command.to_env()["ASCEND_COMMENT_HELP"] == "1"


def test_parse_random_command_with_reasonable_defaults():
    parser = load_parser()

    command = parser.parse_comment_command("/ascend benchmark random")

    assert command is not None
    assert command.action == "benchmark"
    assert command.scenario == "random-online"
    assert command.publish_to_hf is False
    assert command.to_env()["PUBLISH_TO_HF"] == "0"
    assert command.to_env()["PUBLISH_TO_BENCHMARK_REPO"] == "0"
    assert command.num_prompts == "8"
    assert command.request_rate == "inf"
    assert command.max_concurrency == "4"
    assert command.to_env()["PUBLISH_TO_BENCHMARK_REPO"] == "0"


def test_parse_smoke_command_maps_to_random_preview():
    parser = load_parser()

    command = parser.parse_comment_command("/ascend smoke --num-prompts 4")

    assert command is not None
    assert command.action == "smoke"
    assert command.scenario == "random-online"
    assert command.num_prompts == "4"
    assert command.publish_to_hf is False
    assert command.to_env()["PUBLISH_TO_HF"] == "0"
    assert command.to_env()["PUBLISH_TO_BENCHMARK_REPO"] == "0"


def test_parse_scenario_command_maps_to_supported_scenario():
    parser = load_parser()

    command = parser.parse_comment_command("/ascend scenario random --request-rate 2")

    assert command is not None
    assert command.action == "scenario"
    assert command.scenario == "random-online"
    assert command.request_rate == "2"


def test_parse_group_command_maps_to_supported_group():
    parser = load_parser()

    command = parser.parse_comment_command("/ascend group smoke")

    assert command is not None
    assert command.action == "group"
    assert command.scenario == "random-online"


def test_parse_sharegpt_command_with_explicit_options():
    parser = load_parser()

    command = parser.parse_comment_command(
        "/ascend benchmark sharegpt --dataset-path /data/sharegpt.jsonl "
        "--constraints-file /mnt/data/constraints.json --num-prompts 32 "
        "--request-rate 2 --max-concurrency 4"
    )

    assert command is not None
    assert command.scenario == "sharegpt-online"
    assert command.dataset_path == "/data/sharegpt.jsonl"
    assert command.constraints_file == "/mnt/data/constraints.json"
    assert command.num_prompts == "32"
    assert command.request_rate == "2"
    assert command.max_concurrency == "4"
    assert command.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert command.publish_to_hf is False


def test_parse_scenario_sharegpt_command_with_explicit_options():
    parser = load_parser()

    command = parser.parse_comment_command(
        "/ascend scenario sharegpt --dataset-path /data/sharegpt.jsonl "
        "--constraints-file /mnt/data/constraints.json --num-prompts 16"
    )

    assert command is not None
    assert command.action == "scenario"
    assert command.scenario == "sharegpt-online"
    assert command.dataset_path == "/data/sharegpt.jsonl"
    assert command.constraints_file == "/mnt/data/constraints.json"
    assert command.num_prompts == "16"


def test_parse_rejects_comment_model_override():
    parser = load_parser()

    try:
        parser.parse_comment_command("/ascend benchmark random --model Qwen/test")
    except ValueError as exc:
        assert "invalid /ascend command" in str(exc)
    else:
        raise AssertionError("expected comment-triggered model override to be rejected")


def test_parse_rejects_resource_limits_above_preview_bounds():
    parser = load_parser()

    for body, expected in [
        ("/ascend benchmark random --num-prompts 33", "--num-prompts"),
        ("/ascend benchmark random --request-rate 4.1", "--request-rate"),
        ("/ascend benchmark random --request-rate nan", "--request-rate"),
        ("/ascend benchmark random --request-rate NaN", "--request-rate"),
        ("/ascend benchmark random --request-rate Infinity", "--request-rate"),
        ("/ascend benchmark random --request-rate -inf", "--request-rate"),
        ("/ascend benchmark random --max-concurrency 5", "--max-concurrency"),
        ("/ascend benchmark random --input-length 4097", "--input-length"),
        ("/ascend benchmark random --output-length 4097", "--output-length"),
    ]:
        try:
            parser.parse_comment_command(body)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(
                f"expected oversized preview parameter to be rejected: {body}"
            )


def test_parse_rejects_sharegpt_paths_outside_allowlist():
    parser = load_parser()

    for body, expected in [
        (
            "/ascend benchmark sharegpt --dataset-path relative.jsonl "
            "--constraints-file /data/constraints.json",
            "absolute path",
        ),
        (
            "/ascend benchmark sharegpt --dataset-path /tmp/sharegpt.jsonl "
            "--constraints-file /data/constraints.json",
            "under one of",
        ),
        (
            "/ascend benchmark sharegpt --dataset-path /data/sharegpt.txt "
            "--constraints-file /data/constraints.json",
            ".json or .jsonl",
        ),
    ]:
        try:
            parser.parse_comment_command(body)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(
                f"expected unsafe sharegpt path to be rejected: {body}"
            )


def test_parse_rejects_comment_publish_flag():
    parser = load_parser()

    try:
        parser.parse_comment_command("/ascend benchmark random --publish-to-hf")
    except ValueError as exc:
        assert "publish" in str(exc)
    else:
        raise AssertionError("expected comment-triggered publish flag to be rejected")


def test_parse_ignores_non_command_comments():
    parser = load_parser()

    assert parser.parse_comment_command("please run ascend benchmark") is None
    assert parser.parse_comment_command("/test all") is None


def test_parse_accepts_command_from_multiline_comment():
    parser = load_parser()

    command = parser.parse_comment_command(
        "LGTM\n/ascend benchmark random --num-prompts 16\nthanks"
    )

    assert command is not None
    assert command.scenario == "random-online"
    assert command.num_prompts == "16"


def test_parse_rejects_unsupported_scenario():
    parser = load_parser()

    try:
        parser.parse_comment_command("/ascend benchmark moe")
    except ValueError as exc:
        assert "unsupported benchmark scenario" in str(exc)
    else:
        raise AssertionError("expected unsupported scenario to be rejected")


def test_parse_rejects_unsupported_protocol_targets():
    parser = load_parser()

    for body, expected in [
        ("/ascend group moe", "unsupported benchmark group"),
        ("/ascend scenario qwen25", "unsupported benchmark scenario"),
        (
            "/ascend official official-ascend-qwen25",
            "reserved for future formal leaderboard runs",
        ),
    ]:
        try:
            parser.parse_comment_command(body)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(
                f"expected unsupported protocol target to be rejected: {body}"
            )


def test_parse_rejects_sharegpt_without_dataset_and_constraints():
    parser = load_parser()

    try:
        parser.parse_comment_command("/ascend benchmark sharegpt")
    except ValueError as exc:
        message = str(exc)
        assert "sharegpt-online requires" in message
        assert "--dataset-path" in message
        assert "--constraints-file" in message
    else:
        raise AssertionError("expected incomplete sharegpt command to be rejected")


def test_resolve_issue_comment_pr_context_accepts_same_repo_pr():
    parser = load_parser()
    payload = {
        "issue": {
            "number": 42,
            "pull_request": {
                "url": "https://api.github.test/repos/o/r/pulls/42",
            },
        },
        "comment": {
            "body": "/ascend benchmark random",
            "author_association": "CONTRIBUTOR",
            "user": {"login": "alice"},
        },
        "repository": {"full_name": "o/r"},
    }
    pr_payload = {
        "number": 42,
        "user": {"login": "alice"},
        "head": {"sha": "head-sha", "repo": {"full_name": "o/r"}},
        "base": {"sha": "base-sha", "repo": {"full_name": "o/r"}},
    }

    context = parser.resolve_issue_comment_pr_context(payload, pr_payload)

    assert context.pr_number == "42"
    assert context.head_sha == "head-sha"
    assert context.base_sha == "base-sha"
    assert context.is_same_repo is True
    assert context.to_env()["ISSUE_COMMENT_PR_HEAD_SHA"] == "head-sha"


def test_resolve_issue_comment_pr_context_accepts_collaborator_commenter():
    parser = load_parser()
    payload = {
        "issue": {
            "number": 42,
            "pull_request": {
                "url": "https://api.github.test/repos/o/r/pulls/42",
            },
        },
        "comment": {
            "body": "/ascend benchmark random",
            "author_association": "COLLABORATOR",
            "user": {"login": "maintainer"},
        },
        "repository": {"full_name": "o/r"},
    }
    pr_payload = {
        "number": 42,
        "user": {"login": "alice"},
        "head": {"sha": "head-sha", "repo": {"full_name": "o/r"}},
        "base": {"sha": "base-sha", "repo": {"full_name": "o/r"}},
    }

    context = parser.resolve_issue_comment_pr_context(payload, pr_payload)

    assert context.is_same_repo is True


def test_resolve_issue_comment_pr_context_rejects_untrusted_commenter():
    parser = load_parser()
    payload = {
        "issue": {
            "number": 42,
            "pull_request": {
                "url": "https://api.github.test/repos/o/r/pulls/42",
            },
        },
        "comment": {
            "body": "/ascend benchmark random",
            "author_association": "CONTRIBUTOR",
            "user": {"login": "bob"},
        },
        "repository": {"full_name": "o/r"},
    }
    pr_payload = {
        "number": 42,
        "user": {"login": "alice"},
        "head": {"sha": "head-sha", "repo": {"full_name": "o/r"}},
        "base": {"sha": "base-sha", "repo": {"full_name": "o/r"}},
    }

    try:
        parser.resolve_issue_comment_pr_context(payload, pr_payload)
    except ValueError as exc:
        assert "PR author or a repository collaborator" in str(exc)
    else:
        raise AssertionError("expected untrusted commenter to be rejected")


def test_resolve_issue_comment_pr_context_rejects_non_pr_issue():
    parser = load_parser()
    payload = {
        "issue": {"number": 42},
        "comment": {"body": "/ascend benchmark random"},
        "repository": {"full_name": "o/r"},
    }

    try:
        parser.resolve_issue_comment_pr_context(payload, {})
    except ValueError as exc:
        assert "not a pull request" in str(exc)
    else:
        raise AssertionError("expected non-PR issue comment to be rejected")


def test_resolve_issue_comment_pr_context_rejects_fork_pr():
    parser = load_parser()
    payload = {
        "issue": {
            "number": 42,
            "pull_request": {
                "url": "https://api.github.test/repos/o/r/pulls/42",
            },
        },
        "comment": {
            "body": "/ascend benchmark random",
            "author_association": "COLLABORATOR",
            "user": {"login": "maintainer"},
        },
        "repository": {"full_name": "o/r"},
    }
    pr_payload = {
        "number": 42,
        "user": {"login": "alice"},
        "head": {"sha": "head-sha", "repo": {"full_name": "external/r"}},
        "base": {"sha": "base-sha", "repo": {"full_name": "o/r"}},
    }

    try:
        parser.resolve_issue_comment_pr_context(payload, pr_payload)
    except ValueError as exc:
        assert "fork PR" in str(exc)
    else:
        raise AssertionError("expected fork PR issue comment to be rejected")


def test_cli_writes_help_output_without_benchmark(tmp_path):
    output_path = tmp_path / "output.env"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--comment-body",
            "/ascend help",
            "--github-output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    output = output_path.read_text(encoding="utf-8")
    assert "ASCEND_COMMENT_COMMAND<<__ASCEND_COMMENT_COMMAND_EOF__\n0\n" in output
    assert "ASCEND_COMMENT_HELP<<__ASCEND_COMMENT_HELP_EOF__\n1\n" in output


def test_cli_writes_help_output_without_pr_context(tmp_path):
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "output.env"
    event_path.write_text(
        json.dumps(
            {
                "issue": {"number": 42},
                "comment": {"body": "/ascend help"},
                "repository": {"full_name": "o/r"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--comment-body",
            "/ascend help",
            "--event-payload",
            str(event_path),
            "--github-output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    output = output_path.read_text(encoding="utf-8")
    assert "ASCEND_COMMENT_COMMAND<<__ASCEND_COMMENT_COMMAND_EOF__\n0\n" in output
    assert "ASCEND_COMMENT_HELP<<__ASCEND_COMMENT_HELP_EOF__\n1\n" in output
    assert "ASCEND_COMMENT_DENY_REASON" not in output


def test_cli_writes_deny_reason_for_malformed_ascend_prefix(tmp_path):
    output_path = tmp_path / "output.env"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--comment-body",
            "/ascendfoo",
            "--github-output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    output = output_path.read_text(encoding="utf-8")
    assert "ASCEND_COMMENT_COMMAND<<__ASCEND_COMMENT_COMMAND_EOF__\n0\n" in output
    assert "ASCEND_COMMENT_DENY_REASON" in output
    assert "invalid /ascend command" in output


def test_cli_writes_deny_reason_for_fork_pr_comment(tmp_path):
    event_path = tmp_path / "event.json"
    pr_path = tmp_path / "pr.json"
    output_path = tmp_path / "output.env"
    event_path.write_text(
        json.dumps(
            {
                "issue": {
                    "number": 42,
                    "pull_request": {
                        "url": "https://api.github.test/repos/o/r/pulls/42",
                    },
                },
                "comment": {
                    "body": "/ascend benchmark random",
                    "author_association": "COLLABORATOR",
                    "user": {"login": "maintainer"},
                },
                "repository": {"full_name": "o/r"},
            }
        ),
        encoding="utf-8",
    )
    pr_path.write_text(
        json.dumps(
            {
                "number": 42,
                "user": {"login": "alice"},
                "head": {"sha": "head-sha", "repo": {"full_name": "external/r"}},
                "base": {"sha": "base-sha", "repo": {"full_name": "o/r"}},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--comment-body",
            "/ascend benchmark random",
            "--event-payload",
            str(event_path),
            "--pr-payload",
            str(pr_path),
            "--github-output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    output = output_path.read_text(encoding="utf-8")
    assert "ASCEND_COMMENT_COMMAND<<__ASCEND_COMMENT_COMMAND_EOF__\n0\n" in output
    assert "ASCEND_COMMENT_DENY_REASON" in output
    assert "fork PR" in output


def test_cli_writes_deny_reason_for_invalid_command(tmp_path):
    output_path = tmp_path / "output.env"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--comment-body",
            "/ascend benchmark moe",
            "--github-output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    output = output_path.read_text(encoding="utf-8")
    assert "ASCEND_COMMENT_COMMAND<<__ASCEND_COMMENT_COMMAND_EOF__\n0\n" in output
    assert "ASCEND_COMMENT_DENY_REASON" in output
    assert "unsupported benchmark scenario" in output


def test_write_env_file_uses_github_multiline_format(tmp_path):
    parser = load_parser()
    env_file = tmp_path / "github-env"
    command = parser.parse_comment_command("/ascend benchmark random --num-prompts 16")
    assert command is not None

    parser.write_env_file(str(env_file), command.to_env())

    content = env_file.read_text(encoding="utf-8")
    assert "ASCEND_COMMENT_COMMAND<<__ASCEND_COMMENT_COMMAND_EOF__\n1\n" in content
    assert (
        "INPUT_BENCHMARK_SCENARIO<<__INPUT_BENCHMARK_SCENARIO_EOF__\nrandom-online\n"
    ) in content
    assert "BENCH_NUM_PROMPTS<<__BENCH_NUM_PROMPTS_EOF__\n16\n" in content
