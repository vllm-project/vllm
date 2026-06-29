#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from ascend_targeted_scenarios import load_targeted_scenario_registry

TARGETED_SCENARIOS = load_targeted_scenario_registry()
SUPPORTED_SCENARIOS = TARGETED_SCENARIOS.supported_scenarios
SUPPORTED_GROUPS = TARGETED_SCENARIOS.supported_groups
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ALLOWED_AUTHOR_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
ALLOWED_SHAREGPT_PATH_PREFIXES = ("/data/", "/mnt/data/")
MAX_COMMENT_NUM_PROMPTS = 32
MAX_COMMENT_MAX_CONCURRENCY = 4
MAX_COMMENT_REQUEST_RATE = 4.0
MAX_COMMENT_LOGICAL_LEN = 4096


@dataclass(frozen=True)
class IssueCommentPrContext:
    pr_number: str
    head_sha: str
    base_sha: str
    head_repo: str
    base_repo: str
    repository: str
    is_same_repo: bool

    def to_env(self) -> dict[str, str]:
        return {
            "ISSUE_COMMENT_PR_NUMBER": self.pr_number,
            "ISSUE_COMMENT_PR_HEAD_SHA": self.head_sha,
            "ISSUE_COMMENT_PR_BASE_SHA": self.base_sha,
            "ISSUE_COMMENT_PR_HEAD_REPO": self.head_repo,
            "ISSUE_COMMENT_PR_BASE_REPO": self.base_repo,
            "ISSUE_COMMENT_PR_SAME_REPO": "1" if self.is_same_repo else "0",
        }


@dataclass(frozen=True)
class AscendHelpCommand:
    action: str = "help"

    def to_env(self) -> dict[str, str]:
        return {
            "ASCEND_COMMENT_COMMAND": "0",
            "ASCEND_COMMENT_HELP": "1",
        }


@dataclass(frozen=True)
class AscendCommentCommand:
    action: str
    scenario: str
    model_name: str = DEFAULT_MODEL_NAME
    dataset_path: str = ""
    constraints_file: str = ""
    num_prompts: str = "8"
    request_rate: str = "inf"
    max_concurrency: str = "4"
    input_length: str = ""
    output_length: str = ""
    publish_to_hf: bool = False

    def to_env(self) -> dict[str, str]:
        return {
            "ASCEND_COMMENT_COMMAND": "1",
            "INPUT_BENCHMARK_SCENARIO": self.scenario,
            "MODEL_NAME": self.model_name,
            "BENCH_DATASET_PATH": self.dataset_path,
            "BENCH_CONSTRAINTS_FILE": self.constraints_file,
            "BENCH_NUM_PROMPTS": self.num_prompts,
            "BENCH_REQUEST_RATE": self.request_rate,
            "BENCH_MAX_CONCURRENCY": self.max_concurrency,
            "BENCH_INPUT_LEN": self.input_length,
            "BENCH_OUTPUT_LEN": self.output_length,
            "PUBLISH_TO_HF": "1" if self.publish_to_hf else "0",
            "PUBLISH_TO_BENCHMARK_REPO": "0",
        }


def _command_line(comment_body: str) -> str | None:
    for raw_line in comment_body.splitlines():
        line = raw_line.strip()
        if line == "/ascend" or line.startswith("/ascend "):
            return line
        if line.startswith("/ascend"):
            raise ValueError(f"invalid /ascend command: {line}")
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="/ascend", description="Parse Ascend benchmark PR comment command."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("help")
    smoke = subparsers.add_parser("smoke")
    _add_preview_options(smoke)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("scenario")
    _add_preview_options(benchmark)

    scenario = subparsers.add_parser("scenario")
    scenario.add_argument("scenario")
    _add_preview_options(scenario)

    group = subparsers.add_parser("group")
    group.add_argument("group")
    _add_preview_options(group)

    official = subparsers.add_parser("official")
    official.add_argument("spec_id")
    return parser


def _add_preview_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--constraints-file", default="")
    parser.add_argument("--num-prompts", default="8")
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", default="4")
    parser.add_argument("--input-length", default="")
    parser.add_argument("--output-length", default="")


def _parse_bounded_int(value: str, name: str, minimum: int, maximum: int) -> str:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return str(parsed)


def _parse_request_rate(value: str) -> str:
    if value == "inf":
        return value
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError("--request-rate must be 'inf' or a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0 or parsed > MAX_COMMENT_REQUEST_RATE:
        raise ValueError(
            "--request-rate must be 'inf' or between 0 and "
            f"{MAX_COMMENT_REQUEST_RATE:g}"
        )
    return str(parsed).rstrip("0").rstrip(".") if "." in str(parsed) else str(parsed)


def _validate_sharegpt_path(path: str, name: str) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    path_text = resolved.as_posix()
    if not path_text.endswith((".json", ".jsonl")):
        raise ValueError(f"{name} must end with .json or .jsonl")
    if not path_text.startswith(ALLOWED_SHAREGPT_PATH_PREFIXES):
        prefixes = ", ".join(ALLOWED_SHAREGPT_PATH_PREFIXES)
        raise ValueError(f"{name} must be under one of: {prefixes}")
    if ".." in resolved.parts:
        raise ValueError(f"{name} must not contain '..'")
    return path_text


def _validate_command_args(args: argparse.Namespace, scenario: str) -> dict[str, str]:
    validated = {
        "num_prompts": _parse_bounded_int(
            args.num_prompts, "--num-prompts", 1, MAX_COMMENT_NUM_PROMPTS
        ),
        "request_rate": _parse_request_rate(args.request_rate),
        "max_concurrency": _parse_bounded_int(
            args.max_concurrency,
            "--max-concurrency",
            1,
            MAX_COMMENT_MAX_CONCURRENCY,
        ),
        "input_length": "",
        "output_length": "",
        "dataset_path": "",
        "constraints_file": "",
    }
    if args.input_length:
        validated["input_length"] = _parse_bounded_int(
            args.input_length, "--input-length", 1, MAX_COMMENT_LOGICAL_LEN
        )
    if args.output_length:
        validated["output_length"] = _parse_bounded_int(
            args.output_length,
            "--output-length",
            1,
            MAX_COMMENT_LOGICAL_LEN,
        )
    if scenario == "sharegpt-online":
        validated["dataset_path"] = _validate_sharegpt_path(
            args.dataset_path, "--dataset-path"
        )
        validated["constraints_file"] = _validate_sharegpt_path(
            args.constraints_file, "--constraints-file"
        )
    elif args.dataset_path or args.constraints_file:
        raise ValueError(
            "--dataset-path and --constraints-file are only supported "
            "for sharegpt-online"
        )
    return validated


def parse_comment_command(comment_body: str) -> AscendCommentCommand | None:
    line = _command_line(comment_body)
    if line is None:
        return None

    tokens = shlex.split(line)
    parser = _build_parser()
    try:
        args = parser.parse_args(tokens[1:])
    except SystemExit as exc:
        raise ValueError(f"invalid /ascend command: {line}") from exc

    if args.action == "help":
        return AscendHelpCommand()

    if args.action == "official":
        raise ValueError(
            "/ascend official is reserved for future formal leaderboard runs "
            "and is not supported yet"
        )

    if args.action == "smoke":
        scenario = TARGETED_SCENARIOS.resolve_group("smoke")
    elif args.action == "group":
        scenario = TARGETED_SCENARIOS.resolve_group(args.group)
        if scenario is None:
            supported = ", ".join(SUPPORTED_GROUPS)
            raise ValueError(
                f"unsupported benchmark group {args.group!r}; supported: {supported}"
            )
    else:
        scenario = TARGETED_SCENARIOS.resolve_scenario(args.scenario)
    if scenario is None:
        supported = ", ".join(SUPPORTED_SCENARIOS)
        raise ValueError(
            f"unsupported benchmark scenario {args.scenario!r}; supported: {supported}"
        )

    missing = []
    if scenario == "sharegpt-online":
        if not args.dataset_path:
            missing.append("--dataset-path")
        if not args.constraints_file:
            missing.append("--constraints-file")
    if missing:
        raise ValueError("sharegpt-online requires " + ", ".join(missing))

    validated = _validate_command_args(args, scenario)

    return AscendCommentCommand(
        action=args.action,
        scenario=scenario,
        model_name=DEFAULT_MODEL_NAME,
        dataset_path=validated["dataset_path"],
        constraints_file=validated["constraints_file"],
        num_prompts=validated["num_prompts"],
        request_rate=validated["request_rate"],
        max_concurrency=validated["max_concurrency"],
        input_length=validated["input_length"],
        output_length=validated["output_length"],
        publish_to_hf=False,
    )


def _nested_value(payload: dict, *keys: str) -> str:
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return str(value or "")


def _has_allowed_commenter(event_payload: dict, pr_payload: dict) -> bool:
    author_association = _nested_value(event_payload, "comment", "author_association")
    if author_association in ALLOWED_AUTHOR_ASSOCIATIONS:
        return True
    commenter = _nested_value(event_payload, "comment", "user", "login")
    pr_author = _nested_value(pr_payload, "user", "login")
    return bool(commenter and pr_author and commenter == pr_author)


def resolve_issue_comment_pr_context(
    event_payload: dict,
    pr_payload: dict,
) -> IssueCommentPrContext:
    issue = event_payload.get("issue") if isinstance(event_payload, dict) else None
    if not isinstance(issue, dict) or not issue.get("pull_request"):
        raise ValueError("issue_comment event is not a pull request comment")

    repository = _nested_value(event_payload, "repository", "full_name")
    pr_number = str(issue.get("number") or pr_payload.get("number") or "")
    head_sha = _nested_value(pr_payload, "head", "sha")
    base_sha = _nested_value(pr_payload, "base", "sha")
    head_repo = _nested_value(pr_payload, "head", "repo", "full_name")
    base_repo = _nested_value(pr_payload, "base", "repo", "full_name")
    missing = [
        name
        for name, value in {
            "repository.full_name": repository,
            "issue.number": pr_number,
            "pull_request.head.sha": head_sha,
            "pull_request.base.sha": base_sha,
            "pull_request.head.repo.full_name": head_repo,
            "pull_request.base.repo.full_name": base_repo,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError("missing issue_comment PR metadata: " + ", ".join(missing))

    is_same_repo = head_repo == repository and base_repo == repository
    if not is_same_repo:
        raise ValueError(
            "fork PR issue_comment benchmark is not allowed: "
            f"head={head_repo}, base={base_repo}, repository={repository}"
        )
    if not _has_allowed_commenter(event_payload, pr_payload):
        raise ValueError(
            "issue_comment benchmark requires the PR author or a repository "
            "collaborator"
        )

    return IssueCommentPrContext(
        pr_number=pr_number,
        head_sha=head_sha,
        base_sha=base_sha,
        head_repo=head_repo,
        base_repo=base_repo,
        repository=repository,
        is_same_repo=is_same_repo,
    )


def _load_json_file(path: str) -> dict:
    if not path:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _github_multiline_delimiter(key: str, value: str) -> str:
    delimiter = f"__{key}_EOF__"
    while delimiter in value:
        delimiter = f"_{delimiter}_"
    return delimiter


def write_env_file(path: str, values: dict[str, str]) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            delimiter = _github_multiline_delimiter(key, value)
            handle.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")


def _deny_values(reason: str) -> dict[str, str]:
    return {
        "ASCEND_COMMENT_COMMAND": "0",
        "ASCEND_COMMENT_DENY_REASON": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse an Ascend benchmark PR comment command."
    )
    parser.add_argument("--comment-body", default=os.environ.get("COMMENT_BODY", ""))
    parser.add_argument(
        "--event-payload", default=os.environ.get("GITHUB_EVENT_PATH", "")
    )
    parser.add_argument(
        "--pr-payload", default=os.environ.get("ISSUE_COMMENT_PR_PAYLOAD", "")
    )
    parser.add_argument("--github-env", default=os.environ.get("GITHUB_ENV", ""))
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    args = parser.parse_args()

    try:
        command = parse_comment_command(args.comment_body)
        values = (
            command.to_env() if command is not None else {"ASCEND_COMMENT_COMMAND": "0"}
        )
        if isinstance(command, AscendHelpCommand):
            write_env_file(args.github_env, values)
            write_env_file(args.github_output, values)
            print("Parsed Ascend comment command: help")
            return 0
        if command is not None and args.event_payload:
            context = resolve_issue_comment_pr_context(
                _load_json_file(args.event_payload),
                _load_json_file(args.pr_payload),
            )
            values.update(context.to_env())
    except ValueError as exc:
        reason = str(exc)
        print(f"Ascend comment command denied: {reason}", file=sys.stderr)
        values = _deny_values(reason)
        write_env_file(args.github_env, values)
        write_env_file(args.github_output, values)
        return 0

    if command is None:
        print("No Ascend comment command found.")
    elif isinstance(command, AscendHelpCommand):
        print("Parsed Ascend comment command: help")
    else:
        print(f"Parsed Ascend comment command: {command.action} {command.scenario}")

    write_env_file(args.github_env, values)
    write_env_file(args.github_output, values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
