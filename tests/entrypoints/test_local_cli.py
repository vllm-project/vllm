# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from vllm.entrypoints.cli import local as local_cli
from vllm.entrypoints.cli import local_runtime
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _set_runtime_paths(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(local_runtime, "LOCAL_RUNTIME_DIR", root)
    monkeypatch.setattr(local_runtime, "LOCAL_MODELS_REGISTRY", root / "models.json")
    monkeypatch.setattr(
        local_runtime,
        "LOCAL_SERVICES_REGISTRY",
        root / "services.json",
    )
    monkeypatch.setattr(local_runtime, "LOCAL_USER_ALIASES", root / "aliases.json")
    monkeypatch.setattr(local_runtime, "LOCAL_LOG_DIR", root / "logs")


def test_resolve_builtin_alias():
    resolved = local_runtime.resolve_model_reference("deepseek-r1:8b")
    assert resolved.alias == "deepseek-r1:8b"
    assert resolved.model == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    assert resolved.source == "alias"


def test_resolve_exact_hf_repo():
    resolved = local_runtime.resolve_model_reference("meta-llama/Llama-3.1-8B-Instruct")
    assert resolved.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert resolved.source == "hf"


def test_record_and_remove_pulled_model(tmp_path, monkeypatch):
    _set_runtime_paths(monkeypatch, tmp_path)

    resolved = local_runtime.resolve_model_reference("deepseek-r1:8b")
    fake_snapshot = tmp_path / "snapshot"
    fake_snapshot.mkdir()

    record = local_runtime.record_pulled_model(resolved, str(fake_snapshot))
    assert record["local_path"] == str(fake_snapshot)

    registry = local_runtime.load_models_registry()
    assert len(registry["models"]) == 1

    removed, purged_path = local_runtime.remove_pulled_model(
        resolved,
        purge_cache=True,
    )
    assert removed is True
    assert purged_path == str(fake_snapshot)
    assert not fake_snapshot.exists()
    assert local_runtime.load_models_registry()["models"] == []


def test_parse_run_command():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    run_parser = local_cli.RunCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "run",
            "deepseek-r1:8b",
            "--prompt",
            "hello",
            "--complete",
            "--max-tokens",
            "32",
        ]
    )
    assert args.subparser == "run"
    assert args.model == "deepseek-r1:8b"
    assert args.prompt == "hello"
    assert args.complete is True
    assert args.max_tokens == 32
    assert isinstance(run_parser, FlexibleArgumentParser)


def test_parse_serve_local_service_flags():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    ServeSubcommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "deepseek-r1:8b",
            "--foreground",
            "--service-name",
            "deepseek",
            "--no-wait",
        ]
    )
    assert args.subparser == "serve"
    assert args.model_tag == "deepseek-r1:8b"
    assert args.foreground is True
    assert args.service_name == "deepseek"
    assert args.no_wait is True


def test_parse_list_alias_command():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    local_cli.ListAliasCommand().subparser_init(subparsers)

    args = parser.parse_args(["list"])
    assert args.subparser == "list"
