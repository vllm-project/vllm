# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from vllm.entrypoints.cli import local as local_cli
from vllm.entrypoints.cli import local_backends
from vllm.entrypoints.cli import local_runtime
from vllm.entrypoints.cli import main as main_cli
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


def _fake_platform(*, memory_bytes: int = 32 * 1024**3):
    class FakePlatform:
        device_type = "cpu"
        device_name = "CPU"
        supported_dtypes: list[object] = []
        supported_quantization: list[str] = []

        def is_cuda(self) -> bool:
            return False

        def is_rocm(self) -> bool:
            return False

        def is_xpu(self) -> bool:
            return False

        def is_cpu(self) -> bool:
            return True

        def is_out_of_tree(self) -> bool:
            return False

        def get_device_total_memory(self) -> int:
            return memory_bytes

        def support_static_graph_mode(self) -> bool:
            return False

    return FakePlatform()


def _fake_apple_platform(*, memory_bytes: int = 32 * 1024**3):
    class FakeApplePlatform:
        device_type = "metal"
        device_name = "Apple Metal"
        supported_dtypes: list[object] = []
        supported_quantization: list[str] = []

        def is_cuda(self) -> bool:
            return False

        def is_rocm(self) -> bool:
            return False

        def is_xpu(self) -> bool:
            return False

        def is_cpu(self) -> bool:
            return False

        def is_out_of_tree(self) -> bool:
            return True

        def get_device_total_memory(self) -> int:
            return memory_bytes

        def support_static_graph_mode(self) -> bool:
            return False

    return FakeApplePlatform()


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


def test_aliases_command_registered():
    commands = {command.name for command in local_cli.cmd_init()}
    assert "aliases" in commands
    assert "doctor" in commands
    assert "status" in commands
    assert "preflight" in commands


def test_local_command_specs_match_registered_commands():
    advertised = {
        spec.name
        for spec in main_cli.COMMAND_SPECS
        if spec.module == "vllm.entrypoints.cli.local"
    }
    registered = {command.name for command in local_cli.cmd_init()}
    assert advertised <= registered


def test_parse_aliases_command():
    parser = FlexibleArgumentParser(description="test")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    local_cli.AliasesCommand().subparser_init(subparsers)

    args = parser.parse_args(["aliases", "--json"])
    assert args.subparser == "aliases"
    assert args.json is True


def test_background_serve_honors_port_equals_form(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.resolve_model_reference",
        lambda model, revision=None: local_runtime.ResolvedModel(
            requested=model,
            model="repo/model",
            source="alias",
        ),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.ensure_model_available",
        lambda resolved, download_dir=None: {"local_path": str(tmp_path / "model")},
    )
    monkeypatch.setattr("vllm.entrypoints.cli.serve.find_service", lambda _: None)
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.build_doctor_report",
        lambda **_: SimpleNamespace(
            selected_backend="cpu",
            fallback_reason=None,
            selection_reason="cpu fallback",
            preflight=None,
        ),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.allocate_service_port",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(AssertionError("unexpected random port allocation")),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.spawn_service_process",
        lambda command, log_path: captured.update(
            {"command": command, "log_path": str(log_path)}
        )
        or SimpleNamespace(pid=4321),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.register_service",
        lambda record: captured.setdefault("record", record),
    )

    args = argparse.Namespace(
        model="deepseek-r1:8b",
        model_tag=None,
        revision=None,
        download_dir=None,
        foreground=False,
        service_name=None,
        no_wait=True,
        host="127.0.0.1",
        port=8000,
        profile="balanced",
        backend="auto",
        dtype="auto",
        quantization=None,
        max_model_len=None,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=None,
        enforce_eager=False,
        _argv=["serve", "deepseek-r1:8b", "--port=8000"],
    )

    ServeSubcommand.cmd(args)

    assert args.port == 8000
    assert "--port=8000" in captured["command"]
    assert captured["record"]["port"] == 8000


def test_background_serve_honors_port_split_form(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.resolve_model_reference",
        lambda model, revision=None: local_runtime.ResolvedModel(
            requested=model,
            model="repo/model",
            source="alias",
        ),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.ensure_model_available",
        lambda resolved, download_dir=None: {"local_path": str(tmp_path / "model")},
    )
    monkeypatch.setattr("vllm.entrypoints.cli.serve.find_service", lambda _: None)
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.build_doctor_report",
        lambda **_: SimpleNamespace(
            selected_backend="cpu",
            fallback_reason=None,
            selection_reason="cpu fallback",
            preflight=None,
        ),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.allocate_service_port",
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(AssertionError("unexpected random port allocation")),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.spawn_service_process",
        lambda command, log_path: captured.update(
            {"command": command, "log_path": str(log_path)}
        )
        or SimpleNamespace(pid=4321),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve.register_service",
        lambda record: captured.setdefault("record", record),
    )

    args = argparse.Namespace(
        model="deepseek-r1:8b",
        model_tag=None,
        revision=None,
        download_dir=None,
        foreground=False,
        service_name=None,
        no_wait=True,
        host="127.0.0.1",
        port=8001,
        profile="balanced",
        backend="auto",
        dtype="auto",
        quantization=None,
        max_model_len=None,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=None,
        enforce_eager=False,
        _argv=["serve", "deepseek-r1:8b", "--port", "8001"],
    )

    ServeSubcommand.cmd(args)

    assert args.port == 8001
    assert (
        captured["command"][-3:] == ["--port", "8001", "--foreground"]
        or captured["command"][-3:] == ["--foreground", "--port", "8001"]
    )
    assert captured["record"]["port"] == 8001


def test_select_backend_prefers_apple_plugin(monkeypatch):
    monkeypatch.setattr(local_backends.py_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local_backends.py_platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        local_backends,
        "_discover_platform_plugins",
        lambda: [{"name": "vllm-metal", "value": "vllm_metal.platform:plugin"}],
    )
    monkeypatch.setattr(local_backends, "_safe_import_torch", lambda: None)
    monkeypatch.setattr(
        local_backends,
        "_safe_import_current_platform",
        lambda: _fake_apple_platform(memory_bytes=64 * 1024**3),
    )

    selection, _backends, _env, _platform = local_backends.select_backend()
    assert selection.selected_backend == "apple-metal"


def test_select_backend_falls_back_to_cpu_without_apple_plugin(monkeypatch):
    monkeypatch.setattr(local_backends.py_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local_backends.py_platform, "machine", lambda: "arm64")
    monkeypatch.setattr(local_backends, "_discover_platform_plugins", lambda: [])
    monkeypatch.setattr(local_backends, "_safe_import_torch", lambda: None)
    monkeypatch.setattr(
        local_backends,
        "_safe_import_current_platform",
        lambda: _fake_platform(memory_bytes=32 * 1024**3),
    )

    selection, _backends, _env, _platform = local_backends.select_backend()
    assert selection.selected_backend == "cpu"
    assert "apple-metal" in selection.rejected_backends


def test_doctor_report_includes_preflight(monkeypatch):
    monkeypatch.setattr(local_backends.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(local_backends.py_platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(local_backends, "_discover_platform_plugins", lambda: [])
    monkeypatch.setattr(local_backends, "_safe_import_torch", lambda: None)
    monkeypatch.setattr(
        local_backends,
        "_safe_import_current_platform",
        lambda: _fake_platform(memory_bytes=64 * 1024**3),
    )
    monkeypatch.setattr(
        local_backends,
        "_safe_import_psutil",
        lambda: SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(available=64 * 1024**3)
        ),
    )

    report = local_backends.build_doctor_report(
        model="deepseek-r1:8b",
        requested_backend="auto",
    )
    assert report.preflight is not None
    assert report.preflight.estimated_total_bytes is not None
    assert report.trtllm is not None


def test_install_script_requires_manual_uv(tmp_path):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"
    python_dir = Path(sys.executable).resolve().parent
    env = {
        "HOME": str(tmp_path),
        "PATH": f"{python_dir}:/usr/bin:/bin:/usr/sbin:/sbin",
    }
    result = subprocess.run(
        ["bash", str(script_path), "--user", "--python", sys.executable],
        cwd=script_path.parents[1],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "uv is required" in combined_output
    assert "docs.astral.sh/uv/getting-started/installation" in combined_output
