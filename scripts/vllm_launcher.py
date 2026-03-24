#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

ALIASES_RELATIVE_PATH = Path("vllm") / "entrypoints" / "cli" / "model_aliases.py"


def _config_root() -> Path:
    if "VLLM_CONFIG_ROOT" in os.environ:
        return Path(os.environ["VLLM_CONFIG_ROOT"]).expanduser()
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return xdg / "vllm"


LOCAL_RUNTIME_DIR = _config_root() / "local"
LOCAL_MODELS_REGISTRY = LOCAL_RUNTIME_DIR / "models.json"
LOCAL_SERVICES_REGISTRY = LOCAL_RUNTIME_DIR / "services.json"
LOCAL_USER_ALIASES = LOCAL_RUNTIME_DIR / "aliases.json"


def ensure_runtime_dirs() -> None:
    LOCAL_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    (LOCAL_RUNTIME_DIR / "logs").mkdir(parents=True, exist_ok=True)


def _find_aliases_path() -> Path:
    repo_candidate = Path(__file__).resolve().parents[1] / ALIASES_RELATIVE_PATH
    if repo_candidate.exists():
        return repo_candidate

    for entry in map(Path, sys.path):
        candidate = entry / ALIASES_RELATIVE_PATH
        if candidate.exists():
            return candidate

    raise RuntimeError("Unable to locate model_aliases.py for the vLLM launcher.")


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_runtime_dirs()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def _load_builtin_aliases() -> dict[str, dict[str, str]]:
    aliases_path = _find_aliases_path()
    spec = importlib.util.spec_from_file_location("_vllm_aliases", aliases_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load aliases from {aliases_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "BUILTIN_MODEL_ALIASES")


def _load_user_aliases() -> dict[str, dict[str, str]]:
    return _read_json(LOCAL_USER_ALIASES, {})


def _all_aliases() -> dict[str, dict[str, str]]:
    aliases = dict(_load_builtin_aliases())
    for alias, value in _load_user_aliases().items():
        if isinstance(value, dict) and "model" in value:
            aliases[alias] = value
    return aliases


def _default_models_registry() -> dict[str, Any]:
    return {"version": 1, "models": []}


def _default_services_registry() -> dict[str, Any]:
    return {"version": 1, "services": []}


def _load_models_registry() -> dict[str, Any]:
    return _read_json(LOCAL_MODELS_REGISTRY, _default_models_registry())


def _save_models_registry(registry: dict[str, Any]) -> None:
    _write_json(LOCAL_MODELS_REGISTRY, registry)


def _load_services_registry() -> dict[str, Any]:
    registry = _read_json(LOCAL_SERVICES_REGISTRY, _default_services_registry())
    live_services = []
    for service in registry["services"]:
        if _is_process_running(service["pid"]):
            live_services.append(service)
    if len(live_services) != len(registry["services"]):
        registry["services"] = live_services
        _write_json(LOCAL_SERVICES_REGISTRY, registry)
    return registry


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    if not rows:
        return
    widths: dict[str, int] = {}
    for key, label in columns:
        widths[key] = max(len(label), *(len(str(row.get(key, ""))) for row in rows))
    print("  ".join(label.ljust(widths[key]) for key, label in columns))
    print("  ".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns))


def _print_help() -> None:
    print(
        "vLLM local runtime\n\n"
        "Usage:\n"
        "  vllm pull <model>\n"
        "  vllm run <model> [options]\n"
        "  vllm serve <model> [options]\n"
        "  vllm ls\n"
        "  vllm list\n"
        "  vllm aliases\n"
        "  vllm inspect <model>\n"
        "  vllm doctor [model]\n"
        "  vllm status [model]\n"
        "  vllm preflight <model>\n"
        "  vllm ps\n"
        "  vllm stop <service>\n"
        "  vllm logs <service>\n"
        "  vllm rm <model>\n\n"
        "Advanced commands:\n"
        "  vllm chat\n"
        "  vllm complete\n"
        "  vllm bench\n"
        "  vllm collect-env\n"
        "  vllm run-batch\n"
        "  vllm launch\n\n"
        "Examples:\n"
        "  vllm pull deepseek-r1:8b\n"
        "  vllm run llama3.2:3b-instruct\n"
        "  vllm serve qwen2.5:7b-instruct\n"
        "  vllm doctor deepseek-r1:8b\n"
        "  vllm preflight llama3.1:8b-instruct --profile low-memory\n"
        "  vllm aliases\n"
    )


def _print_version() -> None:
    import importlib.metadata

    try:
        print(importlib.metadata.version("vllm"))
    except importlib.metadata.PackageNotFoundError:
        print("vllm (not installed)")


def _resolve_model(model_ref: str) -> dict[str, Any]:
    path = Path(model_ref).expanduser()
    if path.exists():
        return {
            "requested": model_ref,
            "model": str(path.resolve()),
            "source": "path",
            "alias": None,
        }

    if "/" in model_ref:
        return {
            "requested": model_ref,
            "model": model_ref,
            "source": "hf",
            "alias": None,
        }

    aliases = _all_aliases()
    if model_ref in aliases:
        return {
            "requested": model_ref,
            "model": aliases[model_ref]["model"],
            "source": "alias",
            "alias": model_ref,
            "description": aliases[model_ref].get("description"),
        }

    raise SystemExit(
        "Unknown model alias. Use `vllm aliases` to browse built-ins or pass an "
        "exact Hugging Face repo like `meta-llama/Llama-3.1-8B-Instruct`."
    )


def _record_pulled_model(resolved: dict[str, Any], local_path: str) -> None:
    registry = _load_models_registry()
    record = {
        "requested": resolved["requested"],
        "alias": resolved.get("alias"),
        "source": resolved["source"],
        "model": resolved["model"],
        "revision": None,
        "local_path": local_path,
        "pulled_at": int(time.time()),
        "description": resolved.get("description"),
    }
    models = []
    replaced = False
    for existing in registry["models"]:
        if existing["model"] == record["model"]:
            models.append(record)
            replaced = True
        else:
            models.append(existing)
    if not replaced:
        models.append(record)
    registry["models"] = models
    _save_models_registry(registry)


def _cmd_aliases() -> int:
    rows = []
    for alias, value in sorted(_all_aliases().items()):
        rows.append(
            {
                "alias": alias,
                "model": value["model"],
                "description": value.get("description", ""),
            }
        )
    _print_table(
        rows,
        [
            ("alias", "ALIAS"),
            ("model", "HUGGING_FACE_REPO"),
            ("description", "DESCRIPTION"),
        ],
    )
    return 0


def _cmd_ls() -> int:
    registry = _load_models_registry()
    rows = []
    for record in sorted(registry["models"], key=lambda item: item["requested"]):
        rows.append(
            {
                "requested": record["requested"],
                "resolved": record["model"],
                "source": record["source"],
                "local_path": record["local_path"],
            }
        )
    if not rows:
        print("No pulled models found. Use `vllm pull <model>` first.")
        return 0
    _print_table(
        rows,
        [
            ("requested", "REQUESTED"),
            ("resolved", "RESOLVED"),
            ("source", "SOURCE"),
            ("local_path", "LOCAL_PATH"),
        ],
    )
    return 0


def _cmd_inspect(args: list[str]) -> int:
    if not args:
        raise SystemExit("Usage: vllm inspect <model>")
    resolved = _resolve_model(args[0])
    print(json.dumps(resolved, indent=2, sort_keys=True))
    return 0


def _cmd_pull(args: list[str]) -> int:
    if not args:
        raise SystemExit("Usage: vllm pull <model>")
    resolved = _resolve_model(args[0])
    if resolved["source"] == "path":
        _record_pulled_model(resolved, resolved["model"])
        print(f"Recorded local model path: {resolved['model']}")
        return 0

    from huggingface_hub import snapshot_download

    local_path = snapshot_download(repo_id=resolved["model"])
    _record_pulled_model(resolved, local_path)
    print(f"Pulled {resolved['requested']} -> {resolved['model']}")
    print(f"Local path: {local_path}")
    return 0


def _cmd_ps() -> int:
    registry = _load_services_registry()
    rows = []
    for service in sorted(registry["services"], key=lambda item: item["name"]):
        rows.append(
            {
                "name": service["name"],
                "pid": service["pid"],
                "port": service["port"],
                "status": "running",
                "model": service["requested_model"],
            }
        )
    if not rows:
        print("No running vLLM local services.")
        return 0
    _print_table(
        rows,
        [
            ("name", "NAME"),
            ("pid", "PID"),
            ("port", "PORT"),
            ("status", "STATUS"),
            ("model", "MODEL"),
        ],
    )
    return 0


def _cmd_stop(args: list[str]) -> int:
    if not args:
        raise SystemExit("Usage: vllm stop <service>")
    name_or_pid = args[0]
    registry = _load_services_registry()
    kept = []
    target = None
    for service in registry["services"]:
        if service["name"] == name_or_pid or str(service["pid"]) == name_or_pid:
            target = service
            continue
        kept.append(service)
    if target is None:
        raise SystemExit(f"Unknown service: {name_or_pid}")
    os.kill(target["pid"], signal.SIGTERM)
    registry["services"] = kept
    _write_json(LOCAL_SERVICES_REGISTRY, registry)
    print(f"Stopped {target['name']} (pid {target['pid']}).")
    return 0


def _cmd_logs(args: list[str]) -> int:
    if not args:
        raise SystemExit("Usage: vllm logs <service>")
    name_or_pid = args[0]
    registry = _load_services_registry()
    target = None
    for service in registry["services"]:
        if service["name"] == name_or_pid or str(service["pid"]) == name_or_pid:
            target = service
            break
    if target is None:
        raise SystemExit(f"Unknown service: {name_or_pid}")
    log_path = Path(target["log_path"])
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")
    with log_path.open() as f:
        lines = f.readlines()[-40:]
    print("".join(lines), end="")
    return 0


def _cmd_rm(args: list[str]) -> int:
    if not args:
        raise SystemExit("Usage: vllm rm <model>")
    resolved = _resolve_model(args[0])
    registry = _load_models_registry()
    new_models = []
    removed = False
    for record in registry["models"]:
        if record["model"] == resolved["model"]:
            removed = True
            continue
        new_models.append(record)
    registry["models"] = new_models
    _save_models_registry(registry)
    if removed:
        print("Removed local metadata entry.")
    else:
        print("Model metadata not found.")
    return 0


def _delegate_to_vllm(args: list[str]) -> int:
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            *args,
        ],
    )
    return 0


def main() -> int:
    ensure_runtime_dirs()

    argv = sys.argv[1:]
    if not argv:
        _print_help()
        return 0

    if argv[0] in {"-h", "--help", "help"}:
        _print_help()
        return 0

    if argv[0] in {"-v", "--version"}:
        _print_version()
        return 0

    command = argv[0]
    rest = argv[1:]

    if command == "aliases":
        return _cmd_aliases()
    if command in {"ls", "list"}:
        return _cmd_ls()
    if command == "inspect":
        return _cmd_inspect(rest)
    if command == "pull":
        return _cmd_pull(rest)
    if command == "ps":
        return _cmd_ps()
    if command == "stop":
        return _cmd_stop(rest)
    if command == "logs":
        return _cmd_logs(rest)
    if command == "rm":
        return _cmd_rm(rest)

    return _delegate_to_vllm(argv)


if __name__ == "__main__":
    raise SystemExit(main())
