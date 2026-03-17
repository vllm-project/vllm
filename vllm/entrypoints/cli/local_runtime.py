# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Helpers for the Ollama-like local vLLM CLI experience."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

import vllm.envs as envs
from vllm.entrypoints.cli.model_aliases import BUILTIN_MODEL_ALIASES
from vllm.utils.network_utils import get_open_port

LOCAL_RUNTIME_DIR = Path(envs.VLLM_CONFIG_ROOT) / "local"
LOCAL_MODELS_REGISTRY = LOCAL_RUNTIME_DIR / "models.json"
LOCAL_SERVICES_REGISTRY = LOCAL_RUNTIME_DIR / "services.json"
LOCAL_USER_ALIASES = LOCAL_RUNTIME_DIR / "aliases.json"
LOCAL_LOG_DIR = LOCAL_RUNTIME_DIR / "logs"


@dataclass
class ResolvedModel:
    requested: str
    model: str
    source: str
    alias: str | None = None
    description: str | None = None
    revision: str | None = None


def ensure_runtime_dirs() -> None:
    LOCAL_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)


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
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as tmp_file:
            json.dump(payload, tmp_file, indent=2, sort_keys=True)
            tmp_file.write("\n")
        os.replace(tmp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def _default_models_registry() -> dict[str, Any]:
    return {"version": 1, "models": []}


def _default_services_registry() -> dict[str, Any]:
    return {"version": 1, "services": []}


def load_models_registry() -> dict[str, Any]:
    return _read_json(LOCAL_MODELS_REGISTRY, _default_models_registry())


def save_models_registry(registry: dict[str, Any]) -> None:
    _write_json(LOCAL_MODELS_REGISTRY, registry)


def load_user_aliases() -> dict[str, Any]:
    return _read_json(LOCAL_USER_ALIASES, {})


def load_services_registry() -> dict[str, Any]:
    registry = _read_json(LOCAL_SERVICES_REGISTRY, _default_services_registry())
    live_services = [
        svc for svc in registry["services"] if is_process_running(svc["pid"])
    ]
    if len(live_services) != len(registry["services"]):
        registry["services"] = live_services
        save_services_registry(registry)
    return registry


def save_services_registry(registry: dict[str, Any]) -> None:
    _write_json(LOCAL_SERVICES_REGISTRY, registry)


def iter_known_aliases() -> dict[str, dict[str, str]]:
    aliases = dict(BUILTIN_MODEL_ALIASES)
    for alias, value in load_user_aliases().items():
        if not isinstance(value, dict) or "model" not in value:
            continue
        aliases[alias] = value
    return aliases


def resolve_model_reference(
    model_ref: str,
    revision: str | None = None,
) -> ResolvedModel:
    if Path(model_ref).expanduser().exists():
        return ResolvedModel(
            requested=model_ref,
            model=str(Path(model_ref).expanduser().resolve()),
            source="path",
            revision=revision,
        )

    if "/" in model_ref:
        return ResolvedModel(
            requested=model_ref,
            model=model_ref,
            source="hf",
            revision=revision,
        )

    aliases = iter_known_aliases()
    alias_entry = aliases.get(model_ref)
    if alias_entry is None:
        raise ValueError(
            "Unknown model alias. Use an exact Hugging Face repo like "
            "`meta-llama/Llama-3.1-8B-Instruct` or a built-in alias like "
            "`deepseek-r1:8b`."
        )

    return ResolvedModel(
        requested=model_ref,
        model=alias_entry["model"],
        source="alias",
        alias=model_ref,
        description=alias_entry.get("description"),
        revision=revision,
    )


def record_pulled_model(resolved: ResolvedModel, local_path: str) -> dict[str, Any]:
    registry = load_models_registry()
    models = registry["models"]
    record = {
        "requested": resolved.requested,
        "alias": resolved.alias,
        "source": resolved.source,
        "model": resolved.model,
        "revision": resolved.revision,
        "local_path": local_path,
        "pulled_at": int(time.time()),
        "description": resolved.description,
    }

    updated = False
    for index, existing in enumerate(models):
        if (
            existing["model"] == record["model"]
            and existing.get("revision") == record.get("revision")
        ):
            models[index] = record
            updated = True
            break
    if not updated:
        models.append(record)
    save_models_registry(registry)
    return record


def get_pulled_model(resolved: ResolvedModel) -> dict[str, Any] | None:
    registry = load_models_registry()
    for record in registry["models"]:
        if (
            record["model"] == resolved.model
            and record.get("revision") == resolved.revision
        ):
            return record
    return None


def ensure_model_available(
    resolved: ResolvedModel,
    download_dir: str | None = None,
) -> dict[str, Any]:
    if resolved.source == "path":
        return record_pulled_model(resolved, resolved.model)

    existing = get_pulled_model(resolved)
    if existing is not None and Path(existing["local_path"]).exists():
        return existing

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for `vllm pull`, `vllm run`, and alias-based "
            "serving. Install the repo environment first."
        ) from exc

    local_path = snapshot_download(
        repo_id=resolved.model,
        revision=resolved.revision,
        cache_dir=download_dir,
    )
    return record_pulled_model(resolved, local_path)


def remove_pulled_model(
    resolved: ResolvedModel,
    purge_cache: bool = False,
) -> tuple[bool, str | None]:
    registry = load_models_registry()
    new_models = []
    removed_record = None
    for record in registry["models"]:
        if (
            record["model"] == resolved.model
            and record.get("revision") == resolved.revision
        ):
            removed_record = record
            continue
        new_models.append(record)

    if removed_record is None:
        return False, None

    registry["models"] = new_models
    save_models_registry(registry)

    purged_path = None
    if purge_cache:
        local_path = Path(removed_record["local_path"])
        if removed_record["source"] != "path" and local_path.exists():
            shutil.rmtree(local_path, ignore_errors=True)
            purged_path = str(local_path)

    return True, purged_path


def slugify_model_name(model_ref: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_ref).strip("-").lower()
    return slug or "model"


def build_service_name(explicit_name: str | None, model_ref: str) -> str:
    return explicit_name or slugify_model_name(model_ref)


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def find_service(name_or_id: str) -> dict[str, Any] | None:
    registry = load_services_registry()
    for service in registry["services"]:
        if service["name"] == name_or_id or str(service["pid"]) == str(name_or_id):
            return service
    return None


def register_service(record: dict[str, Any]) -> None:
    registry = load_services_registry()
    services = [svc for svc in registry["services"] if svc["name"] != record["name"]]
    services.append(record)
    registry["services"] = services
    save_services_registry(registry)


def remove_service(name_or_id: str) -> dict[str, Any] | None:
    registry = load_services_registry()
    removed = None
    kept = []
    for service in registry["services"]:
        if service["name"] == name_or_id or str(service["pid"]) == str(name_or_id):
            removed = service
            continue
        kept.append(service)
    if removed is not None:
        registry["services"] = kept
        save_services_registry(registry)
    return removed


def allocate_service_port(preferred_port: int | None = None) -> int:
    return preferred_port if preferred_port is not None else get_open_port()


def spawn_service_process(command: list[str], log_path: Path) -> subprocess.Popen[str]:
    ensure_runtime_dirs()
    log_file = log_path.open("a")
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


def wait_for_service(url: str, pid: int, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if not is_process_running(pid):
            raise RuntimeError("Service process exited before becoming ready.")
        try:
            with request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for service readiness: {last_error}")


def stop_service(name_or_id: str, force: bool = False) -> dict[str, Any]:
    service = find_service(name_or_id)
    if service is None:
        raise ValueError(f"Unknown service: {name_or_id}")

    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(service["pid"], sig)
    if force:
        remove_service(name_or_id)
        return service

    deadline = time.time() + 10
    while time.time() < deadline:
        if not is_process_running(service["pid"]):
            remove_service(name_or_id)
            return service
        time.sleep(0.25)

    raise RuntimeError(
        "Timed out waiting for service to stop. Retry with `vllm stop --force`."
    )


def format_service_rows() -> list[dict[str, Any]]:
    registry = load_services_registry()
    rows = []
    for service in sorted(registry["services"], key=lambda item: item["name"]):
        rows.append(
            {
                "name": service["name"],
                "pid": service["pid"],
                "port": service["port"],
                "model": service["requested_model"],
                "resolved_model": service["resolved_model"],
                "status": (
                    "running" if is_process_running(service["pid"]) else "stopped"
                ),
                "uptime_s": max(int(time.time() - service["started_at"]), 0),
            }
        )
    return rows


def default_service_record(
    *,
    name: str,
    pid: int,
    port: int,
    requested_model: str,
    resolved_model: str,
    log_path: Path,
    command: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "pid": pid,
        "port": port,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "log_path": str(log_path),
        "started_at": int(time.time()),
        "command": command,
    }


def print_kv(data: dict[str, Any]) -> None:
    width = max(len(key) for key in data) if data else 0
    for key, value in data.items():
        print(f"{key.ljust(width)} : {value}")


def print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    if not rows:
        return
    widths = {}
    for key, label in columns:
        widths[key] = max(len(label), *(len(str(row.get(key, ""))) for row in rows))
    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    print(header)
    print("  ".join("-" * widths[key] for key, _label in columns))
    for row in rows:
        print(
            "  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _label in columns)
        )


def tail_file(path: Path, follow: bool = False, lines: int = 40) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open() as f:
        content = f.readlines()
        for line in content[-lines:]:
            print(line, end="")

        if not follow:
            return

        while True:
            next_line = f.readline()
            if next_line:
                print(next_line, end="")
            else:
                time.sleep(0.5)
