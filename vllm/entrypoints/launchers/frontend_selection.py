# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Throwaway CI probe: a crude `auto` frontend choice for `vllm serve`.

`VLLM_USE_RUST_FRONTEND=1` forces Rust and `=0` forces Python. When unset,
`resolve_frontend` picks Rust only for text-only generate models that pass the
static gates below and survive a Rust startup preflight, and records the
decision and its reason so a CI run can be aggregated by frontend.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import regex as re

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

PROBE_FILE_ENV = "VLLM_FRONTEND_PROBE_FILE"
PREFLIGHT_MARKER = "resolved coordinator mode"
PREFLIGHT_TIMEOUT_S = 300.0


def _record(frontend: str, mode: str, reason: str, model: str | None) -> None:
    logger.warning(
        "[FRONTEND-PROBE] frontend=%s mode=%s reason=%s model=%s test=%s",
        frontend,
        mode,
        reason,
        model,
        os.environ.get("PYTEST_CURRENT_TEST", ""),
    )
    path = os.environ.get(PROBE_FILE_ENV)
    if not path:
        return
    entry = {
        "pid": os.getpid(),
        "time": time.time(),
        "frontend": frontend,
        "mode": mode,
        "reason": reason,
        "model": model,
        "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
        "argv": sys.argv[1:],
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _static_gate(args: argparse.Namespace) -> tuple[str | None, "ModelConfig | None"]:
    """Return the first gate that sends this config to Python, if any, plus the
    probe ModelConfig."""
    from vllm.engine.arg_utils import AsyncEngineArgs

    if getattr(args, "grpc", False):
        return "python-grpc", None
    if args.tokenizer_mode == "mistral":
        return "tokenizer-mode-mistral", None
    # Rust has no derender or main-server render routes. Under dev mode the
    # flag defaults to on, so only an explicit request gates.
    if args.enable_scale_out and (
        not envs.VLLM_SERVER_DEV_MODE or "--enable-scale-out" in sys.argv
    ):
        return "scale-out", None
    try:
        model_config = AsyncEngineArgs.from_cli_args(args).create_model_config()
    except Exception as e:
        logger.warning("Frontend probe could not build ModelConfig: %r", e)
        return "model-config-error", None
    if model_config.runner_type != "generate":
        return f"runner-{model_config.runner_type}", model_config
    if model_config.registry.is_transcription_model(
        model_config.architectures, model_config
    ):
        return "transcription", model_config
    if model_config.is_multimodal_model:
        return "multimodal", model_config
    if model_config.tokenizer_mode == "mistral":
        return "tokenizer-mode-mistral", model_config
    return None, model_config


def _rust_preflight(
    args: argparse.Namespace, model_config: "ModelConfig", binary: str
) -> str | None:
    """Run the Rust frontend until it blocks on the engine handshake.

    The frontend parses its arguments, validates the config and loads its
    tokenizer and renderer before it connects to an engine. Return None when it
    gets that far, else the reason it died.
    """
    from vllm.v1.utils import build_rust_frontend_args

    args_json, _ = build_rust_frontend_args(args, model_config)
    payload = json.loads(args_json)
    payload.pop("logging_config", None)
    socks = [socket.socket() for _ in range(3)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
        s.listen()
    fds = [s.fileno() for s in socks]
    cmd = [
        binary,
        "frontend",
        "--listen-fd",
        str(fds[0]),
        "--input-listener-fd",
        str(fds[1]),
        "--output-listener-fd",
        str(fds[2]),
        "--args-json",
        json.dumps(payload),
    ]
    env = {**os.environ, "VLLM_LOGGING_LEVEL": "INFO", "NO_COLOR": "1"}
    for s in socks:
        os.set_inheritable(s.fileno(), True)
    lines: list[str] = []
    reached = threading.Event()
    try:
        proc = subprocess.Popen(
            cmd,
            pass_fds=fds,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except OSError as e:
        return f"rust-binary-unusable: {e}"

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip())
            if PREFLIGHT_MARKER in line:
                reached.set()
                return

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    deadline = time.monotonic() + PREFLIGHT_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            if reached.is_set():
                return None
            if proc.poll() is not None:
                t.join(2)
                if reached.is_set():
                    return None
                return _summarize_failure(lines)
            time.sleep(0.05)
        return None  # slow tokenizer load: let the real start decide
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for s in socks:
            s.close()


def _summarize_failure(lines: list[str]) -> str:
    text = "\n".join(lines)
    bullets = re.findall(r"^- (\S+)$", text, re.M)
    if "not implemented in Rust frontend yet" in text and bullets:
        return "rust-unsupported-args:" + ",".join(bullets)
    errors = [ln for ln in lines if re.search(r"error|failed|invalid", ln, re.I)]
    tail = (errors or lines or ["no output"])[-1]
    return "rust-startup-error:" + re.sub(r"\s+", " ", tail)[:300]


def resolve_frontend(args: argparse.Namespace) -> None:
    """Fix `VLLM_USE_RUST_FRONTEND` to "0" or "1" for this process tree."""
    if getattr(args, "model_tag", None):
        args.model = args.model_tag
    model = getattr(args, "model", None)

    forced = envs.VLLM_USE_RUST_FRONTEND
    if forced is not None:
        _record("rust" if forced else "python", "forced", "env", model)
        return

    frontend, reason = "python", ""
    if getattr(args, "grpc_port", None) is not None:
        frontend, reason = "rust", "grpc-port"
    elif (gate_and_config := _static_gate(args))[0]:
        reason = f"gate:{gate_and_config[0]}"
    elif (model_config := gate_and_config[1]) is None:
        reason = "gate:no-model-config"
    else:
        os.environ["VLLM_USE_RUST_FRONTEND"] = "1"
        try:
            binary = envs.VLLM_RUST_FRONTEND_PATH
        except FileNotFoundError:
            binary = None
        if not binary or not os.path.isfile(binary):
            reason = "rust-binary-missing"
        elif failure := _rust_preflight(args, model_config, binary):
            reason = failure
        else:
            frontend, reason = "rust", "auto"
    os.environ["VLLM_USE_RUST_FRONTEND"] = "1" if frontend == "rust" else "0"
    _record(frontend, "auto", reason, model)
