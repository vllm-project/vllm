# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.snapshot.manifest import _write_json_atomic
from vllm.snapshot.types import Oracle, oracles_match

_CANARY_PROMPT = "The capital of France is"


class SnapshotCanaryError(RuntimeError):
    """The initialized engine did not produce a valid snapshot oracle."""


class SnapshotBarrierError(RuntimeError):
    """The controller supplied an invalid snapshot release marker."""


@dataclass(frozen=True)
class ListenerConfig:
    host: str | None
    port: int


@dataclass(frozen=True)
class ControlArgs:
    ready_file: Path
    release_file: Path
    release_timeout_s: float


def write_ready_atomic(path: Path, oracle: Oracle) -> None:
    _write_json_atomic(
        path,
        {
            "sampled_token_logprob": oracle.sampled_token_logprob,
            "token_ids": oracle.token_ids,
            "text": oracle.text,
        },
    )


def read_release_marker(path: Path) -> ListenerConfig:
    path = Path(path)
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode):
        raise SnapshotBarrierError("release marker must not be a symlink")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SnapshotBarrierError("release marker is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("release") is not True:
        raise SnapshotBarrierError("release marker is missing release=true")
    host = payload.get("host")
    port = payload.get("port")
    if host is not None and not isinstance(host, str):
        raise SnapshotBarrierError("release marker host must be a string")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise SnapshotBarrierError("release marker port must be between 1 and 65535")
    return ListenerConfig(host=host, port=port)


def parse_control_args(argv: list[str]) -> tuple[ControlArgs, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--release-timeout-s", type=float, default=900.0)
    control_args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if control_args.release_timeout_s <= 0:
        raise ValueError("release timeout must be positive")
    return (
        ControlArgs(
            ready_file=control_args.ready_file,
            release_file=control_args.release_file,
            release_timeout_s=control_args.release_timeout_s,
        ),
        remaining,
    )


async def _release_reloadable_state(engine: Any) -> None:
    """Discard model and KV state before the process image is captured."""
    await engine.sleep(level=2)


async def _restore_reloadable_state(engine: Any) -> None:
    """Rebuild state discarded by ``_release_reloadable_state``."""
    await engine.wake_up(tags=["weights"])
    await engine.collective_rpc("reload_weights")
    await engine.wake_up(tags=["kv_cache"])


def oracle_from_request_output(request_output: Any) -> Oracle:
    try:
        candidate = request_output.outputs[0]
        token_ids = tuple(candidate.token_ids)
        (sampled_token_id,) = token_ids
        return Oracle(
            token_ids=token_ids,
            text=candidate.text,
            sampled_token_logprob=candidate.logprobs[0][sampled_token_id].logprob,
        )
    except ValueError as error:
        raise SnapshotCanaryError(
            "snapshot canary must produce exactly one finite token logprob"
        ) from error
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise SnapshotCanaryError(
            "snapshot canary did not return sampled token logprob"
        ) from error


async def run_engine_canary(engine: Any) -> Oracle:
    from vllm import SamplingParams

    final_output = None
    sampling_params = SamplingParams(
        temperature=0,
        min_tokens=1,
        max_tokens=1,
        seed=0,
        logprobs=0,
    )
    async for output in engine.generate(
        _CANARY_PROMPT,
        sampling_params,
        request_id="vllm-snapshot-canary",
    ):
        final_output = output
    if final_output is None:
        raise SnapshotCanaryError("canary generation returned no output")
    return oracle_from_request_output(final_output)


def detach_snapshot_streams() -> None:
    """Remove launch-log descriptors from the reusable process image."""
    sys.stdout.flush()
    sys.stderr.flush()
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, sys.stdout.fileno())
        os.dup2(sink, sys.stderr.fileno())
    finally:
        os.close(sink)


async def wait_for_release_marker(
    path: Path,
    *,
    timeout_s: float,
    poll_interval_s: float = 0.05,
) -> ListenerConfig:
    if timeout_s <= 0:
        raise ValueError("release timeout must be positive")
    if poll_interval_s <= 0:
        raise ValueError("release poll interval must be positive")
    remaining_s = timeout_s
    while not path.exists():
        if remaining_s <= 0:
            raise TimeoutError(f"release marker not found before timeout: {path}")
        delay_s = min(poll_interval_s, remaining_s)
        await asyncio.sleep(delay_s)
        remaining_s -= delay_s
    return read_release_marker(path)


def parse_vllm_args(argv: list[str]) -> Any:
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(prog="python -m vllm.snapshot.server")
    args = make_arg_parser(parser).parse_args(argv)
    if args.model_tag is not None:
        args.model = args.model_tag
    if args.grpc or args.headless:
        raise ValueError("snapshot server requires the HTTP frontend")
    if args.api_server_count not in (None, 1):
        raise ValueError("snapshot server supports one API server")
    args.api_server_count = None
    validate_parsed_serve_args(args)
    return args


async def run_vllm_snapshot_child(control: ControlArgs, args: Any) -> None:
    from vllm.entrypoints.launchers.api_server.entry import (
        build_and_serve,
        build_async_engine_client,
    )
    from vllm.entrypoints.launchers.launcher import setup_server

    args.enable_sleep_mode = True

    async with build_async_engine_client(args) as engine:
        oracle = await run_engine_canary(engine)
        try:
            await _release_reloadable_state(engine)
            await _restore_reloadable_state(engine)
            rehearsal_oracle = await run_engine_canary(engine)
            if not oracles_match(oracle, rehearsal_oracle):
                raise SnapshotCanaryError("snapshot rehearsal changed canary output")
            await _release_reloadable_state(engine)
        except SnapshotCanaryError:
            raise
        except Exception as error:
            raise SnapshotCanaryError("snapshot rehearsal failed") from error

        detach_snapshot_streams()
        write_ready_atomic(control.ready_file, oracle)
        listener = await wait_for_release_marker(
            control.release_file,
            timeout_s=control.release_timeout_s,
        )
        args.host = listener.host
        args.port = listener.port
        await _restore_reloadable_state(engine)
        listen_address, sock = setup_server(args, reuse_port=False)
        try:
            shutdown_task = await build_and_serve(
                engine,
                listen_address,
                sock,
                args,
            )
            await shutdown_task
        finally:
            sock.close()


def main(argv: list[str] | None = None) -> None:
    control, remaining = parse_control_args(sys.argv[1:] if argv is None else argv)
    args = parse_vllm_args(remaining)
    import uvloop

    uvloop.run(run_vllm_snapshot_child(control, args))


if __name__ == "__main__":
    main()
