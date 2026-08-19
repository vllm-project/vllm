# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import os
import stat
import sys
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from vllm.snapshot.types import Oracle, oracles_match

EngineT = TypeVar("EngineT")


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
    prompt: str
    include_model_state: bool


def validate_oracle(oracle: Oracle) -> None:
    if not oracle.token_ids:
        raise SnapshotCanaryError("snapshot canary produced no token")


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_ready_atomic(path: Path, oracle: Oracle) -> None:
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = json.dumps(
            {
                "sampled_token_logprob": oracle.sampled_token_logprob,
                "token_ids": oracle.token_ids,
                "text": oracle.text,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        with os.fdopen(file_descriptor, "wb", closefd=True) as ready_file:
            os.fchmod(ready_file.fileno(), 0o600)
            ready_file.write(payload)
            ready_file.flush()
            os.fsync(ready_file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    parser.add_argument("--canary-prompt", default="The capital of France is")
    parser.add_argument("--include-model-state", action="store_true")
    control_args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if control_args.release_timeout_s <= 0:
        raise ValueError("release timeout must be positive")
    if not control_args.canary_prompt:
        raise ValueError("canary prompt must not be empty")
    return (
        ControlArgs(
            ready_file=control_args.ready_file,
            release_file=control_args.release_file,
            release_timeout_s=control_args.release_timeout_s,
            prompt=control_args.canary_prompt,
            include_model_state=control_args.include_model_state,
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
    candidates = request_output.outputs
    if not candidates:
        raise SnapshotCanaryError("canary generation returned no candidates")
    candidate = candidates[0]
    try:
        token_ids = tuple(candidate.token_ids)
        if len(candidate.logprobs) != 1:
            raise ValueError("canary must produce exactly one logprob position")
        (sampled_token_id,) = token_ids
        sampled_token_logprob = candidate.logprobs[0][sampled_token_id].logprob
        oracle = Oracle(
            token_ids=token_ids,
            text=candidate.text,
            sampled_token_logprob=sampled_token_logprob,
        )
    except ValueError as error:
        raise SnapshotCanaryError(
            "snapshot canary must produce exactly one finite token logprob"
        ) from error
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise SnapshotCanaryError(
            "snapshot canary did not return sampled token logprob"
        ) from error
    validate_oracle(oracle)
    return oracle


async def run_engine_canary(engine: Any, prompt: str) -> Oracle:
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
        prompt,
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
    from vllm.entrypoints.openai.api_server import (
        build_and_serve,
        build_async_engine_client,
        setup_server,
    )

    if not control.include_model_state:
        args.enable_sleep_mode = True

    async def run_canary(engine: Any) -> Oracle:
        return await run_engine_canary(engine, control.prompt)

    async def wait_for_release() -> None:
        listener = await wait_for_release_marker(
            control.release_file,
            timeout_s=control.release_timeout_s,
        )
        args.host = listener.host
        args.port = listener.port

    async def bind_and_serve(engine: Any) -> None:
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

    await run_snapshot_child(
        engine_context=build_async_engine_client(args),
        run_canary=run_canary,
        prepare_snapshot=detach_snapshot_streams,
        write_ready=lambda oracle: write_ready_atomic(control.ready_file, oracle),
        wait_for_release=wait_for_release,
        bind_and_serve=bind_and_serve,
        release_reloadable_state=(
            None if control.include_model_state else _release_reloadable_state
        ),
        restore_reloadable_state=(
            None if control.include_model_state else _restore_reloadable_state
        ),
    )


def main(argv: list[str] | None = None) -> None:
    control, remaining = parse_control_args(sys.argv[1:] if argv is None else argv)
    args = parse_vllm_args(remaining)
    import uvloop

    uvloop.run(run_vllm_snapshot_child(control, args))


async def run_snapshot_child(
    *,
    engine_context: AbstractAsyncContextManager[EngineT],
    run_canary: Callable[[EngineT], Awaitable[Oracle]],
    prepare_snapshot: Callable[[], None],
    write_ready: Callable[[Oracle], None],
    wait_for_release: Callable[[], Awaitable[None]],
    bind_and_serve: Callable[[EngineT], Awaitable[None]],
    release_reloadable_state: Callable[[EngineT], Awaitable[None]] | None = None,
    restore_reloadable_state: Callable[[EngineT], Awaitable[None]] | None = None,
) -> None:
    """Initialize and validate an engine before creating its HTTP listener."""
    async with engine_context as engine:
        oracle = await run_canary(engine)
        validate_oracle(oracle)
        if release_reloadable_state is not None:
            try:
                await release_reloadable_state(engine)
                if restore_reloadable_state is None:
                    raise RuntimeError("compact snapshot restore callback is missing")
                await restore_reloadable_state(engine)
                rehearsal_oracle = await run_canary(engine)
                validate_oracle(rehearsal_oracle)
                if not oracles_match(oracle, rehearsal_oracle):
                    raise SnapshotCanaryError(
                        "compact snapshot rehearsal changed canary output; retry "
                        "create with --include-model-state"
                    )
                await release_reloadable_state(engine)
            except SnapshotCanaryError:
                raise
            except Exception as error:
                raise SnapshotCanaryError(
                    "compact snapshot rehearsal failed; retry create with "
                    "--include-model-state"
                ) from error
        prepare_snapshot()
        write_ready(oracle)
        await wait_for_release()
        if restore_reloadable_state is not None:
            await restore_reloadable_state(engine)
        await bind_and_serve(engine)


if __name__ == "__main__":
    main()
