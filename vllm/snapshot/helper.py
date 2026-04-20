# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The snapshot-helper process.

This process exists for one purpose: import everything the APIServer
would import during a normal `vllm serve` cold start, initialize the
CUDA context on the target GPU, and then pause in a signal handler so
that `criu dump` (paired with `cuda-checkpoint --toggle`) can capture
its state.

After restore, a pre-installed SIGUSR2 handler reads the resume payload
(argv, env, cwd, stdio FDs), swaps them in, and calls `main()` from
`vllm.entrypoints.cli.main`. The payload path is passed via the
`VLLM_RESUME_PAYLOAD` env var which the restore wrapper sets before
kicking SIGUSR2.

USAGE:
    # In 'dry-run' mode (no actual CUDA init, no pause):
    VLLM_SNAPSHOT_DRY_RUN=1 python -m vllm.snapshot.helper
    # In real mode (touches CUDA, pauses forever):
    python -m vllm.snapshot.helper
"""

from __future__ import annotations

import json
import os
import signal
import sys


def _eager_imports() -> None:
    """Pull in every module the CLI dispatch path eventually loads.

    Kept explicit rather than relying on transitive imports so that the
    snapshot content is stable across minor vllm refactors. Any module
    listed here will be in `sys.modules` at snapshot time and therefore
    already resolved on restore.
    """
    # Core
    import vllm  # noqa: F401
    import vllm.env_override  # noqa: F401 — triggers monkey-patches
    import vllm.envs  # noqa: F401

    # Logger, config, compilation
    import vllm.logger  # noqa: F401
    import vllm.config  # noqa: F401
    import vllm.config.compilation  # noqa: F401
    import vllm.config.model  # noqa: F401
    import vllm.config.vllm  # noqa: F401

    # Entry-point dispatch surface
    import vllm.entrypoints.cli.main  # noqa: F401
    import vllm.entrypoints.cli.serve  # noqa: F401
    import vllm.entrypoints.openai.api_server  # noqa: F401
    import vllm.entrypoints.openai.cli_args  # noqa: F401

    # Engine + runtime
    import vllm.engine.arg_utils  # noqa: F401
    import vllm.v1.engine.async_llm  # noqa: F401
    import vllm.v1.engine.core  # noqa: F401

    # Heavy third-party
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import fastapi  # noqa: F401
    import uvloop  # noqa: F401


def _init_cuda_context() -> None:
    """Create the CUDA context on device 0 (or CUDA_VISIBLE_DEVICES[0]).

    We don't allocate any tensors — just make sure the driver has handed
    out a context for this process. cuda-checkpoint then has a context
    to freeze.
    """
    import torch

    if not torch.cuda.is_available():
        return

    # Touching a CUDA property initializes the context without allocating
    # real tensors. _get_device_properties is internal-ish but stable.
    _ = torch.cuda.get_device_name(0)
    torch.cuda.init()
    # A sync ensures the driver has fully negotiated the context.
    torch.cuda.synchronize()


def _resume(signum: int, frame) -> None:  # type: ignore[no-untyped-def]
    """Post-restore entry point.

    Reads the payload deposited by the restore wrapper, swaps argv/env/cwd
    back to the user's invocation, dup2's stdio FDs if provided, then
    hands off to the normal CLI main().
    """
    payload_path = os.environ.get("VLLM_RESUME_PAYLOAD")
    if not payload_path or not os.path.exists(payload_path):
        # Nothing to do — exit the helper quietly.
        os._exit(0)

    with open(payload_path) as f:
        payload = json.load(f)

    # Argv
    sys.argv = [sys.argv[0]] + list(payload.get("argv", [])[1:])

    # Env (wholesale replace; caller's env is authoritative)
    if "env" in payload:
        os.environ.clear()
        os.environ.update(payload["env"])

    # Cwd
    if "cwd" in payload:
        try:
            os.chdir(payload["cwd"])
        except OSError:
            pass

    # FD handover (indexes into an inherited-fd table set up by the
    # wrapper; omitted in prototype)
    for std_fd, key in ((0, "stdin_fd"), (1, "stdout_fd"), (2, "stderr_fd")):
        src = payload.get(key)
        if isinstance(src, int) and src >= 0:
            try:
                os.dup2(src, std_fd)
                os.close(src)
            except OSError:
                pass

    # Clean up payload file
    try:
        os.unlink(payload_path)
    except OSError:
        pass

    # Off to the normal CLI main — same path a fresh-start `vllm serve`
    # would have taken. At this point all imports are resolved and CUDA
    # context is live.
    from vllm.entrypoints.cli.main import main

    try:
        main()
    finally:
        os._exit(0)


def main() -> None:
    dry_run = os.environ.get("VLLM_SNAPSHOT_DRY_RUN") == "1"
    snap_ready_file = os.environ.get("VLLM_SNAPSHOT_READY_FILE")

    # Install the resume handler BEFORE any snapshot — the snapshot image
    # must include this handler, otherwise SIGUSR2 on restore has no effect.
    signal.signal(signal.SIGUSR2, _resume)

    _eager_imports()
    _init_cuda_context()

    if snap_ready_file:
        # Let the snapshot driver know we're ready to be captured.
        with open(snap_ready_file, "w") as f:
            f.write(f"{os.getpid()}\n")

    if dry_run:
        print(
            f"[snapshot.helper] Dry run complete. pid={os.getpid()} "
            f"would now pause(); all imports + CUDA init done.",
            flush=True,
        )
        return

    # Pause forever. `criu dump` will capture this state. On restore,
    # SIGUSR2 is kicked and _resume() fires.
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
