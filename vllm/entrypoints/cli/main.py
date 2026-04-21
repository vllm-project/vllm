# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import contextlib
import importlib.metadata
import os
import sys
import threading as _threading


# [startup] Kick off torch + transformers .so/module loading in a background
# thread before we touch vllm.logger (which pulls vllm/__init__.py ->
# vllm.env_override -> `import torch` on the main thread). Python import
# lock serializes the same-module import across threads, but the .so dlopen
# inside torch's init releases the GIL during file I/O. Main thread's
# non-torch imports (vllm.envs submodules, stdlib, fastapi, etc.) can make
# progress on the CPU while the background thread pays the ~2 s of cuda
# .so loading. `import transformers` is also ~2 s of cold-disk work and
# depends on torch; chain it after torch in the same thread so subsequent
# `from transformers import ...` lines on the main thread hit a warm
# module cache.
def _bg_preload_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        return
    with contextlib.suppress(Exception):
        import transformers  # noqa: F401


_threading.Thread(
    target=_bg_preload_torch, daemon=True, name="vllm-torch-preload"
).start()


# [startup] Pre-spawn EngineCore via forkserver preload, in a background
# thread. Only fires for `vllm serve` (the only subcommand that spawns a
# long-running EngineCore). The forkserver process is forked once and
# preloaded with vllm.v1.engine.async_llm (~3-5 s of imports). When
# AsyncLLM.from_vllm_config later runs, Process.start() forks from the
# already-warm forkserver instead of paying spawn() cost (~5 s in child
# for fresh Python + imports).
#
# Kicking the preload in a BG thread lets the ~3-5 s ensure_running cost
# overlap with APIServer's argparse + config resolution (~5-10 s on cold
# disk). Default cli_env_setup sets spawn; we override to forkserver
# before that runs so the path is consistent.
def _bg_prewarm_forkserver() -> None:
    try:
        import multiprocessing
        import multiprocessing.forkserver as forkserver

        # set_start_method MUST be called before ensure_running. It also
        # can only be called once per process; any later override by
        # vllm's build_async_engine_client will just see the existing
        # setting.
        multiprocessing.set_start_method("forkserver", force=False)
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
    except Exception:
        pass


if len(sys.argv) > 1 and sys.argv[1] == "serve":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
    # daemon=True so early CLI exits (bad args, --help, import errors)
    # don't hang waiting for ensure_running(). The forkserver subprocess
    # itself is tracked by module-level state in multiprocessing.forkserver
    # and survives this thread exiting; subsequent spawn() calls reuse it.
    _threading.Thread(
        target=_bg_prewarm_forkserver,
        daemon=True,
        name="vllm-forkserver-prewarm",
    ).start()


from vllm.logger import init_logger  # noqa: E402

logger = init_logger(__name__)


def main():
    import vllm.entrypoints.cli.benchmark.main
    import vllm.entrypoints.cli.collect_env
    import vllm.entrypoints.cli.launch
    import vllm.entrypoints.cli.openai
    import vllm.entrypoints.cli.run_batch
    import vllm.entrypoints.cli.serve
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        vllm.entrypoints.cli.openai,
        vllm.entrypoints.cli.serve,
        vllm.entrypoints.cli.launch,
        vllm.entrypoints.cli.benchmark.main,
        vllm.entrypoints.cli.collect_env,
        vllm.entrypoints.cli.run_batch,
    ]

    cli_env_setup()

    # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        from vllm import platforms

        if platforms.current_platform.is_unspecified():
            from vllm.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
