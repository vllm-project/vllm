# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import copy
import signal
import socket
from argparse import Namespace

from vllm import AsyncEngineArgs, envs
from vllm.config import ModelConfig, VllmConfig
from vllm.logger import configure_logging_from_args, init_logger

from ..app import build_app
from ..launcher import serve_http, setup_server
from ..utils.server_utils import get_uvicorn_log_config
from .app_state import init_render_app_state

logger = init_logger("vllm.entrypoints.launchers.render.entry")


def _build_render_model_config(model: str, base_args: Namespace) -> ModelConfig:
    """Build a render-safe `ModelConfig` for one extra served model.

    Clones the parsed CLI args, swaps the model name, and reuses the
    `AsyncEngineArgs.create_model_config` path — same code as the primary
    model so every knob (dtype, tokenizer_mode, trust_remote_code,
    revision, hf_overrides…) applies to extras too.
    """
    per_model_args = copy.copy(base_args)
    per_model_args.model = model
    # Extra models default to their own repo as the served name; users can
    # still override with `--extra-served-model NAME=REPO`.
    per_model_args.served_model_name = None
    engine_args = AsyncEngineArgs.from_cli_args(per_model_args)
    model_config = engine_args.create_model_config()
    model_config.quantization = None
    return model_config


async def build_and_serve_renderer(
    vllm_config: VllmConfig,
    listen_address: str,
    sock: socket.socket,
    args: Namespace,
    extra_model_configs: dict[str, ModelConfig] | None = None,
    **uvicorn_kwargs,
) -> asyncio.Task:
    """Build FastAPI app for a CPU-only render server, initialize state, and
    start serving.

    `extra_model_configs` maps served-model-name -> `ModelConfig` for models
    beyond the primary `--model`. Each extra config produces its own renderer
    inside a `MultiModelOnlineRenderer` so a single render process can serve
    N models via the `model` field on `/tokenize`, `/detokenize`, and — once
    the `ServingRender` refactor lands — `/render`.

    Returns the shutdown task for the caller to await.
    """
    # Get uvicorn log config (from file or with endpoint filter)
    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    app = build_app(args, ("render",))
    await init_render_app_state(
        vllm_config, app.state, args, extra_model_configs=extra_model_configs
    )

    logger.info("Starting vLLM server on %s", listen_address)

    return await serve_http(
        app,
        sock=sock,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        # NOTE: When the 'disable_uvicorn_access_log' value is True,
        # no access log will be output.
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        ssl_ciphers=args.ssl_ciphers,
        h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
        h11_max_header_count=args.h11_max_header_count,
        **uvicorn_kwargs,
    )


async def run_launch_fastapi(args: argparse.Namespace) -> None:
    """Run the online serving layer with FastAPI (no GPU inference)."""

    # Interrupt initialization if SIGTERM arrives before uvicorn installs
    # its own signal handlers. Once uvicorn is running it replaces this.
    def _interrupt_init(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, _interrupt_init)

    # 1. Socket binding
    listen_address, sock = setup_server(args, reuse_port=False)

    # 2. Build and serve the API server
    engine_args = AsyncEngineArgs.from_cli_args(args)
    model_config = engine_args.create_model_config()

    # Render servers preprocess data only — no inference, no quantized kernels.
    # Clear quantization so VllmConfig skips quant dtype/capability validation.
    model_config.quantization = None

    # Render servers never allocate KV cache; suppress the spurious CPU KV
    # cache space warning from CpuPlatform.check_and_update_config.
    envs.VLLM_CPU_KVCACHE_SPACE = 0

    vllm_config = VllmConfig(
        model_config=model_config,
        logging_config=engine_args.create_logging_config(),
    )

    extra_model_configs: dict[str, ModelConfig] = {}
    for spec in getattr(args, "extra_served_model", None) or []:
        name, _, repo = spec.partition("=")
        if not repo:
            name, repo = spec, spec
        if name in extra_model_configs:
            raise ValueError(
                f"--extra-served-model {name!r} is specified more than once"
            )
        extra_model_configs[name] = _build_render_model_config(repo, args)
        logger.info("Loaded extra render model %r from %r", name, repo)

    shutdown_task = await build_and_serve_renderer(
        vllm_config,
        listen_address,
        sock,
        args,
        extra_model_configs=extra_model_configs or None,
    )
    try:
        await shutdown_task
    finally:
        sock.close()


if __name__ == "__main__":
    import uvloop

    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from ..cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="Starts a GPU-less rendering server "
        "for preprocessing and postprocessing only"
    )
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--extra-served-model",
        action="append",
        default=[],
        metavar="NAME[=REPO]",
        help=(
            "Add a model to the render server beyond the primary --model. "
            "Repeatable. Each entry is either a bare HF repo ID (used as "
            "both the served name and the repo) or NAME=REPO to serve the "
            "same repo under a custom name. Every extra model is loaded "
            "with the same tokenizer/dtype/trust_remote_code/hf_overrides "
            "as the primary model."
        ),
    )
    args = parser.parse_args()
    configure_logging_from_args(args)
    validate_parsed_serve_args(args)

    uvloop.run(run_launch_fastapi(args))
