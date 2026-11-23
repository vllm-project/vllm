# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup checks for the opt-in streaming session API."""

from argparse import Namespace

from vllm.engine.arg_utils import AsyncEngineArgs


def configure_streaming(args: Namespace, engine_args: AsyncEngineArgs) -> None:
    if not engine_args.enforce_eager:
        raise ValueError("--enable-streaming requires --enforce-eager")
    if engine_args.async_scheduling:
        raise ValueError("--enable-streaming requires --no-async-scheduling")
    if (getattr(args, "api_server_count", None) or 1) > 1:
        raise ValueError("--enable-streaming requires --api-server-count 1")
    # Async scheduling defaults on for many models; retention mutates state
    # between completed chunks and needs the synchronous scheduler.
    engine_args.async_scheduling = False
