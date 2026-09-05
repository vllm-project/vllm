# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import re
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

from torch.fx import GraphModule

from vllm.logger import init_logger

logger = init_logger(__name__)


def _safe_file_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return component or "model"


def dump_fx_graph(gm: GraphModule, dump_dir: Path, prefix: str) -> Path:
    """Persist a Dynamo FX graph before its compiler backend is invoked."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_file_component(prefix)
    path = dump_dir / f"fx_graph_{name}_pid{os.getpid()}_{uuid4().hex[:8]}.txt"
    readable = gm.print_readable(print_output=False)
    path.write_text(f"{readable}\n\n==== raw graph ====\n{gm.graph}\n", encoding="utf-8")
    return path


def wrap_backend_with_fx_dump(
    backend: str | Callable[..., Any], dump_dir: Path, prefix: str
) -> Callable[..., Any]:
    """Dump FX graphs and lower them with the direct InferRT backend.

    A string backend is retained in the vLLM configuration for compatibility
    with its backend validation, but is deliberately not resolved through
    TorchDynamo's registry. InferRT is an external backend and is not
    registered there under the ``inductor`` name.
    """
    if isinstance(backend, str):
        from ms_inferrt.torch.fx_backend import backend as compiler_fn
    else:
        compiler_fn = backend

    @wraps(compiler_fn)
    def dumping_backend(
        gm: GraphModule, example_inputs: list[Any], **kwargs: Any
    ) -> Any:
        path = dump_fx_graph(gm, dump_dir, prefix)
        logger.info("Enter InferRT Backend; Dynamo FX graph saved to %s", path)
        # InferRT accepts only the graph and example inputs. Any kwargs here
        # belong to the original compiler backend rather than InferRT.
        return compiler_fn(gm, example_inputs)

    return dumping_backend
