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
_DUMP_GRAPH_COUNTER = 0


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
    """Dump FX graphs and lower them with an external FX backend.

    A string backend is retained in the vLLM configuration for compatibility
    with its backend validation, but is deliberately not resolved through
    TorchDynamo's registry. External backends are not registered there under
    the ``inductor`` name.
    """
    if isinstance(backend, str):
        # Keep InferRT as the historical default, while allowing the same
        # wrapper to exercise FXRT without changing vLLM's backend registry.
        # This is intentionally selected before worker start (environment is
        # inherited by every DP/TP worker).
        external_backend = os.getenv("VLLM_EXTERNAL_FX_BACKEND", "ms_inferrt")
        if external_backend == "fxrt":
            from fxrt.torch.fx_backend import backend as compiler_fn
        elif external_backend == "ms_inferrt":
            from ms_inferrt.torch.fx_backend import backend as compiler_fn
        else:
            raise ValueError(
                "VLLM_EXTERNAL_FX_BACKEND must be 'fxrt' or 'ms_inferrt', "
                f"got {external_backend!r}"
            )
    else:
        compiler_fn = backend

    @wraps(compiler_fn)
    def dumping_backend(
        gm: GraphModule, example_inputs: list[Any], **kwargs: Any
    ) -> Any:
        global _DUMP_GRAPH_COUNTER
        _DUMP_GRAPH_COUNTER += 1
        path = dump_fx_graph(gm, dump_dir, prefix)
        logger.info(
            "Enter external FX backend graph_id=%d pid=%d; Dynamo FX graph saved to %s",
            _DUMP_GRAPH_COUNTER,
            os.getpid(),
            path,
        )
        # External backends accept only the graph and example inputs. Any
        # kwargs here belong to the original compiler backend.
        return compiler_fn(gm, example_inputs)

    return dumping_backend
