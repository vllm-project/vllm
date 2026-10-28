# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build each model on the meta device and dump what vLLM constructs.

Records ModelConfig-derived sizes, every parameter shape, and scalar
attributes of every module (head_dim, num_kv_heads, sliding_window, ...).
Needs no GPU and no weights. Run once per vLLM tree, then compare the two
outputs with diff_dumps.py:

    PYTHONPATH=<vllm tree> .venv/bin/python dump_model.py out.jsonl <repo>...

Pass --trust-remote-code to resolve checkpoints the way users who set it do.
"""

import json
import socket
import sys

import torch

from vllm.config import set_current_vllm_config
from vllm.config.vllm import VllmConfig
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype

# Construction only: skip model runner checks that need a GPU or Triton
VllmConfig._get_v1_model_runner_unsupported_features = lambda self: []  # type: ignore[method-assign]

# Model code that moves submodules to a device must leave them on meta
_module_to = torch.nn.Module.to


def _meta_safe_to(self, *args, **kwargs):
    kwargs.pop("device", None)
    args = tuple(a for a in args if not isinstance(a, (torch.device, str, int)))
    return _module_to(self, *args, **kwargs)


torch.nn.Module.to = _meta_safe_to  # type: ignore[method-assign]

SCALARS = (bool, int, float, str)


def module_attrs(model: torch.nn.Module) -> dict[str, dict]:
    attrs = {}
    for name, module in model.named_modules():
        scalars = {
            key: value
            for key, value in vars(module).items()
            if not key.startswith("_") and isinstance(value, SCALARS)
        }
        if scalars:
            attrs[name or "<root>"] = scalars
    return attrs


def dump(repo: str, trust_remote_code: bool) -> dict:
    args = EngineArgs(
        model=repo,
        load_format="dummy",
        enforce_eager=True,
        max_model_len=4096,
        trust_remote_code=trust_remote_code,
    )
    vllm_config = args.create_engine_config()
    model_config = vllm_config.model_config
    with set_current_vllm_config(vllm_config):
        if not model_parallel_is_initialized():
            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            init_distributed_environment(1, 0, f"tcp://127.0.0.1:{port}", 0, "gloo")
            initialize_model_parallel(1, 1)
        with set_default_torch_dtype(model_config.dtype), torch.device("meta"):
            model = initialize_model(vllm_config=vllm_config)
    return {
        "repo": repo,
        "cls": type(model).__name__,
        "cfg": {
            "model_config": {
                "head_size": model_config.get_head_size(),
                "total_num_kv_heads": model_config.get_total_num_kv_heads(),
                "sliding_window": model_config.get_sliding_window(),
                "num_layers": model_config.get_num_layers(vllm_config.parallel_config),
            },
            "params": {n: list(p.shape) for n, p in model.named_parameters()},
            "modules": module_attrs(model),
        },
    }


def main() -> None:
    out, *args = sys.argv[1:]
    trust_remote_code = "--trust-remote-code" in args
    repos = [arg for arg in args if arg != "--trust-remote-code"]
    with open(out, "w") as f:
        for repo in repos:
            try:
                record = dump(repo, trust_remote_code)
            except Exception as e:
                error = f"{type(e).__name__}: {e}".splitlines()[0]
                record = {"repo": repo, "error": error}
            f.write(json.dumps(record, default=str, sort_keys=True) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
