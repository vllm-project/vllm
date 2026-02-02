#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reproduce V1 subprocess weight-loading failure for non-checkpoint params.

This script demonstrates the core failure mode seen in V1 multiprocessing:

- vLLM's DefaultModelLoader computes the set of model parameters that should be
  initialized from the checkpoint.
- If a model registers additional parameters that do not exist in the
  checkpoint, and the model reports loaded-weights tracking, vLLM raises:

  ValueError: Following weights were not initialized from checkpoint: {...}

We reproduce it in both:
- current process
- a spawned subprocess (mirrors V1 mp behavior)

Run:
  /root/learning/vllm/.venv/bin/python repro_v1_missing_weights.py

Expected output:
  PASS (current process): ...
  PASS (spawn subprocess): ...
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import torch
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@dataclass
class _StubModelConfig:
    model: str = "unused"
    revision: str | None = None
    quantization: str | None = None


class _FakeLoader(DefaultModelLoader):
    """Override weight sourcing so the repro does not depend on real files."""

    def get_all_weights(self, model_config: _StubModelConfig, model: nn.Module):
        # Only provide a weight for `linear.weight`, leaving `extra.weight`
        # uninitialized.
        del model_config
        del model
        yield ("linear.weight", torch.ones((2, 2), dtype=torch.float16))


class _ModelWithExtraParam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.extra = nn.Linear(2, 2, bias=False)

    def load_weights(self, weights_iter):
        # Mimic vLLM's tracked-loading contract: return a set of parameter names
        # that were initialized from the checkpoint.
        loaded: set[str] = set()
        for name, tensor in weights_iter:
            if name == "linear.weight":
                self.linear.weight.data.copy_(tensor)
                loaded.add(name)
        return loaded


def _run_once() -> None:
    loader = _FakeLoader(load_config=LoadConfig())
    model = _ModelWithExtraParam()

    try:
        loader.load_weights(model, _StubModelConfig())
    except ValueError as e:
        if "Following weights were not initialized" not in str(e):
            raise
        return

    raise RuntimeError("Expected ValueError was not raised")


def _run_in_subprocess(queue: mp.Queue) -> None:
    try:
        _run_once()
    except Exception as e:  # noqa: BLE001
        queue.put(repr(e))
    else:
        queue.put(None)


def main() -> None:
    _run_once()
    print("PASS (current process): missing-weights error reproduced")

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    proc = ctx.Process(target=_run_in_subprocess, args=(queue,))
    proc.start()
    proc.join(timeout=30)

    if proc.exitcode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {proc.exitcode}")

    err = queue.get(timeout=5)
    if err is not None:
        raise RuntimeError(f"Subprocess raised unexpected error: {err}")

    print("PASS (spawn subprocess): missing-weights error reproduced")


if __name__ == "__main__":
    main()
