# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capture eager MRV2 module inputs, outputs, and batch metadata for debugging."""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict
from fnmatch import fnmatchcase
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from torch.utils._pytree import tree_map_only

from vllm.config.tensor_dump import TensorDumpConfig, TensorDumpOptions
from vllm.distributed import parallel_state
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


_current_dumper: ContextVar["TensorDumper | None"] = ContextVar(
    "tensor_dumper", default=None
)


def dump(name: str, value: Any) -> None:
    """Save tensors during an active model forward, subject to capture filters."""
    if (dumper := _current_dumper.get()) is not None:
        dumper.dump(name, value)


def _tensors(value: Any, path: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, IntermediateTensors):
        yield from _tensors(value.tensors, path)
    elif isinstance(value, (dict, tuple, list)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            yield from _tensors(item, f"{path}.{key}")


def _parallel_groups() -> dict:
    if not parallel_state.model_parallel_is_initialized():
        return {}
    groups = {
        "tp": parallel_state.get_tp_group(),
        "pp": parallel_state.get_pp_group(),
        "dp": parallel_state.get_dp_group(),
        "dcp": parallel_state.get_dcp_group(),
        "pcp": parallel_state.get_pcp_group(),
    }
    with suppress(AssertionError):
        groups["ep"] = parallel_state.get_ep_group()
    return {
        name: {"rank": group.rank_in_group, "ranks": group.ranks}
        for name, group in groups.items()
    }


class TensorDumper:
    def __init__(self, config: TensorDumpConfig, model: torch.nn.Module):
        self.config = config
        self.rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        assert config.output_dir is not None
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self.directory = Path(
            tempfile.mkdtemp(
                prefix=f"rank{self.rank}-pid{os.getpid()}-", dir=config.output_dir
            )
        )
        self.step = 0
        self.forward_step = 0
        self._start_step = 0
        self._index = 0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for name, module in model.named_modules():
            name = name or "<root>"
            self._handles.extend(
                (
                    module.register_forward_pre_hook(
                        partial(self._inputs, name), with_kwargs=True
                    ),
                    module.register_forward_hook(partial(self._output, name)),
                )
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def capture(self, batch: "InputBatch") -> Iterator[None]:
        forward_step = self.forward_step
        self.forward_step += 1
        phase = (
            "prefill"
            if batch.is_prefilling_np.all()
            else "mixed"
            if batch.has_prefill
            else "decode"
        )
        if (
            not self.config.enabled
            or forward_step < self.config.start_step
            or self.step - self._start_step >= self.config.max_steps
            or (self.config.ranks is not None and self.rank not in self.config.ranks)
            or (self.config.phase != "all" and phase != self.config.phase)
        ):
            yield
            return
        token = _current_dumper.set(self)
        try:
            self._save(
                "batch",
                {
                    "parallel_groups": _parallel_groups(),
                    "phase": phase,
                    "forward_step": forward_step,
                    "is_prefilling": batch.is_prefilling_np.tolist(),
                    "request_ids": list(batch.req_ids),
                    "input_ids": batch.input_ids[: batch.num_tokens],
                    "positions": batch.positions[: batch.num_tokens],
                    "query_start_loc": batch.query_start_loc[: batch.num_reqs + 1],
                    "seq_lens": batch.seq_lens[: batch.num_reqs],
                    "num_tokens": batch.num_tokens,
                    "num_tokens_after_padding": batch.num_tokens_after_padding,
                },
            )
            yield
        finally:
            _current_dumper.reset(token)
            self.step += 1

    def dump(self, name: str, value: Any) -> None:
        if _current_dumper.get() is not self:
            return
        if self.config.layers is not None:
            layer = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if layer is None or int(layer[1]) not in self.config.layers:
                return
        for suffix, tensor in _tensors(value):
            tensor_name = name + suffix
            if any(
                fnmatchcase(tensor_name, pattern) for pattern in self.config.patterns
            ):
                self._save(tensor_name, tensor)

    def _save(self, name: str, value: Any) -> None:
        value = tree_map_only(
            torch.Tensor,
            lambda tensor: tensor.detach().to(device="cpu", copy=True),
            value,
        )
        path = self.directory / f"{self._index:08d}.pt"
        self._index += 1
        torch.save(
            {"name": name, "step": self.step, "rank": self.rank, "value": value}, path
        )

    def _inputs(self, name, module, args, kwargs):
        self.dump(name + ".input", {"args": args, "kwargs": kwargs})

    def _output(self, name, module, args, output):
        self.dump(name + ".output", output)

    def control(self, action: str, config: dict | None = None) -> dict:
        if action == "configure":
            options = TensorDumpOptions(**(config or {}))
            self.config = TensorDumpConfig(
                output_dir=self.config.output_dir, **asdict(options)
            )
        elif action == "reset":
            self._start_step = self.step
            self.forward_step = 0
        elif action != "get_state":
            raise ValueError(f"Unknown tensor dump action: {action}")
        return {
            "config": asdict(self.config),
            "step": self.step,
            "forward_step": self.forward_step,
            "captured_steps": self.step - self._start_step,
            "rank": self.rank,
            "directory": str(self.directory),
        }
