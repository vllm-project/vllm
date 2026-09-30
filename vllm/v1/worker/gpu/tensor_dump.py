# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils import _pytree as pytree

from vllm.distributed.parallel_state import get_world_group
from vllm.v1.worker.gpu.input_batch import InputBatch


class TensorDumper:
    def __init__(
        self, model: nn.Module, output_folder: str, layers: list[int] | None
    ) -> None:
        self.output_folder = Path(output_folder) / f"rank{get_world_group().rank}"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self._record: dict[str, Any] | None = None
        self._num_tokens = 0
        self._pass_id = 0

        selected_layers = set(layers) if layers is not None else None
        modules = {}
        for name, module in model.named_modules():
            parts = name.rsplit(".", 2)
            if (
                len(parts) >= 2
                and parts[-2] == "layers"
                and parts[-1].isdigit()
                and (selected_layers is None or int(parts[-1]) in selected_layers)
            ):
                modules[name] = module

        if selected_layers is not None:
            matched_layers = {int(name.rsplit(".", 1)[-1]) for name in modules}
            if missing := selected_layers - matched_layers:
                raise ValueError(
                    f"Model does not contain decoder layers {sorted(missing)}"
                )
        for name, module in modules.items():
            module.register_forward_hook(partial(self._capture, name))
        model.register_forward_hook(self._save)

    def prepare_forward(self, batch: InputBatch) -> None:
        self._num_tokens = batch.num_tokens
        prefix = "vllm.forward_batch_info."
        self._record = {
            prefix + "input_ids": batch.input_ids[: self._num_tokens].cpu(),
            prefix + "positions": batch.positions[: self._num_tokens].cpu(),
            prefix + "rids": batch.req_ids,
            prefix + "extend_seq_lens": torch.from_numpy(
                batch.num_scheduled_tokens.copy()
            ),
        }

    def _capture(
        self, name: str, _module: nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        if self._record is not None:
            self._record[name] = pytree.tree_map_only(
                torch.Tensor, self._to_cpu, output
            )

    def _to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim and tensor.shape[0] >= self._num_tokens:
            tensor = tensor[: self._num_tokens]
        return tensor.detach().cpu()

    def _save(self, _module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        if self._record is None:
            return
        torch.save(self._record, self.output_folder / f"Pass{self._pass_id}.pt")
        self._pass_id += 1
        self._record = None
