# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from fnmatch import fnmatchcase

import torch


class FrozenWeights:
    """Retain explicitly frozen parameters across level-2 sleep.

    Patterns name runtime modules, not checkpoint tensors. Configure after initial
    loading. Frozen modules must keep their parameter objects and storage intact.
    """

    def __init__(self, model: torch.nn.Module, patterns: list[str]):
        modules = dict(model.named_modules())
        selected: set[str] = set()
        for pattern in patterns:
            matches = {name for name in modules if fnmatchcase(name, pattern)}
            if not matches:
                raise ValueError(
                    f"Frozen weight module pattern matched nothing: {pattern}"
                )
            if "" in matches:
                raise ValueError("Select frozen submodules, not the entire model")
            selected.update(matches)
        names = {
            name
            for name in modules
            if any(
                not root or name == root or name.startswith(root + ".")
                for root in selected
            )
        }
        self.model = model
        self.parameters = {
            name: param
            for name, param in model.named_parameters(remove_duplicate=False)
            if name.rpartition(".")[0] in names
        }
        if patterns and not self.parameters:
            raise ValueError("Frozen weight modules contain no parameters")
        frozen_storage = {
            (param.device, param.untyped_storage().data_ptr())
            for param in self.parameters.values()
            if param.numel()
        }
        for name, param in model.named_parameters(remove_duplicate=False):
            if (
                name not in self.parameters
                and param.numel()
                and (param.device, param.untyped_storage().data_ptr()) in frozen_storage
            ):
                raise ValueError(f"Frozen storage aliases mutable parameter: {name}")
        for name, buffer in model.named_buffers(remove_duplicate=False):
            if (
                name.rpartition(".")[0] not in names
                and buffer.numel()
                and (buffer.device, buffer.untyped_storage().data_ptr())
                in frozen_storage
            ):
                raise ValueError(f"Frozen storage aliases mutable buffer: {name}")
        self.backups: dict[str, torch.Tensor] = {}
        self._needs_restore = False
        self.layouts = {
            name: (param.data_ptr(), param.shape, param.stride(), param.dtype)
            for name, param in self.parameters.items()
        }

    def _validate_storage(self) -> None:
        for name, param in self.parameters.items():
            if self.model.get_parameter(name) is not param or self.layouts[name] != (
                param.data_ptr(),
                param.shape,
                param.stride(),
                param.dtype,
            ):
                raise RuntimeError(f"Frozen parameter storage was replaced: {name}")

    @torch.no_grad()
    def save(self) -> None:
        self._validate_storage()
        for name, param in self.parameters.items():
            if param.device.type != "cpu" and name not in self.backups:
                self.backups[name] = param.to(device="cpu", copy=True)
        self._needs_restore = True

    @torch.no_grad()
    def restore(self) -> None:
        if not self._needs_restore:
            return
        self._validate_storage()
        for name, backup in self.backups.items():
            self.parameters[name].copy_(backup)
        self._needs_restore = False
