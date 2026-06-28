# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug tensor dump for capturing intermediate activations.

Registers forward hooks on every leaf module of the model to capture
intermediate activations during inference. Each forward pass produces a
``.pt`` file containing all recorded tensors.

Usage::

    VLLM_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER=./dump \\
        python -m vllm.entrypoints.openai.api_server --model <model>

Environment variables:

* ``VLLM_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER`` -- Output directory for dumps.
  When unset the feature is disabled.
* ``VLLM_DEBUG_TENSOR_DUMP_LAYERS`` -- Comma-separated layer indices to
  dump (e.g. ``"0,1,31"``). When unset all layers are dumped.
* ``VLLM_DEBUG_TENSOR_DUMP_SKIP_PASSES`` -- Number of initial forward
  passes to skip (useful for skipping warmup). Default ``0``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from vllm.config import ParallelConfig, VllmConfig

logger = logging.getLogger(__name__)


class TensorDumper:
    """Collects tensors from forward hooks and saves them per pass.

    Args:
        dump_dir: Base directory for output files.
        dump_layers: Optional list of layer indices to capture. When
            ``None`` all layers are captured.
        tp_size: Tensor-parallel world size.
        tp_rank: Tensor-parallel rank of this worker.
        pp_rank: Pipeline-parallel rank of this worker.
        skip_passes: Number of leading forward passes to skip before
            recording (useful for warmup).
    """

    def __init__(
        self,
        dump_dir: str,
        dump_layers: list[int] | None,
        tp_size: int,
        tp_rank: int,
        pp_rank: int,
        skip_passes: int = 0,
    ) -> None:
        self._dump_layers = dump_layers
        self._forward_pass_id = 0
        self._skip_passes = skip_passes
        self._pid = os.getpid()
        self._current_tensors: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        self._base_dir = Path(dump_dir)
        rank = tp_size * pp_rank + tp_rank
        self._process_dir = (
            self._base_dir / f"TP{tp_rank}_PP{pp_rank}_Rank{rank}_pid{self._pid}"
        )
        self._process_dir.mkdir(parents=True, exist_ok=True)

    def get_dump_dir(self) -> str:
        """Return the process-specific dump directory path."""
        return str(self._process_dir)

    def add_tensor(self, name: str, tensor_item: torch.Tensor | tuple | list) -> None:
        """Record a tensor (or tuple/list of tensors) under *name*.

        Tensors are moved to CPU immediately to avoid holding GPU memory.
        If *skip_passes* has not yet been exhausted the call is a no-op.
        """
        if self._skip_passes > 0:
            return
        if isinstance(tensor_item, (tuple, list)):
            tensors = [t.cpu() for t in tensor_item if isinstance(t, torch.Tensor)]
            if len(tensors) == 1:
                self._current_tensors[name] = tensors[0]
            elif len(tensors) > 1:
                self._current_tensors[name] = tensors
        elif isinstance(tensor_item, torch.Tensor):
            self._current_tensors[name] = tensor_item.cpu()
        else:
            logger.warning("Unsupported type: %s: %s", type(tensor_item), tensor_item)

    def dump_current_tensors(self) -> None:
        """Flush all accumulated tensors to a ``.pt`` file.

        During the skip phase the tensors are silently discarded.
        """
        if self._skip_passes > 0:
            self._skip_passes -= 1
            self._current_tensors = {}
            logger.info("Skipping warmup pass (remaining: %d)", self._skip_passes)
            return
        if len(self._current_tensors) == 0:
            return
        tensor_file = self._process_dir / f"Pass{self._forward_pass_id:05d}.pt"
        logger.info("Dump %05dth pass to %s", self._forward_pass_id, tensor_file)
        torch.save(self._current_tensors, str(tensor_file))
        self._current_tensors = {}
        self._forward_pass_id += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_hook_recursive(
        self,
        model: nn.Module,
        prefix: str,
        top_level_module_name: str,
        layers_module_name: str,
    ) -> tuple[bool, int]:
        """Walk *model* and attach forward hooks to leaf modules.

        Returns:
            A tuple ``(top_level_matched, child_count)``.
        """
        model_top_level_module_matched = False
        layers_prefix = top_level_module_name + "." + layers_module_name
        for name, module in model._modules.items():
            top_level_model = False
            if len(prefix) == 0:
                cur_name = name
                if cur_name == top_level_module_name:
                    model_top_level_module_matched = True
                    top_level_model = True
            else:
                cur_name = prefix + "." + name
            if (
                self._dump_layers is not None
                and name.isdigit()
                and prefix == layers_prefix
            ):
                cur_layer = int(name)
                if cur_layer not in self._dump_layers:
                    continue
            if module is not None:
                _, sub_count = self._add_hook_recursive(
                    module,
                    cur_name,
                    top_level_module_name,
                    layers_module_name,
                )
                if sub_count == 0 or top_level_model:
                    module.register_forward_hook(
                        self._dump_hook(cur_name, top_level_model)
                    )
        return model_top_level_module_matched, len(model._modules.items())

    def _dump_hook(self, tensor_name: str, do_dump: bool):
        """Return a forward-hook closure for *tensor_name*."""

        def inner_dump_hook(module, input, output):  # noqa: A002
            if do_dump:
                # Top-level model hook: extract input_ids and positions
                # from the model's forward() arguments.
                # vLLM LlamaModel.forward(input_ids, positions, ...)
                if isinstance(input, (tuple, list)) and len(input) >= 2:
                    if isinstance(input[0], torch.Tensor):
                        self.add_tensor(
                            tensor_name + ".forward_batch_info.input_ids",
                            input[0],
                        )
                    if isinstance(input[1], torch.Tensor):
                        self.add_tensor(
                            tensor_name + ".forward_batch_info.positions",
                            input[1],
                        )
                self.dump_current_tensors()
            if output is not None:
                self.add_tensor(tensor_name, output)

        return inner_dump_hook


def register_forward_hook_for_model(
    model: nn.Module,
    dump_dir: str,
    dump_layers: list[int] | None = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    pp_rank: int = 0,
    skip_passes: int = 0,
) -> TensorDumper:
    """Register forward hooks on *model* and return the ``TensorDumper``.

    Args:
        model: The model to instrument.
        dump_dir: Base output directory.
        dump_layers: Optional layer indices to capture.
        tp_size: Tensor-parallel world size.
        tp_rank: Tensor-parallel rank.
        pp_rank: Pipeline-parallel rank.
        skip_passes: Forward passes to skip before recording.

    Returns:
        The :class:`TensorDumper` instance managing the hooks.
    """
    tensor_dumper = TensorDumper(
        dump_dir,
        dump_layers,
        tp_size,
        tp_rank,
        pp_rank,
        skip_passes,
    )
    import vllm.envs as envs

    top_level_module_name = envs.VLLM_DEBUG_TENSOR_DUMP_TOP_LEVEL_MODULE_NAME
    layers_module_name = envs.VLLM_DEBUG_TENSOR_DUMP_LAYERS_MODULE_NAME
    model_top_level_module_matched, _ = tensor_dumper._add_hook_recursive(
        model,
        "",
        top_level_module_name,
        layers_module_name,
    )
    assert model_top_level_module_matched, (
        f"model should have a module named {top_level_module_name}"
    )
    logger.info(
        "Tensor dump hooks registered. Output dir: %s", tensor_dumper.get_dump_dir()
    )
    return tensor_dumper


def maybe_setup_tensor_dump(
    model: nn.Module,
    vllm_config: VllmConfig,
    parallel_config: ParallelConfig,
) -> TensorDumper | None:
    """Set up tensor dumping if enabled via environment variables.

    This is the main entry point called from the model runner during model
    loading.  When ``VLLM_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER`` is not set,
    this function returns ``None`` and is effectively a no-op.

    The function:

    1. Disables ``torch.compile`` and CUDA graph capture (both are
       incompatible with the ``.cpu()`` calls and ``torch.save`` inside
       the forward hooks).
    2. Removes custom ``__call__`` methods injected by
       ``@support_torch_compile`` so that ``nn.Module.__call__`` (which
       fires forward hooks) is used instead.
    3. Registers forward hooks on every leaf module of *model*.

    Args:
        model: The loaded model.
        vllm_config: The global vLLM configuration object.
        parallel_config: Parallel (TP/PP) configuration.

    Returns:
        The :class:`TensorDumper` instance, or ``None`` if the feature is
        disabled.
    """
    import vllm.envs as envs

    dump_dir = envs.VLLM_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER
    if dump_dir is None:
        return None

    # --- Disable torch.compile and CUDAGraph ---------------------------
    from vllm.config import CompilationMode, CUDAGraphMode

    vllm_config.compilation_config.mode = CompilationMode.NONE
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    for _, mod in model.named_modules():
        if hasattr(mod, "do_not_compile"):
            mod.do_not_compile = True

    # --- Remove custom __call__ from @support_torch_compile ------------
    # @support_torch_compile replaces cls.__call__ which bypasses
    # nn.Module.__call__() where forward hooks are fired.
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    patched_classes: set[type] = set()
    for _, mod in model.named_modules():
        cls = type(mod)
        if (
            cls not in patched_classes
            and issubclass(cls, TorchCompileWithNoGuardsWrapper)
            and "__call__" in cls.__dict__
        ):
            delattr(cls, "__call__")
            patched_classes.add(cls)

    logger.info("Tensor dump enabled: torch.compile and CUDAGraph disabled.")
    logger.warning(
        "Tensor dump patches __call__ at the class level. "
        "This is a global, irreversible change within the process and is "
        "NOT safe for multi-model serving."
    )

    # --- Parse env vars and register hooks -----------------------------
    from vllm.distributed import get_pp_group, get_tp_group

    tp_size = parallel_config.tensor_parallel_size
    tp_rank = get_tp_group().rank_in_group
    pp_rank = get_pp_group().rank_in_group

    dump_layers_str = envs.VLLM_DEBUG_TENSOR_DUMP_LAYERS
    dump_layers: list[int] | None = None
    if dump_layers_str:
        dump_layers = [int(x) for x in dump_layers_str.split(",")]

    skip_passes = envs.VLLM_DEBUG_TENSOR_DUMP_SKIP_PASSES

    return register_forward_hook_for_model(
        model,
        dump_dir=dump_dir,
        dump_layers=dump_layers,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        skip_passes=skip_passes,
    )
