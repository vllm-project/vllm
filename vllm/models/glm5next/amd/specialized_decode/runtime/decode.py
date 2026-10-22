# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash specialized decode (layer mode): one routing decision per step,
then a whole-layer MonoKernel launch for every MoE decoder layer 3..43 inside the
stock layer loop.

Built by the model state once the weights are loaded (``Glm5NextModelState``): reads
vLLM's loaded weights in place (``weights``) and builds every width's launches
before memory profiling and graph capture. ``bind_caches`` binds them to every KV
cache vLLM allocates (the CUDA-graph memory-profiling cache, then the real one).
Each step ``plan`` returns a ``Step`` or None; ``Step.layer(i)`` is the launch
replacing decoder layer i (None for vLLM's layers). Weight reloads are not
supported."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from . import layers
from .context import DecodeContext, DecodeStep
from .failstop import OUTPUT_RANK, HealthWatch, counters
from .guards import model_refusals
from .routing import step_reason
from .spec import STAGES, DecodeConfig, kernel_layers

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

logger = init_logger(__name__)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _resolve(module: nn.Module, path: str) -> torch.Tensor:
    attrs, _, index = path.partition("[")
    value = operator.attrgetter(attrs)(module)
    if index:
        value = value[int(index.rstrip("]"))]
    elif isinstance(value, (list, tuple)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} is not a tensor")
    return value


@dataclass(frozen=True)
class Step:
    """A planned kernel step and its launch per kernel layer."""

    step: DecodeStep
    kernels: dict[int, Any]

    def layer(self, idx: int) -> Callable[..., Any] | None:
        """The launch replacing decoder layer idx (it takes and returns what
        ``Glm5NextDecoderLayer`` does), or None for vLLM's layer."""
        kernel = self.kernels.get(idx)
        if kernel is None:
            return None
        return partial(_launch, kernel, self.step)


def _launch(kernel: Any, step: DecodeStep, positions: torch.Tensor, *args: Any):
    return kernel.launch(step, *args)


class SpecializedDecode:
    """The executor: packs the kernel layers' weights, builds every width's launches,
    binds them to each KV cache and plans every step."""

    def __init__(self, backbone: nn.Module, cfg: DecodeConfig) -> None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )
        from vllm.model_executor.models.utils import PPMissingLayer

        reasons = model_refusals(backbone)
        if reasons:
            raise ValueError(
                "--monokernel cannot run the GLM-5.3-Flash specialized decode on "
                "this model: " + "; ".join(reasons) + "."
            )
        self.cfg = cfg
        start = getattr(backbone, "start_layer", 0)
        end = getattr(backbone, "end_layer", len(backbone.layers))
        self.layers: dict[int, nn.Module] = {
            i: backbone.layers[i]
            for i in kernel_layers(backbone.config)
            if start <= i < end and not isinstance(backbone.layers[i], PPMissingLayer)
        }
        vllm_config = backbone.vllm_config
        self.ctx = DecodeContext(
            vllm_config=vllm_config,
            tp_size=get_tensor_model_parallel_world_size(),
            tp_rank=get_tensor_model_parallel_rank(),
            cpu_group=get_tp_group().cpu_group,
            device=torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu"),
        )
        self.ctx.state["poll_limit"] = cfg.poll_limit
        self.packed = layers.pack(backbone, self.ctx)
        self.metadata_names = layers.metadata_names(self.layers)
        self.kernels = {
            width: {
                idx: layers.build(self.ctx, self.packed, layer, idx, width)
                for idx, layer in self.layers.items()
            }
            for width in cfg.widths
        }
        words, seen = [], set()
        for per_width in self.kernels.values():
            for kernel in per_width.values():
                w = kernel.health()
                if w.data_ptr() not in seen:
                    seen.add(w.data_ptr())
                    words.append(w)
        flags = self.ctx.state.get("flags") if self.ctx.tp_rank == OUTPUT_RANK else None
        self.watch = HealthWatch(
            "GLM-5.3-Flash", words, STAGES, self.ctx.tp_rank, flags
        )
        self._cache_key: tuple | None = None
        logger.info_once(
            "GLM-5.3-Flash specialized decode: %d MonoKernel layers, widths %s, "
            "step tokens %s; numerics: %s.",
            len(self.layers),
            "/".join(map(str, cfg.widths)),
            "/".join(map(str, cfg.tokens)),
            "; ".join(cfg.numerics),
        )
        mode = vllm_config.compilation_config.cudagraph_mode
        if mode is None or mode == CUDAGraphMode.NONE:
            logger.warning("GLM-5.3-Flash MonoKernel running eagerly: debugging only.")

    def caches(self) -> dict[str, torch.Tensor]:
        out = {}
        for idx, layer in self.layers.items():
            for name, (path, _, when) in layers.CACHES.items():
                if when(layer):
                    out[f"{idx}.{name}"] = _resolve(layer, path)
        return out

    def bind_caches(self) -> None:
        """Check and bind the KV caches vLLM just bound (the CUDA-graph memory
        profiling cache, then the real one); a no-op if none moved."""
        caches = self.caches()
        key = tuple((k, t.data_ptr(), t.numel()) for k, t in caches.items())
        if key == self._cache_key:
            return
        if _capturing():
            raise RuntimeError(
                "GLM-5.3-Flash MonoKernel caches moved inside CUDA graph capture"
            )
        reasons = []
        for k, t in caches.items():
            dtypes = layers.CACHES[k.split(".", 1)[1]][1]
            if t.dtype not in dtypes:
                reasons.append(f"{k}: dtype {t.dtype}")
            if t.dim() and t.stride(-1) != 1:
                reasons.append(f"{k}: last dim not contiguous")
        reasons += layers.bind(self.ctx, caches)
        if reasons:
            raise ValueError(
                "GLM-5.3-Flash MonoKernel cannot use these caches: "
                + "; ".join(reasons)
            )
        self._cache_key = key

    def plan(self, positions: torch.Tensor, intermediate_tensors: Any) -> Step | None:
        """This step's plan, or None to run vLLM's layers: pipeline stage inputs and
        steps ``routing.step_reason`` or the layers' eligibility decline. A kernel
        step re-binds the caches if they moved (a no-op otherwise)."""
        if intermediate_tensors is not None or not is_forward_context_available():
            return None
        fc = get_forward_context()
        tokens = positions.shape[-1]
        reason = step_reason(fc, tokens, self.cfg, self.metadata_names)
        step = None
        if reason is None:
            width = self.cfg.width_for(tokens)
            assert width is not None
            step = DecodeStep(tokens, width, positions, fc.attn_metadata)
            reason = layers.eligible(self.ctx, step)
        if reason is not None:
            key = f"vllm_path:{reason}"
            if not counters[key]:
                logger.debug("GLM-5.3-Flash MonoKernel: vLLM path for %s", reason)
            counters[key] += 1
            return None
        assert step is not None
        self.bind_caches()
        counters["mono_steps"] += 1
        return Step(step, self.kernels[step.width])

    def health(self) -> torch.Tensor:
        return self.watch.health()

    def after_step(self) -> None:
        """Once per executed step, from compute_logits (eager on every rank, also
        under graph replay). Fail-stop on expired kernel waits."""
        self.watch.after_step()
