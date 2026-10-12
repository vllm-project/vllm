# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MonoKernel runtime of RFC #60904, model-agnostic: ``MonoSpec`` and the
step types, ``configure`` (before load), ``attach`` (after load),
``forward_layer`` (the layer loop) with its per-step decision, and ``health``.
Private to Kimi-K3 until ``vllm/models/common/mono/`` exists; a model brings
only its ``MonoSpec``.

A step runs on the kernels only when every rank decides alike: the decision
reads the attention metadata and the batch, never rank-local state. Under a
FULL CUDA graph it is a function of the captured size: FULL graphs serve
uniform decode batches and every width is a capture size, so no replay pads a
request's rows. PIECEWISE steps run vLLM's layers.
"""

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class MonoUnsupported(Exception):
    """The configuration or the loaded model is outside what the kernels serve."""


@dataclass(frozen=True)
class CacheReq:
    group: str
    dtype: str
    layout: str


@dataclass
class MonoContext:
    vllm_config: "VllmConfig"
    rank: int
    device: torch.device
    state: Any = None  # the spec's per-rank launch state (``pack``'s)


@dataclass
class MonoStep:
    num_tokens: int
    width: int
    positions: torch.Tensor
    attn_metadata: Any
    runtime_mode: CUDAGraphMode


class LayerKernel(Protocol):
    # attention layers whose caches ``bind`` checks
    cache_layers: tuple[str, ...]

    def launch(self, step: MonoStep, *args, **kwargs) -> Any: ...

    def health(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class MonoSpec:
    name: str
    layers: Callable[[Any], list[int]]  # decoder layers that get a kernel
    # rows a launch serves, by config (spec-verify kernels: whole requests)
    widths: Callable[["VllmConfig"], tuple[int, ...]]
    supports: frozenset[str]  # features in use but not listed are refused
    caches: Mapping[str, CacheReq]
    weight_source: str  # "views" or "convert"
    options: Mapping[str, Any]
    refuse_config: Callable[["VllmConfig"], list[str]]
    refuse_model: Callable[[Any], list[str]]
    pack: Callable[[Any, MonoContext], Any]
    bind: Callable[[MonoContext, Mapping[str, Any]], list[str]]
    eligible: Callable[[MonoContext, MonoStep], str | None]
    build: Callable[[MonoContext, Any, Any, int], LayerKernel]


def enabled(vllm_config: "VllmConfig") -> bool:
    """``--monokernel`` (``VLLM_ROCM_MONO_DECODE=1`` as on DeepSeek-V4.1)."""
    return vllm_config.kernel_config.monokernel or envs.VLLM_ROCM_MONO_DECODE


def load_spec(path: str) -> MonoSpec:
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def features_in_use(vllm_config: "VllmConfig") -> dict[str, bool]:
    pc = vllm_config.parallel_config
    oc = vllm_config.offload_config
    return {
        "pipeline_parallel": pc.pipeline_parallel_size > 1,
        "data_parallel": pc.data_parallel_size > 1,
        "expert_parallel": pc.enable_expert_parallel,
        "eplb": pc.enable_eplb,
        "dcp": pc.decode_context_parallel_size > 1,
        "pcp": pc.prefill_context_parallel_size > 1,
        "lora": vllm_config.lora_config is not None,
        "kv_connector": vllm_config.kv_transfer_config is not None,
        "routed_expert_capture": (
            vllm_config.aux_output_config.enable_return_routed_experts
        ),
        "speculative_decoding": vllm_config.speculative_config is not None,
        "sleep_mode": vllm_config.model_config.enable_sleep_mode,
        "cpu_weight_offload": (
            oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0
        ),
        "weight_updates": vllm_config.weight_transfer_config is not None,
    }


def refuse_config(spec: MonoSpec, vllm_config: "VllmConfig") -> list[str]:
    """Default-deny, the runner and the graph mode, then the spec's own."""
    reasons = [
        f"{f} is not supported"
        for f, on in features_in_use(vllm_config).items()
        if on and f not in spec.supports
    ]
    if not vllm_config.use_v2_model_runner:
        reasons.append("needs Model Runner V2")
    reasons += spec.refuse_config(vllm_config)
    cc = vllm_config.compilation_config
    mode = cc.cudagraph_mode
    if mode != CUDAGraphMode.NONE:
        if mode.decode_mode() != CUDAGraphMode.FULL:
            reasons.append(f"CUDA graph mode {mode.name}: decode needs FULL graphs")
        elif not reasons:
            sizes = set(cc.cudagraph_capture_sizes or ())
            missing = [w for w in spec.widths(vllm_config) if w not in sizes]
            if missing:
                reasons.append(f"CUDA graph capture sizes miss widths {missing}")
    return reasons


def configure(vllm_config: "VllmConfig", spec_path: str) -> None:
    """Before load: imports only the spec; raises with every refusal reason."""
    spec = load_spec(spec_path)
    reasons = refuse_config(spec, vllm_config)
    if reasons:
        raise MonoUnsupported(f"MonoKernel {spec.name} refused: " + "; ".join(reasons))
    if vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        logger.warning(
            "MonoKernel %s under eager mode: for debugging only, serve with FULL "
            "CUDA graphs",
            spec.name,
        )


def _agree(fn: Callable[[], list[str]], group, what: str) -> None:
    """``fn``'s refusals on this rank, then every rank's verdict agreed: every
    TP rank must decide alike before anything collective follows."""
    reasons = fn()
    flag = torch.tensor([int(not reasons)], dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
    if reasons:
        raise MonoUnsupported(f"{what}: " + "; ".join(reasons))
    if not flag.item():
        raise MonoUnsupported(f"{what}: another TP rank refused")


class Mono:
    def __init__(self, spec: MonoSpec, model, vllm_config: "VllmConfig"):
        from vllm.distributed import get_tp_group

        tp = get_tp_group()
        self.spec = spec
        _agree(
            lambda: spec.refuse_model(model),
            tp.cpu_group,
            f"MonoKernel {spec.name} refused for the loaded model",
        )
        self.ctx = MonoContext(
            vllm_config, tp.rank_in_group, next(model.parameters()).device
        )
        packed = spec.pack(model, self.ctx)
        self.widths = spec.widths(vllm_config)
        self.kernels: dict[tuple[int, int], LayerKernel] = {}
        for i in spec.layers(model.config):
            if model.start_layer <= i < model.end_layer:
                for w in self.widths:
                    self.kernels[i, w] = spec.build(
                        self.ctx, packed, model.layers[i], w
                    )
        self.cache_layers = sorted(
            {n for k in self.kernels.values() for n in k.cache_layers}
        )
        self._fc: Any = None
        self._bound: tuple | None = None
        self.step: MonoStep | None = None
        logger.info(
            "MonoKernel %s: %d layers, widths %s",
            spec.name,
            len({i for i, _ in self.kernels}),
            list(self.widths),
        )

    def _bind(self) -> None:
        """Re-checks the caches whenever they moved: memory profiling runs real
        steps on throwaway caches before the real ones exist."""
        sfc = self.ctx.vllm_config.compilation_config.static_forward_context
        caches = {n: sfc[n].kv_cache for n in self.cache_layers}
        key = tuple(
            (t.data_ptr(), t.numel())
            for c in caches.values()
            for t in (c if isinstance(c, (list, tuple)) else (c,))
        )
        if key == self._bound:
            return
        reasons = self.spec.bind(self.ctx, caches)
        if reasons:
            raise MonoUnsupported(
                f"MonoKernel {self.spec.name} caches: " + "; ".join(reasons)
            )
        self._bound = key

    def begin_step(self, positions: torch.Tensor) -> str | None:
        """Decides this step for every kernel layer: None runs the kernels (and
        sets ``step``), else the reason it runs vLLM's layers."""
        self.step = None
        fc = get_forward_context()
        if fc.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return "piecewise"
        n = positions.size(-1)
        if n not in self.widths:
            return "width"
        step = MonoStep(n, n, positions, fc.attn_metadata, fc.cudagraph_runtime_mode)
        why = self.spec.eligible(self.ctx, step)
        if why is None:
            self._bind()
            self.step = step
        return why

    def health(self) -> torch.Tensor:
        return next(iter(self.kernels.values())).health()


def attach(model, vllm_config: "VllmConfig", spec_path: str) -> Mono | None:
    """After load; None with the switch off."""
    if not enabled(vllm_config):
        return None
    return Mono(load_spec(spec_path), model, vllm_config)


def forward_layer(mono: Mono | None, i: int, layer, positions, *args, **kwargs):
    """Layer ``i`` on its kernel when this step runs on the kernels, else the
    vLLM layer unchanged. The first call under a forward context decides."""
    if mono is not None:
        fc = get_forward_context()
        if fc is not mono._fc:
            mono._fc = fc
            mono.begin_step(positions)
        step = mono.step
        if step is not None and (k := mono.kernels.get((i, step.width))) is not None:
            return k.launch(step, positions, *args, **kwargs)
    return layer(positions, *args, **kwargs)


def health(mono: Mono | None) -> torch.Tensor | None:
    """The step's error words, read by the runner after each step: any set
    word fails the step before its tokens are returned."""
    return None if mono is None else mono.health()
