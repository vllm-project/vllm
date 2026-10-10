# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative deployment gate for MonoKernels.

A ``MonoSpec`` says what a model's persistent decode kernel can run. Gating on a
spec is fail-closed for :class:`Feature`: a feature vLLM reports as on and the
spec does not list in ``supports`` refuses the configuration, so a kernel written
before a feature existed refuses it rather than silently miscomputing. The
parametric fields (``dtypes``, ``tp_sizes``, ...) hold no opinion when left
empty. Anything a spec cannot express goes in ``constraints``.

``refuse`` is plain Python over ``VllmConfig`` and is CPU-testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch

from vllm.config import CUDAGraphMode, VllmConfig


class Feature(Enum):
    """A vLLM capability a MonoKernel must opt into to run alongside."""

    EXPERT_PARALLEL = "expert parallelism"
    EXPERT_LOAD_BALANCING = "expert load balancing (EPLB)"
    DATA_PARALLEL = "data parallelism"
    PIPELINE_PARALLEL = "pipeline parallelism"
    DECODE_CONTEXT_PARALLEL = "decode context parallelism"
    SPECULATIVE_DECODE = "speculative decoding"
    LORA = "LoRA"
    KV_TRANSFER = "a KV transfer connector"
    ROUTED_EXPERT_CAPTURE = "routed-expert capture"
    SLEEP_MODE = "sleep mode (weights are reloaded on wake)"


def _parallel(name: str) -> Callable[[VllmConfig], bool]:
    return lambda c: bool(getattr(c.parallel_config, name))


def _aux(name: str) -> Callable[[VllmConfig], bool]:
    return lambda c: bool(getattr(getattr(c, "aux_output_config", None), name, False))


_PROBES: dict[Feature, Callable[[VllmConfig], bool]] = {
    Feature.EXPERT_PARALLEL: _parallel("enable_expert_parallel"),
    Feature.EXPERT_LOAD_BALANCING: _parallel("enable_eplb"),
    Feature.DATA_PARALLEL: lambda c: c.parallel_config.data_parallel_size > 1,
    Feature.PIPELINE_PARALLEL: lambda c: c.parallel_config.pipeline_parallel_size > 1,
    Feature.DECODE_CONTEXT_PARALLEL: (
        lambda c: c.parallel_config.decode_context_parallel_size > 1
    ),
    Feature.SPECULATIVE_DECODE: lambda c: c.speculative_config is not None,
    Feature.LORA: lambda c: c.lora_config is not None,
    Feature.KV_TRANSFER: lambda c: c.kv_transfer_config is not None,
    Feature.ROUTED_EXPERT_CAPTURE: _aux("enable_return_routed_experts"),
    Feature.SLEEP_MODE: lambda c: bool(getattr(c.model_config, "enable_sleep_mode", 0)),
}


@dataclass(frozen=True)
class MonoSpec:
    """What a MonoKernel can run.

    Attributes:
        name: Human-readable kernel name, used in every refusal message.
        architectures: ``hf_config.model_type`` values, empty for any.
        supports: Features the kernel is known to run alongside. Every other
            feature vLLM reports as on refuses the configuration.
        dtypes: Model dtypes, empty for any.
        kv_cache_dtypes: ``cache_config.cache_dtype`` values, empty for any.
        tp_sizes: Tensor-parallel sizes, empty for any.
        cdna_versions: ROCm CDNA ISA versions, empty for any (and for non-ROCm).
        min_compute_units: Compute units the persistent launch needs resident.
        graph_modes: CUDA-graph runtime modes a step may be taken in.
        widths: Step widths the kernel builds, ascending. A step's rows are
            padded up to the smallest width that fits.
        constraints: Model-specific gates, each returning a refusal or None.
        opt_in: The switch that turns the kernel on, usually an env var. When it
            returns False the kernel is simply off: no refusal, nothing logged.
        on_refusal: ``"raise"`` when the kernel has already committed resources
            by the time it could refuse, ``"degrade"`` to log and run vLLM. A
            kernel behind an ``opt_in`` switch usually raises: the user asked
            for it, so silently running something else is the worse answer.

    """

    name: str
    architectures: tuple[str, ...] = ()
    supports: frozenset[Feature] = frozenset()
    dtypes: tuple[torch.dtype, ...] = ()
    kv_cache_dtypes: tuple[str, ...] = ()
    tp_sizes: tuple[int, ...] = ()
    cdna_versions: tuple[int, ...] = ()
    min_compute_units: int = 0
    graph_modes: tuple[CUDAGraphMode, ...] = (CUDAGraphMode.NONE, CUDAGraphMode.FULL)
    widths: tuple[int, ...] = ()
    constraints: tuple[Callable[[VllmConfig], str | None], ...] = field(default=())
    opt_in: Callable[[VllmConfig], bool] | None = None
    on_refusal: str = "degrade"

    def wanted(self, vllm_config: VllmConfig) -> bool:
        return self.opt_in is None or bool(self.opt_in(vllm_config))

    def refuse(self, vllm_config: VllmConfig) -> list[str]:
        """Every reason this configuration cannot run the kernel.

        Args:
            vllm_config: The engine's ``VllmConfig``.

        Returns:
            Refusal reasons, empty when the configuration is supported.

        """
        why = [
            f"{feature.value} is on and {self.name} does not support it"
            for feature, probe in _PROBES.items()
            if feature not in self.supports and _probe(probe, vllm_config)
        ]
        mc, pc = vllm_config.model_config, vllm_config.parallel_config
        arch = getattr(mc.hf_config, "model_type", None)
        if self.architectures and arch not in self.architectures:
            why.append(f"model_type {arch} (want {'/'.join(self.architectures)})")
        if self.dtypes and mc.dtype not in self.dtypes:
            why.append(f"dtype {mc.dtype} (want {'/'.join(map(str, self.dtypes))})")
        kv = vllm_config.cache_config.cache_dtype
        if self.kv_cache_dtypes and kv not in self.kv_cache_dtypes:
            why.append(f"kv cache dtype {kv} (want {'/'.join(self.kv_cache_dtypes)})")
        if self.tp_sizes and pc.tensor_parallel_size not in self.tp_sizes:
            sizes = "/".join(map(str, self.tp_sizes))
            why.append(f"tensor parallel size {pc.tensor_parallel_size} (want {sizes})")
        why += self._platform()
        why += [r for c in self.constraints if (r := c(vllm_config))]
        return why

    def _platform(self) -> list[str]:
        if not self.cdna_versions and not self.min_compute_units:
            return []
        try:
            from vllm.platforms import current_platform
            from vllm.platforms.rocm import get_cdna_version

            cdna = get_cdna_version()
            units = current_platform.num_compute_units(
                torch.accelerator.current_device_index()
            )
        except Exception as e:  # noqa: BLE001
            return [f"platform check failed: {e!r}"]
        why = []
        if self.cdna_versions and cdna not in self.cdna_versions:
            want = "/".join(f"CDNA{v}" for v in self.cdna_versions)
            why.append(f"CDNA version {cdna} (want {want})")
        if units < self.min_compute_units:
            why.append(
                f"{units} compute units, the resident launch needs "
                f"{self.min_compute_units}"
            )
        return why

    def width_for(self, rows: int) -> int | None:
        """The smallest built width that fits ``rows``, or None."""
        return min((w for w in self.widths if w >= rows), default=None)

    def graph_mode_ok(self, mode) -> bool:
        return mode in self.graph_modes


def _probe(probe: Callable[[VllmConfig], bool], vllm_config: VllmConfig) -> bool:
    try:
        return probe(vllm_config)
    except Exception:  # noqa: BLE001
        return False
