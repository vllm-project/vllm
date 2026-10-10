# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-integrated MonoKernel dispatch.

``Glm5MonoDecode.create`` (after weight loading, ``VLLM_ROCM_USE_GLM5_MONOKERNEL=1``)
builds kernels, buffers and IPC peers before memory profiling and graph capture.
``forward_layer`` takes the per-step go / no-go decision at the first mono layer and
runs either the library's ``torch.ops.vllm.mono_layer`` or vLLM's decoder layer. Config:
``LiveConfig`` defaults overridden by ``VLLM_ROCM_GLM5_MONOKERNEL_CONFIG`` (JSON);
``ckpt`` defaults to vLLM's local checkpoint dir, ``max_model_len`` to vLLM's. The
kernel packs a private weight copy at creation: weight reloads are not supported.
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.models.common.mono import (
    MonoOp,
    MonoRuntime,
    StepDecision,
    active_mono_layer_op,
    mono_layer,
    register_mono_layer_op,
)
from vllm.models.deepseek_v32.amd.mono import envs as mono_envs
from vllm.models.deepseek_v32.amd.mono import guards
from vllm.models.deepseek_v32.amd.mono.live import LiveConfig, MonoLive, _mla_cache
from vllm.models.deepseek_v32.amd.mono.spec import GLM5_MONO

logger = init_logger(__name__)

CKPT_INDEX = "model.safetensors.index.json"


def resolve_ckpt_dir(vllm_config) -> str:
    """Local checkpoint dir vLLM loaded (an HF repo id resolves to its cached snapshot,
    never downloads); mirrors ``DefaultModelLoader._prepare_weights``."""
    mc, lc = vllm_config.model_config, vllm_config.load_config
    model, revision, download_dir = mc.model, mc.revision, lc.download_dir
    fmt = str(lc.load_format or "auto")
    hint = (
        'set "ckpt" (a local checkpoint directory) in VLLM_ROCM_GLM5_MONOKERNEL_CONFIG'
    )
    if fmt.lower().endswith("dummy"):
        raise RuntimeError(
            f"GLM-5.2 MonoKernel: load_format={fmt} has no checkpoint to read the "
            f"kernel weights from; {hint}"
        )
    path = model
    if not os.path.isdir(model):
        try:
            from vllm import envs

            if envs.VLLM_USE_MODELSCOPE:
                from modelscope.hub.snapshot_download import snapshot_download

                kw = dict(model_id=model)
            else:
                from vllm.transformers_utils.repo_utils import hf_api

                snapshot_download, kw = hf_api().snapshot_download, dict(repo_id=model)
            path = snapshot_download(
                **kw, revision=revision, cache_dir=download_dir, local_files_only=True
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"GLM-5.2 MonoKernel: cannot resolve the local snapshot of {model!r} "
                f"(revision {revision!r}, download_dir {download_dir!r}): {e!r}; "
                f"{hint}"
            ) from e
    if not os.path.isfile(os.path.join(path, CKPT_INDEX)):
        raise RuntimeError(
            f"GLM-5.2 MonoKernel: {path!r} (resolved from {model!r}) has no "
            f"{CKPT_INDEX}; {hint}"
        )
    return path


def config_overrides(vllm_config) -> dict:
    """``LiveConfig`` fields the environment overrides, without touching the disk.

    Split out of :func:`config_from_env` so the widths and the graph-mode checks are
    known before the checkpoint is resolved: a refusal should say the configuration is
    unservable, not that the checkpoint is unreadable.
    """
    over = mono_envs.config_overrides()
    if "sizes" in over:
        over["sizes"] = tuple(over["sizes"])
    # step_sync is illegal inside a graph capture: default it off under FULL graphs
    over.setdefault("step_sync", not guards.full_cudagraphs(vllm_config))
    over.setdefault("max_model_len", int(vllm_config.model_config.max_model_len))
    return over


def config_from_env(vllm_config, over: dict | None = None) -> LiveConfig:
    over = dict(config_overrides(vllm_config) if over is None else over)
    if "ckpt" not in over:
        over["ckpt"] = resolve_ckpt_dir(vllm_config)
    return LiveConfig(**over)


class Glm5MonoDecode(MonoOp):
    """GLM-5.2's mono decode layers, one op for the whole model.

    One op serves every mono layer: the layers share the kernels, the peer buffers and
    the step decision, so splitting them per layer would only duplicate state.
    """

    spec = GLM5_MONO

    def __init__(self, vllm_config, model):
        self.model = model
        self.model_id = id(model)
        self.over = config_overrides(vllm_config)
        sizes = tuple(sorted(self.over.get("sizes", LiveConfig.sizes)))
        self.spec = replace(
            type(self).spec,
            widths=sizes,
            constraints=type(self).spec.constraints + (self._config_refusal,),
        )
        super().__init__(vllm_config)

    def _config_refusal(self, vllm_config) -> str | None:
        """What the static spec cannot see: this process and the parsed config."""
        if active_mono_layer_op(self.model_id) is not None:
            return (
                "already built for this model; reloading or updating weights is not "
                "supported with the MonoKernel enabled (restart the engine)"
            )
        if self.over["step_sync"] and guards.full_cudagraphs(vllm_config):
            return "step_sync syncs and all-reduces, which a FULL graph cannot capture"
        return guards.graph_width_mismatch(self.spec.widths, vllm_config)

    def build(self) -> None:
        cfg = self.cfg = config_from_env(self.vllm_config, self.over)
        # RoPE length, graph widths (the KV caches are not bound yet)
        guards.check_before_install(self.model, cfg, self.vllm_config, check_kv=False)
        self.rt = MonoRuntime(
            self.spec,
            self.vllm_config,
            vote=cfg.step_sync,
            vote_each_step=cfg.check_every <= 0,
            enabled=cfg.enabled,
        )
        self.lv = MonoLive(self.model, cfg, self.vllm_config, self.rt)
        self.layers = frozenset(self.lv.layers)
        # (data_ptr, numel) of the first mono layer's KV cache the guards last ran on
        self._guarded: tuple[int, int] | None = None
        self.watch = self._poll_watch()
        if guards.piecewise_graphs_only(self.vllm_config):
            logger.warning(
                "GLM-5.2 MonoKernel: PIECEWISE-only breakable cudagraphs capture "
                "decode steps without attention metadata, so captured steps never "
                "take the mono path; use FULL decode graphs or eager mode"
            )
        register_mono_layer_op(self, self.model_id)
        logger.info(
            "GLM-5.2 MonoKernel dispatch: layers %d..%d, widths %s",
            min(self.layers),
            max(self.layers),
            self.lv.sizes,
        )

    def _poll_watch(self):
        """Fail-stop on expired kernel polls, advanced by ``after_step``;
        ``MONO_LIVE_FAILSTOP=0`` disables it, ``=warn`` logs."""
        if guards.failstop_mode() == "off":
            logger.warning(
                "GLM-5.2 MonoKernel dispatch: MONO_LIVE_FAILSTOP=0 -> no poll-error "
                "fail-stop watch"
            )
            return None
        return guards.PollErrorWatch(self.lv)

    def after_step(self):
        """Once per executed step, from compute_logits (runs eagerly on every rank, also
        under graph replay). Raises on expired polls."""
        if self.watch is not None and not torch.cuda.is_current_stream_capturing():
            self.watch.after_step()

    def _maybe_guard(self, layer) -> bool:
        """KV-dependent guards, run again whenever vLLM binds new caches (CUDA-graph
        memory profiling binds minimal caches before the real ones); False while none
        are bound. First mono layer only."""
        kv = _mla_cache(layer)
        if kv is None or kv.numel() == 0:
            return False
        if self._guarded == (kv.data_ptr(), kv.numel()):
            return True
        guards.check_before_install(self.model, self.cfg, self.vllm_config)
        guards.check_after_install(self.lv, self.vllm_config)
        self._guarded = (kv.data_ptr(), kv.numel())
        return True

    def eligible(self, layer, positions, hidden_states, residual) -> StepDecision:
        """This step's decision, taken once at the first mono layer.

        The reason is rank-uniform (metadata only); rank-local state enters through the
        runtime's vote. A go step then fills the padded per-step inputs at the width the
        runtime chose.
        """
        if residual is None:
            why = "no_residual"
        elif not self._maybe_guard(layer):
            why = "no_kv_cache"
        else:
            why = self.lv.step_reason(layer, hidden_states, residual)
        step = self.rt.step_begin(hidden_states.shape[0], why)
        if step:
            assert step.width is not None  # a go step was given a width
            self.lv.prepare_step(positions, step.width)
        return step

    def forward_layer(self, layer, positions, hidden_states, residual):
        """One mono layer index. The step decision and vLLM's layer on no-go steps stay
        outside the custom op; under FULL graphs this Python runs at capture only."""
        if layer.layer_idx == self.lv.first:
            self.eligible(layer, positions, hidden_states, residual)
        if residual is None or not self.rt.step:
            return layer(positions, hidden_states, residual)
        return mono_layer(
            positions, hidden_states, residual, layer.layer_idx, self.model_id
        )

    def forward(self, positions, hidden_states, residual, layer_idx):
        return self.lv.mono_forward(
            self.lv.layers[layer_idx], positions, hidden_states, residual
        )
