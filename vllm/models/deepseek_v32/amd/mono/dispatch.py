# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-integrated MonoKernel dispatch.

``Glm5MonoDecode.maybe_create`` (called after weight loading when
``VLLM_ROCM_USE_GLM5_MONOKERNEL=1``) builds kernels, buffers and IPC peers before memory
profiling and graph capture. ``forward_layer`` takes the per-step go / no-go decision at
the first mono layer and runs either the custom op
``torch.ops.vllm.glm5_mono_decode_layer`` or vLLM's own decoder layer.

Config: ``LiveConfig`` defaults overridden by ``VLLM_ROCM_GLM5_MONOKERNEL_CONFIG``
(JSON). ``ckpt`` defaults to vLLM's local checkpoint dir, ``max_model_len`` to vLLM's.
The kernel packs a private copy of the weights at creation: weight reloads are not
supported while it is enabled.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger
from vllm.models.deepseek_v32.amd.mono import envs as mono_envs

logger = init_logger(__name__)

_ACTIVE: dict = dict(obj=None)


def active() -> Glm5MonoDecode:
    obj = _ACTIVE["obj"]
    if obj is None:
        raise RuntimeError(
            "glm5_mono_decode_layer called with no active Glm5MonoDecode"
        )
    return obj


def refusal(vllm_config) -> str | None:
    """None if this configuration can run the MonoKernel, else why not."""
    mc, pc = vllm_config.model_config, vllm_config.parallel_config
    if getattr(mc.hf_config, "model_type", None) != "glm_moe_dsa":
        return f"model_type {getattr(mc.hf_config, 'model_type', None)} (GLM-5.2 only)"
    if pc.tensor_parallel_size != 8:
        return f"TP {pc.tensor_parallel_size} (TP8 only)"
    if pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1:
        return "PP/DP > 1"
    if getattr(pc, "enable_expert_parallel", False):
        return "expert parallel"
    if getattr(pc, "decode_context_parallel_size", 1) != 1:
        return "DCP"
    if vllm_config.speculative_config is not None:
        return "speculative decoding"
    # the mono layers run their own weight copy and skip vLLM's attention layers
    if getattr(vllm_config, "lora_config", None) is not None:
        return "LoRA"
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        return "KV transfer connector"
    aux = getattr(vllm_config, "aux_output_config", None)
    if getattr(aux, "enable_return_routed_experts", False):
        return "routed-expert capture"
    import torch

    if getattr(mc, "dtype", torch.bfloat16) != torch.bfloat16:
        return f"dtype {mc.dtype} (bf16 only)"
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        return f"kv_cache_dtype {vllm_config.cache_config.cache_dtype} (bf16 only)"
    try:
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            return "not gfx950"
    except Exception as e:  # noqa: BLE001
        return f"platform check failed: {e!r}"
    return None


CKPT_INDEX = "model.safetensors.index.json"


def resolve_ckpt_dir(vllm_config) -> str:
    """Local checkpoint dir vLLM loaded (an HF repo id resolves to its cached
    snapshot, never downloads); mirrors ``DefaultModelLoader._prepare_weights``."""
    mc = vllm_config.model_config
    lc = getattr(vllm_config, "load_config", None)
    model, revision = mc.model, getattr(mc, "revision", None)
    download_dir = getattr(lc, "download_dir", None)
    fmt = str(getattr(lc, "load_format", "auto") or "auto")
    hint = (
        'set "ckpt" (a local checkpoint directory) in VLLM_ROCM_GLM5_MONOKERNEL_CONFIG'
    )
    if fmt.lower().endswith("dummy"):
        raise RuntimeError(
            f"GLM-5.2 MonoKernel: load_format={fmt} has no checkpoint to read the "
            f"kernel weights from; {hint}"
        )
    if os.path.isdir(model):
        path = model
    else:
        try:
            from vllm import envs

            if envs.VLLM_USE_MODELSCOPE:
                from modelscope.hub.snapshot_download import (
                    snapshot_download as ms_snapshot,
                )

                path = ms_snapshot(
                    model_id=model,
                    cache_dir=download_dir,
                    revision=revision,
                    local_files_only=True,
                )
            else:
                from vllm.transformers_utils.repo_utils import hf_api

                path = hf_api().snapshot_download(
                    repo_id=model,
                    revision=revision,
                    cache_dir=download_dir,
                    local_files_only=True,
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


def config_from_env(vllm_config, over: dict | None = None):
    from vllm.models.deepseek_v32.amd.mono.live import LiveConfig

    over = dict(mono_envs.config_overrides() if over is None else over)
    if "sizes" in over:
        over["sizes"] = tuple(over["sizes"])
    # step_sync is illegal inside a graph capture: default it off under FULL graphs
    from vllm.models.deepseek_v32.amd.mono.guards import full_cudagraphs

    over.setdefault("step_sync", not full_cudagraphs(vllm_config))
    if "ckpt" not in over:
        over["ckpt"] = resolve_ckpt_dir(vllm_config)
    over.setdefault("max_model_len", int(vllm_config.model_config.max_model_len))
    return LiveConfig(**over)


class Glm5MonoDecode:
    @classmethod
    def maybe_create(cls, vllm_config, causal_lm) -> Glm5MonoDecode | None:
        """None when the switch is off. When it is on, a configuration the kernel cannot
        run (``refusal``, FULL-graph capture sizes that differ from the kernel widths)
        raises with the reason."""
        if not mono_envs.enabled():
            return None
        if _ACTIVE["obj"] is not None:
            # the kernel packs a private copy of the weights at creation
            raise RuntimeError(
                "GLM-5.2 MonoKernel: already created in this process; reloading or "
                "updating weights is not supported with the MonoKernel enabled "
                "(restart the engine)"
            )
        why = refusal(vllm_config)
        cfg = None
        if why is None:
            from vllm.models.deepseek_v32.amd.mono.guards import graph_width_mismatch

            cfg = config_from_env(vllm_config)
            why = graph_width_mismatch(cfg.sizes, vllm_config)
        if why is not None:
            raise RuntimeError(
                f"GLM-5.2 MonoKernel cannot run this configuration: {why} (unset "
                f"{mono_envs.ENABLE} to use vLLM's decode path)"
            )
        from vllm.models.deepseek_v32.amd.mono.guards import piecewise_graphs_only

        if piecewise_graphs_only(vllm_config):
            logger.warning(
                "GLM-5.2 MonoKernel: PIECEWISE-only breakable cudagraphs capture "
                "decode steps without attention metadata, so captured steps never "
                "take the mono path; use FULL decode graphs or eager mode"
            )
        obj = cls(vllm_config, causal_lm, cfg)
        _ACTIVE["obj"] = obj
        return obj

    def __init__(self, vllm_config, causal_lm, cfg=None):
        from vllm.models.deepseek_v32.amd.mono import live
        from vllm.models.deepseek_v32.amd.mono.guards import check_before_install

        self.vllm_config = vllm_config
        self.model = causal_lm
        self.cfg = config_from_env(vllm_config) if cfg is None else cfg
        # RoPE length, graph widths
        check_before_install(causal_lm, self.cfg, vllm_config, check_kv=False)
        self.lv = live.MonoLive(causal_lm, self.cfg, vllm_config)
        self.layers = frozenset(self.lv.layers)
        self._by_idx = dict(self.lv.layers)
        # (data_ptr, numel) of the first mono layer's KV cache the guards last ran on
        self._guarded: tuple[int, int] | None = None
        self.watch = None
        from vllm.models.deepseek_v32.amd.ops.glm5_mono import glm5_mono_decode_layer

        self._op = glm5_mono_decode_layer
        self._install_poll_watch()
        logger.info(
            "GLM-5.2 MonoKernel dispatch: layers %d..%d, widths %s",
            min(self.layers),
            max(self.layers),
            self.lv.sizes,
        )

    def _install_poll_watch(self):
        """Fail-stop on expired kernel polls (guards.PollErrorWatch), advanced by
        ``after_step``. ``MONO_LIVE_FAILSTOP=0`` disables it, ``=warn`` logs."""
        from vllm.models.deepseek_v32.amd.mono.guards import (
            PollErrorWatch,
            failstop_mode,
        )

        if failstop_mode() == "off":
            logger.warning(
                "GLM-5.2 MonoKernel dispatch: MONO_LIVE_FAILSTOP=0 -> no poll-error "
                "fail-stop watch"
            )
            return
        self.watch = PollErrorWatch(self.lv)

    def after_step(self):
        """Call once per executed step from compute_logits, which runs eagerly on every
        rank, also under graph replay. Raises on expired polls."""
        import torch

        if self.watch is not None and not torch.cuda.is_current_stream_capturing():
            self.watch.after_step()

    def _maybe_guard(self, layer) -> bool:
        """KV-dependent guards, run again whenever vLLM binds new caches (the CUDA-graph
        memory profiling run binds minimal caches before the real ones); False while
        none are bound (profiling run, between the two bindings). First mono layer
        only."""
        kv = layer.self_attn.kv_cache
        kv = kv[0] if isinstance(kv, (list, tuple)) else kv
        if kv is None or kv.numel() == 0:
            return False
        if self._guarded == (kv.data_ptr(), kv.numel()):
            return True
        from vllm.models.deepseek_v32.amd.mono.guards import (
            check_after_install,
            check_before_install,
        )

        check_before_install(self.model, self.cfg, self.vllm_config)
        check_after_install(self.lv, self.vllm_config)
        self._guarded = (kv.data_ptr(), kv.numel())
        return True

    def forward_layer(self, layer, positions, hidden_states, residual):
        """One mono layer index. The step decision and vLLM's layer on no-go steps stay
        outside the custom op; under FULL graphs this Python runs at capture only."""
        lv = self.lv
        if layer.layer_idx == lv.first:
            if residual is None or not self._maybe_guard(layer):
                lv.active = False
                return layer(positions, hidden_states, residual)
            lv._begin_step(layer, positions, hidden_states, residual)
        if residual is None or not lv.active:
            return layer(positions, hidden_states, residual)
        return self._op(positions, hidden_states, residual, layer.layer_idx)

    def forward_layer_impl(self, layer_idx, positions, hidden_states, residual):
        return self.lv.mono_forward(
            self._by_idx[layer_idx], positions, hidden_states, residual
        )
