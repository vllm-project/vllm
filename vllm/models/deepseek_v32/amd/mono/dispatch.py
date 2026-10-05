# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-integrated MonoKernel dispatch.

``Glm5MonoDecode.maybe_create`` (after weight loading,
``VLLM_ROCM_USE_GLM5_MONOKERNEL=1``) builds kernels, buffers and IPC peers before memory
profiling and graph capture. ``forward_layer`` takes the per-step go / no-go decision at
the first mono layer and runs either the custom op
``torch.ops.vllm.glm5_mono_decode_layer`` or vLLM's decoder layer. Config:
``LiveConfig`` defaults overridden by ``VLLM_ROCM_GLM5_MONOKERNEL_CONFIG`` (JSON);
``ckpt`` defaults to vLLM's local checkpoint dir, ``max_model_len`` to vLLM's. The
kernel packs a private weight copy at creation: weight reloads are not supported.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v32.amd.mono import envs as mono_envs
from vllm.models.deepseek_v32.amd.mono import guards
from vllm.models.deepseek_v32.amd.mono.live import LiveConfig, MonoLive, _mla_cache

logger = init_logger(__name__)

_ACTIVE: dict = dict(obj=None)
CKPT_INDEX = "model.safetensors.index.json"


def active() -> Glm5MonoDecode:
    if _ACTIVE["obj"] is None:
        raise RuntimeError(
            "glm5_mono_decode_layer called with no active Glm5MonoDecode"
        )
    return _ACTIVE["obj"]


def refusal(vllm_config) -> str | None:
    """None if this configuration can run the MonoKernel, else why not."""
    mc, pc = vllm_config.model_config, vllm_config.parallel_config
    model_type = getattr(mc.hf_config, "model_type", None)
    aux = vllm_config.aux_output_config
    checks = (
        (model_type != "glm_moe_dsa", f"model_type {model_type} (GLM-5.2 only)"),
        (pc.tensor_parallel_size != 8, f"TP {pc.tensor_parallel_size} (TP8 only)"),
        (pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1, "PP/DP > 1"),
        (pc.enable_expert_parallel, "expert parallel"),
        (pc.decode_context_parallel_size != 1, "DCP"),
        (vllm_config.speculative_config is not None, "speculative decoding"),
        # the mono layers run their own weight copy and skip vLLM's attention layers
        (vllm_config.lora_config is not None, "LoRA"),
        (vllm_config.kv_transfer_config is not None, "KV transfer connector"),
        (aux.enable_return_routed_experts, "routed-expert capture"),
        (mc.dtype != torch.bfloat16, f"dtype {mc.dtype} (bf16 only)"),
        (
            vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"),
            f"kv_cache_dtype {vllm_config.cache_config.cache_dtype} (bf16 only)",
        ),
    )
    for bad, why in checks:
        if bad:
            return why
    try:
        from vllm.platforms.rocm import on_gfx950

        return None if on_gfx950() else "not gfx950"
    except Exception as e:  # noqa: BLE001
        return f"platform check failed: {e!r}"


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


def config_from_env(vllm_config) -> LiveConfig:
    over = mono_envs.config_overrides()
    if "sizes" in over:
        over["sizes"] = tuple(over["sizes"])
    # step_sync is illegal inside a graph capture: default it off under FULL graphs
    over.setdefault("step_sync", not guards.full_cudagraphs(vllm_config))
    if "ckpt" not in over:
        over["ckpt"] = resolve_ckpt_dir(vllm_config)
    over.setdefault("max_model_len", int(vllm_config.model_config.max_model_len))
    return LiveConfig(**over)


class Glm5MonoDecode:
    @classmethod
    def maybe_create(cls, vllm_config, causal_lm) -> Glm5MonoDecode | None:
        """None when the switch is off; raises with the reason when it is on and the
        kernel cannot run the configuration (``refusal``, FULL-graph capture sizes that
        differ from the kernel widths)."""
        from vllm import envs

        if not envs.VLLM_ROCM_USE_GLM5_MONOKERNEL:
            return None
        if _ACTIVE["obj"] is not None:
            raise RuntimeError(
                "GLM-5.2 MonoKernel: already created in this process; reloading or "
                "updating weights is not supported with the MonoKernel enabled "
                "(restart the engine)"
            )
        why = refusal(vllm_config)
        if why is None:
            cfg = config_from_env(vllm_config)
            why = guards.graph_width_mismatch(cfg.sizes, vllm_config)
        if why is not None:
            raise RuntimeError(
                f"GLM-5.2 MonoKernel cannot run this configuration: {why} (unset "
                f"{mono_envs.ENABLE} to use vLLM's decode path)"
            )
        if guards.piecewise_graphs_only(vllm_config):
            logger.warning(
                "GLM-5.2 MonoKernel: PIECEWISE-only breakable cudagraphs capture "
                "decode steps without attention metadata, so captured steps never "
                "take the mono path; use FULL decode graphs or eager mode"
            )
        _ACTIVE["obj"] = obj = cls(vllm_config, causal_lm, cfg)
        return obj

    def __init__(self, vllm_config, causal_lm, cfg: LiveConfig):
        from vllm.models.deepseek_v32.amd.ops.glm5_mono import glm5_mono_decode_layer

        self.vllm_config, self.model, self.cfg = vllm_config, causal_lm, cfg
        # RoPE length, graph widths (the KV caches are not bound yet)
        guards.check_before_install(causal_lm, cfg, vllm_config, check_kv=False)
        self.lv = MonoLive(causal_lm, cfg, vllm_config)
        self.layers = frozenset(self.lv.layers)
        # (data_ptr, numel) of the first mono layer's KV cache the guards last ran on
        self._guarded: tuple[int, int] | None = None
        self._op = glm5_mono_decode_layer
        self.watch = self._poll_watch()
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
            self.lv.layers[layer_idx], positions, hidden_states, residual
        )
