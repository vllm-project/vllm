# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's DSpark draft layers as one FlyDSL launch a step (``--monokernel``).

A draft step of 1 or 2 requests (7 query rows each) runs the 5 layers and the final
norm of ``MuseSparkDSparkModel`` as one launch of ``kernels/draft.py`` (builds
S = 8 / 16); any other step runs the stock layers. The contract is the target
monokernel's (README.md): the weights are read in place (fp32 copies of the norm
multipliers only), one IPC peer buffer a rank and build allocated at load time, fixed
launch arguments (FULL CUDA graphs capture the launch), the draft KV cache group's
``block_table`` / ``seq_lens`` / ``slot_mapping`` and the layers' bf16 LBHNC caches
through one descriptor table a step shape (keyed by its contents), and health words
that the target's ``compute_logits`` checks. Every rank takes the same path: the
decision reads only the static deployment and the step's shape and attention metadata
counts.
"""

from types import ModuleType
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNormWithInputScale,
)

from .decoder import (
    HEALTH_POLL_INTERVAL,
    NUM_CUS,
    SLIDING_WINDOW,
    WORLD_SIZE,
    MonoDims,
    _FusedAttention,
    kv_cache_refusal,
    register_health_check,
)
from .peer import PeerBuffer

logger = init_logger(__name__)

QUERY_ROWS = 7  # query rows a request of a DSpark draft step
DRAFT_BUILDS = {7: 8, 14: 16}  # rows -> build S
DRAFT_LAYERS = 5

# The draft shape kernels/draft.py is specialized for.
DRAFT_CONFIG: dict[str, float] = {
    "hidden_size": 8192,
    "intermediate_size_mlp": 16384,
    "head_dim": 64,
    "num_attention_heads": 128,
    "num_key_value_heads": 16,
    "num_hidden_layers": DRAFT_LAYERS,
    "rms_norm_eps": 1e-5,
    "rms_post_norm_eps": 1e-8,
}
DRAFT_FLAGS = {
    "use_rope": True,
    "use_qk_norm": True,
    "use_attn_o_norm": True,
    "use_attn_o_gate": True,
    "use_residual_gate": True,
    "use_ffn_learnable_scales": True,
    "norm_affine": True,
    "norm_zero_centered_gamma": True,
    "post_norm_affine": False,
}


def draft_dims(config: Any, world_size: int) -> MonoDims:
    """Per-rank sizes of the draft layers (the MoE fields unused)."""
    return MonoDims(
        world_size=world_size,
        hidden_size=config.hidden_size,
        pooled_size=0,
        intermediate_size=config.intermediate_size_mlp,
        num_experts=0,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )


def draft_config_refusals(config: Any) -> list[str]:
    reasons = []
    for name, want in DRAFT_CONFIG.items():
        got = getattr(config, name, None)
        if not isinstance(got, (int, float)) or abs(got - want) > 1e-12 * max(
            1.0, abs(want)
        ):
            reasons.append(f"draft config.{name}={got} (kernel: {want})")
    for name, want in DRAFT_FLAGS.items():
        if bool(getattr(config, name, None)) != want:
            reasons.append(f"draft config.{name}={getattr(config, name, None)}")
    if getattr(config, "residual_dtype", None) != torch.float32:
        reasons.append("draft residual dtype not fp32")
    if getattr(config, "every_n_layers_nope", None) is not None:
        reasons.append("draft NoPE layers")
    return reasons


def effective_norm_weight_fp32(norm: nn.Module) -> torch.Tensor:
    """The fp32 multiplier of a norm: ``weight (+ gain_center)``, as the stock norms."""
    weight = norm.weight.detach().float()
    if norm.zero_centered_gamma:
        weight = weight + norm.gain_center
    return weight.contiguous()


def _check(bad: list[str], i: int, name: str, w: torch.Tensor, shape: tuple) -> None:
    if tuple(w.shape) != shape or w.dtype != torch.bfloat16 or not w.is_contiguous():
        bad.append(f"draft layer {i}: {name} {tuple(w.shape)} {w.dtype}")


def draft_layer_refusals(model: nn.Module, dims: MonoDims) -> list[str]:
    """Module and tensor guards of the draft layers (``MuseSparkDSparkModel``)."""
    bad: list[str] = []
    layers = model.layers
    if len(layers) != DRAFT_LAYERS:
        return [f"{len(layers)} draft layers"]
    h, w = dims.hidden_size, dims.world_size
    ff = dims.intermediate_size // w
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        if not (
            attn.use_qk_norm
            and attn.use_o_norm
            and attn.use_o_gate
            and attn.fuse_o_norm_gate
            and attn.fused_qk_norm_rope_kv_insert
            and attn.causal
        ) or getattr(attn, "use_attn_temperature_tuning", False):
            bad.append(f"draft layer {i}: attention options")
        backend = getattr(getattr(attn.attn, "attn_backend", None), "__name__", None)
        if backend != "AiterFlashAttentionBackend":
            bad.append(f"draft layer {i}: attention backend {backend}")
        window = getattr(getattr(attn.attn, "impl", None), "sliding_window", None)
        if window != SLIDING_WINDOW:
            bad.append(f"draft layer {i}: sliding window {window}")
        rope = attn.rotary_emb
        if (
            rope is None
            or rope.cos_sin_cache.dtype != torch.float32
            or rope.cos_sin_cache.shape[-1] != dims.head_dim
            or rope.rotary_dim != dims.head_dim
            or rope.is_neox_style
            or rope.base != layers[0].self_attn.rotary_emb.base
        ):
            bad.append(f"draft layer {i}: RoPE")
        _check(bad, i, "qkv_proj", attn.qkv_proj.weight, (dims.qkvg_rows, h))
        _check(bad, i, "o_proj", attn.o_proj.weight, (h, dims.q_rows))
        mlp = layer.feed_forward
        if getattr(layer, "is_moe_layer", True) or not hasattr(mlp, "gate_up_proj"):
            bad.append(f"draft layer {i}: not a dense MLP")
            continue
        _check(bad, i, "gate_up_proj", mlp.gate_up_proj.weight, (2 * ff, h))
        _check(bad, i, "down_proj", mlp.down_proj.weight, (h, ff))
        for name in ("add_attn", "add_ffn"):
            conn = getattr(layer, name)
            if not isinstance(
                conn, MuseSpark1xNormPreservingResidualConnection
            ) or not getattr(conn, "_coefficients_ready", False):
                bad.append(f"draft layer {i}: {name} coefficients not materialized")
        if layer.post_attn_norm.has_weight:
            bad.append(f"draft layer {i}: post_attn_norm has a weight")
        if not isinstance(layer.post_ffn_norm, MuseSpark1xRMSNormWithInputScale) or (
            not layer.post_ffn_norm.has_weight
        ):
            bad.append(f"draft layer {i}: post_ffn_norm type")
        for name in ("pre_ffn_norm",):
            if not getattr(layer, name).has_weight:
                bad.append(f"draft layer {i}: weightless {name}")
        if not attn.pre_qkv_proj_norm.has_weight:
            bad.append(f"draft layer {i}: weightless pre_qkv_proj_norm")
    if not model.norm.has_weight:
        bad.append("draft: weightless final norm")
    return bad


class _DraftBuild:
    """The launcher and persistent buffers of one draft build (S rows)."""

    def __init__(
        self,
        kernel: ModuleType,
        attn_frag: ModuleType,
        S: int,
        table: torch.Tensor,
        scales: dict[int, float],
        dims: MonoDims,
        device: torch.device,
        group: Any,
    ) -> None:
        self.S = S
        self.launch = kernel.build(kernel.build_vllm(S=S, W=dims.world_size))
        self.peers = PeerBuffer(kernel.peer_layout(S)[1] + 256, group)
        self.scratch = torch.zeros(
            kernel.scratch_layout(S)[1] // 4 + 64, dtype=torch.int32, device=device
        )
        self.epoch = torch.zeros(4, dtype=torch.int32, device=device)
        self.health = torch.zeros(kernel.HEALTH_WORDS, dtype=torch.int32, device=device)
        bf16 = torch.bfloat16
        self.q = torch.zeros(S, dims.q_rows, dtype=bf16, device=device)
        self.g = torch.zeros(S, dims.q_rows, dtype=bf16, device=device)
        self.fa = _FusedAttention(
            attn_frag, dict.fromkeys(scales, "swa"), scales, S, dims, device
        )
        # the launch table: the layers' rows, then this build's buffers and peers
        host = table.clone()
        g = kernel.T_G
        host[g, kernel.G_Q] = self.q.data_ptr()
        host[g, kernel.G_Q + 1] = self.g.data_ptr()
        host[g, kernel.G_PEER : kernel.G_PEER + dims.world_size] = torch.tensor(
            self.peers.pointers, dtype=torch.int64
        )
        self.table = host.to(device)


class DSparkDrafter:
    """Runs the draft layers of eligible DSpark draft steps as kernels/draft.py."""

    def __init__(
        self,
        model: nn.Module,
        kernels: dict[str, ModuleType],
        rank: int,
        group: Any,
        device: torch.device,
        dims: MonoDims,
    ) -> None:
        self.rank = rank
        self.dims = dims
        self.kernel, attn_frag = kernels["draft"], kernels["attn_frag"]
        self.layer_names = [layer.self_attn.attn.layer_name for layer in model.layers]
        scales = {
            i: float(layer.self_attn.attn.impl.scale)
            for i, layer in enumerate(model.layers)
        }
        self.cos_sin = model.layers[0].self_attn.rotary_emb.cos_sin_cache
        table = self._weight_table(model, device)
        self.builds = {
            S: _DraftBuild(
                self.kernel, attn_frag, S, table, scales, self.dims, device, group
            )
            for S in sorted(set(DRAFT_BUILDS.values()))
        }
        self._fallbacks: set[str] = set()
        self._engaged: set[int] = set()
        self._steps = 0
        self._health_host = {
            S: torch.zeros_like(b.health, device="cpu").pin_memory()
            for S, b in self.builds.items()
        }
        self._health_event: torch.cuda.Event | None = None

    def _weight_table(self, model: nn.Module, device: torch.device) -> torch.Tensor:
        """The launch table rows of kernels/draft.py (weights as views of the
        parameters; the norm multipliers as fp32 copies, kept alive here)."""
        k = self.kernel
        h = self.dims.hidden_size
        self._keep = [
            torch.ones(h, dtype=torch.float32, device=device),
            torch.zeros(h, dtype=torch.float32, device=device),
        ]
        ones, zeros = self._keep

        def keep(t: torch.Tensor) -> int:
            self._keep.append(t)
            return t.data_ptr()

        table = torch.zeros(k.TAB_ROWS, k.TAB_WORDS, dtype=torch.int64)
        layers = model.layers
        for i in range(DRAFT_LAYERS + 1):
            if i == 0:
                bnd = [ones.data_ptr(), zeros.data_ptr(), ones.data_ptr()]
            else:
                prev = layers[i - 1]
                bnd = [
                    prev.add_ffn.alpha.data_ptr(),
                    prev.add_ffn.beta.data_ptr(),
                    keep(effective_norm_weight_fp32(prev.post_ffn_norm)),
                ]
            nxt = (
                model.norm
                if i == DRAFT_LAYERS
                else layers[i].self_attn.pre_qkv_proj_norm
            )
            bnd.append(keep(effective_norm_weight_fp32(nxt)))
            table[i, k.T_BND : k.T_BND + 4] = torch.tensor(bnd, dtype=torch.int64)
        for i, layer in enumerate(layers):
            attn, mlp = layer.self_attn, layer.feed_forward
            row = [
                attn.qkv_proj.weight.data_ptr(),
                attn.o_proj.weight.data_ptr(),
                layer.add_attn.alpha.data_ptr(),
                layer.add_attn.beta.data_ptr(),
                keep(effective_norm_weight_fp32(layer.pre_ffn_norm)),
                mlp.gate_up_proj.weight.data_ptr(),
                mlp.down_proj.weight.data_ptr(),
            ]
            table[i, k.T_LAYER : k.T_LAYER + len(row)] = torch.tensor(
                row, dtype=torch.int64
            )
        table[k.T_G, k.G_Q + 2] = self.cos_sin.data_ptr()
        return table

    def _step_refusal(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor | None
    ) -> str | None:
        """Why this draft step runs stock (rank-invariant data only); None = mono."""
        rows = positions.shape[0] if positions.dim() == 1 else -1
        if rows not in DRAFT_BUILDS:
            return f"rows:{rows}"
        if positions.dtype != torch.int64:
            return "positions"
        if inputs_embeds is not None and (
            inputs_embeds.dtype != torch.bfloat16
            or tuple(inputs_embeds.shape) != (rows, self.dims.hidden_size)
            or not inputs_embeds.is_contiguous()
        ):
            return "inputs-embeds"
        if not is_forward_context_available():
            return "no-forward-context"
        context = get_forward_context()
        mode = context.cudagraph_runtime_mode
        if mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
            return f"cudagraph:{mode.name}"
        metadata = context.attn_metadata
        if not isinstance(metadata, dict) or not metadata:
            return "attn-metadata"
        md0, slots0 = None, None
        for name in self.layer_names:
            md = metadata.get(name)
            if md is None:
                return "attn-metadata"
            _, _, kv_cache, slots = get_attention_context(name)
            if md0 is None:
                md0, slots0 = md, slots
            # one KV cache group: one block table, seq_lens and slot mapping
            if slots is None or (
                md.block_table.data_ptr(),
                md.seq_lens.data_ptr(),
                slots.data_ptr(),
            ) != (
                md0.block_table.data_ptr(),
                md0.seq_lens.data_ptr(),
                slots0.data_ptr(),
            ):
                return "kv-groups"
            reason = kv_cache_refusal(kv_cache, self.dims.local_kv_heads)
            if reason is not None:
                return reason
        # uniform causal decodes of QUERY_ROWS rows (host counts of the metadata)
        decode = getattr(md0, "decode_metadata", None)
        if (
            getattr(md0, "num_prefills", None) != 0
            or getattr(md0, "num_extends", None) != 0
            or decode is None
            or decode.max_query_len != QUERY_ROWS
            or md0.num_decodes * QUERY_ROWS != rows
            or not getattr(md0, "causal", False)
        ):
            return "not-uniform-causal"
        nreq = rows // QUERY_ROWS
        bt, seq_lens = md0.block_table, md0.seq_lens
        if (
            bt.dtype != torch.int32
            or bt.stride(-1) != 1
            or bt.shape[0] < nreq
            or seq_lens.dtype != torch.int32
            or not seq_lens.is_contiguous()
            or seq_lens.shape[0] < nreq
        ):
            return "block-table"
        if (
            slots0 is None
            or slots0.dtype != torch.int64
            or slots0.shape[0] < rows
            or slots0.stride(-1) != 1
        ):
            return "slot-mapping"
        return None

    def try_forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """The draft step's final hidden states (bf16 [rows, hidden]); None = stock."""
        reason = self._step_refusal(positions, inputs_embeds)
        if reason is not None:
            if reason not in self._fallbacks:
                self._fallbacks.add(reason)
                logger.debug("Muse Spark DSpark drafter: stock step (%s)", reason)
            return None
        rows = positions.shape[0]
        b = self.builds[DRAFT_BUILDS[rows]]
        if rows not in self._engaged:
            self._engaged.add(rows)
            logger.info("Muse Spark DSpark drafter engaged: %d rows, S=%d", rows, b.S)
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        x = inputs_embeds.to(torch.bfloat16).contiguous()
        out = torch.empty_like(x)
        layers = model.layers
        table = b.fa.table(layers, lambda i: b.q, rows // QUERY_ROWS)
        _, _, _, slots = get_attention_context(self.layer_names[0])
        b.launch(
            x.data_ptr(),
            out.data_ptr(),
            b.table.data_ptr(),
            table.data_ptr(),
            positions.data_ptr(),
            slots.data_ptr(),
            b.scratch.data_ptr(),
            b.epoch.data_ptr(),
            b.health.data_ptr(),
            self.rank,
            stream=torch.cuda.current_stream(),
        )
        return out

    def poll_health(self) -> None:
        """Raise if a draft launch timed out waiting for a peer rank (an asynchronous
        copy every HEALTH_POLL_INTERVAL calls, checked one interval later)."""
        if torch.cuda.is_current_stream_capturing():
            return
        self._steps += 1
        if self._health_event is not None and self._health_event.query():
            self._health_event = None
            for S, host in self._health_host.items():
                if int(host[0]) != 0:
                    records = host[8:].view(-1, 8)
                    stuck = records[records[:, 0] != 0][:8].tolist()
                    raise RuntimeError(
                        f"Muse Spark DSpark drafter: rank {self.rank}, S={S}: a launch "
                        f"timed out waiting for a peer rank {stuck}"
                    )
        if self._health_event is None and self._steps % HEALTH_POLL_INTERVAL == 0:
            for S, b in self.builds.items():
                self._health_host[S].copy_(b.health, non_blocking=True)
            self._health_event = torch.cuda.Event()
            self._health_event.record()

    def close(self) -> None:
        for b in self.builds.values():
            b.peers.close()


def static_refusals(lm: nn.Module, tp_size: int) -> list[str]:
    """Every reason this deployment cannot run the draft kernels; empty = supported."""
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return ["not ROCm"]
    from vllm.platforms.rocm import on_gfx950

    reasons = [] if on_gfx950() else ["not gfx950"]
    cus = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).multi_processor_count
    if cus != NUM_CUS:
        reasons.append(f"{cus} CUs (kernel: {NUM_CUS})")
    if tp_size != WORLD_SIZE:
        reasons.append(f"draft tensor_parallel_size={tp_size} (kernel: {WORLD_SIZE})")
    spec = lm.vllm_config.speculative_config
    if spec is None or spec.num_speculative_tokens != QUERY_ROWS:
        reasons.append(f"needs {QUERY_ROWS} draft tokens")
    if not getattr(lm.config, "sample_from_anchor", True):
        reasons.append("DSpark without sample_from_anchor (1 + N query rows)")
    cache = lm.vllm_config.cache_config
    if cache.cache_dtype not in ("auto", "bfloat16"):
        reasons.append(f"kv_cache_dtype={cache.cache_dtype} (kernel: bf16)")
    reasons += draft_config_refusals(lm.config)
    return reasons or draft_layer_refusals(lm.model, draft_dims(lm.config, tp_size))


def maybe_attach_drafter(lm: nn.Module) -> None:
    """Attach the draft kernels to ``lm`` (MuseSparkDSparkForCausalLM) after loading.

    Collective over the TP group: every rank calls it. A deployment that any rank
    cannot run keeps the stock draft layers on every rank (logged).
    """
    if not lm.vllm_config.kernel_config.monokernel:
        return
    from vllm.distributed import get_tp_group

    from . import _ensure_writable_flydsl_cache

    if lm.mono is not None:
        lm.mono.close()
        lm.mono = None
    group = get_tp_group()
    reasons = static_refusals(lm, group.world_size)
    kernels = None
    if not reasons:
        _ensure_writable_flydsl_cache()
        try:
            from .kernels import attn_frag, draft

            kernels = {"draft": draft, "attn_frag": attn_frag}
        except ImportError as e:
            reasons.append(f"needs FlyDSL and AITER's FlyDSL helpers ({e})")
    verdict = torch.tensor([0 if reasons else 1], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if int(verdict.item()) == 0:
        logger.info(
            "Muse Spark DSpark drafter off (stock draft layers): %s",
            "; ".join(reasons) or "a peer rank refused the deployment",
        )
        return
    assert kernels is not None
    lm.mono = DSparkDrafter(
        lm.model,
        kernels,
        rank=group.rank_in_group,
        group=group.cpu_group,
        device=torch.device("cuda", torch.accelerator.current_device_index()),
        dims=draft_dims(lm.config, group.world_size),
    )
    register_health_check(id(lm), lm.mono.poll_health)
    logger.info(
        "Muse Spark DSpark drafter attached: draft steps of %s rows run the %d draft "
        "layers and the final norm as one FlyDSL launch (builds S=%s)",
        "/".join(map(str, DRAFT_BUILDS)),
        DRAFT_LAYERS,
        ",".join(map(str, sorted(set(DRAFT_BUILDS.values())))),
    )


__all__ = ["DSparkDrafter", "maybe_attach_drafter"]
