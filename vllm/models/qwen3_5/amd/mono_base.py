# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vLLM side the Qwen mono decodes share (Qwen3.8 at TP8, Qwen3.8-Flash-Next
at TP2): the deployment's refusals, a step's eligibility, the scratch and peer
buffers, the weight checks' common part and the stock GDN core of a verify
step. A model's ``MonoDecode`` subclasses ``QwenMonoDecode`` with its shape
checks, its kernels' buffer sizes, its weights and its forward.

With MTP (``MTP``), a decode step is a target-verify step (each request's
sampled token and its drafts). K1's recurrence keeps no per-draft-token state
to roll back rejected drafts, so a verify step runs vLLM's GDN core on the
spec-decode metadata between the model's front and K2.

Every TP rank takes the same path at every step (the kernels wait on each
other's pushes): the conditions here read only metadata every rank holds
alike."""

from typing import Any, ClassVar

import torch

from vllm import envs
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context, in_piecewise_cudagraph
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.platforms import current_platform

logger = init_logger(__name__)

# the kernels' ISA: CDNA4 (gfx950) scaled MFMA and MX formats
CDNA_VERSIONS = (4,)


def enabled() -> bool:
    return envs.VLLM_ROCM_MONO_DECODE and current_platform.is_rocm()


def routed_experts(mlp):
    return getattr(mlp.experts, "routed_experts", mlp.experts)


def o_proj(layer):
    if layer.layer_type == "linear_attention":
        return layer.linear_attn.out_proj
    return layer.self_attn.o_proj


def asm_mxfp4(linear) -> bool:
    """An online-MXFP4 linear on AITER's ASM FP4 GEMM: its weight and scales
    shuffled as the kernels' fp4 GEMVs read them."""
    from vllm.model_executor.kernels.linear.mxfp4.aiter import AiterMxfp4LinearKernel
    from vllm.model_executor.layers.quantization.online.mxfp4 import (
        Mxfp4OnlineLinearMethod,
    )

    qm = linear.quant_method
    return (
        isinstance(qm, Mxfp4OnlineLinearMethod)
        and isinstance(qm.kernel, AiterMxfp4LinearKernel)
        and qm.kernel.use_asm_gemm
    )


def gdn_variant(a, gate: tuple[str, ...]) -> bool:
    """A GDN layer as K1 reads it: flat in_proj_qkvz, TP-sharded in_proj_ba,
    SiLU conv without bias, the norm before a ``gate`` output gate."""
    return (
        hasattr(a, "in_proj_qkvz")
        and not a.disable_tp_for_ba_proj
        and a.qkvz_layout == "flat"
        and a.activation == "silu"
        and a.conv1d.bias is None
        and a.norm.activation in gate
        and a.norm.group_size is None
        and a.norm.norm_before_gate
    )


def experts_error(ex, experts: int) -> str | None:
    """Why the kernels cannot read the routed experts ``ex`` (``experts`` of
    them a rank, the fused shared one included), else None."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    backend = getattr(ex.quant_method, "mxfp4_backend", None)
    if backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
        return f"routed experts on {backend}, not AITER_MXFP4_MXFP4"
    if not getattr(ex.w13_weight, "is_shuffled", False):
        return "routed experts not shuffled"
    if ex.w13_weight.shape[0] != experts:
        return "routed experts not all local"
    return None


def dense_error(i: int, dense: dict[str, torch.Tensor]) -> str | None:
    for name, w in dense.items():
        if w.dtype != torch.bfloat16:
            return (
                f"layer {i}: {name} is {w.dtype}, neither bf16 nor in a "
                "quantized layout the kernels read"
            )
    return None


def gdn_core(attn, x) -> torch.Tensor:
    """``QwenGatedDeltaNetAttention.forward_hip`` up to out_proj's input: the
    input projections, AITER's conv + gated delta rule (vLLM's spec-decode
    path on verify steps) and the gated norm -> [S, heads HD]."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _encode_layer_name,
    )

    n = x.size(0)
    qkvz, _ = attn.in_proj_qkvz(x)
    ba, _ = attn.in_proj_ba(x)
    shape = (n, attn.num_v_heads // attn.tp_size, attn.head_v_dim)
    core = torch.empty(shape, dtype=x.dtype, device=x.device)
    z = torch.empty(shape, dtype=qkvz.dtype, device=x.device)
    torch.ops.vllm.qwen_gdn_attention_core(
        qkvz.view(n, -1),
        ba.view(n, -1),
        z,
        core,
        layer_name=_encode_layer_name(attn.prefix),
        use_aiter=True,
    )
    norm = attn.norm
    return _gated_rms_norm(core, z, norm.weight, norm.eps, norm.activation).flatten(-2)


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _gated_rms_norm(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, act: str
) -> torch.Tensor:
    """``RMSNormGated.forward_native`` (no groups, the norm before the gate) in
    one launch: a verify step's forward is not torch.compiled, so the CustomOp
    runs as ~8 eager kernels a GDN layer."""
    return RMSNormGated.forward_static(x, z, weight, eps, x.dtype, activation=act)


class QwenMonoDecode:
    """Bound to one Qwen model; built at model init on every rank."""

    LABEL: ClassVar[str]  # the model, in errors and log lines
    TP_SIZES: ClassVar[tuple[int, ...]]
    MAX_TOKENS: ClassVar[int]
    BLOCKS: ClassVar[int]  # a launch's CTAs, each resident
    MTP: ClassVar[bool]  # verify steps taken

    def __init__(self, model, vllm_config):
        from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer

        self.model = model
        self.spec = vllm_config.speculative_config is not None
        refused = self.refusals(vllm_config)
        if refused:
            raise ValueError(f"VLLM_ROCM_MONO_DECODE {'; '.join(refused)}.")
        why = self.unsupported(model)
        self.ok = why is None
        self._weights_checked = False
        if not self.ok:
            logger.info_once("%s mono decode off for this model: %s", self.LABEL, why)
            return
        tp = get_tp_group()
        self.tp, self.rank = tp.world_size, tp.rank_in_group
        self._probe = next(
            ly.linear_attn.prefix
            for ly in model.layers
            if ly.layer_type == "linear_attention"
        )
        dev = model.embed_tokens.weight.device

        def buf(n: int) -> torch.Tensor:
            return torch.zeros((n + 255) // 256 * 256, dtype=torch.uint8, device=dev)

        k1_bytes, k2_bytes, peer_bytes = self.buffer_bytes()
        widths = range(1, self.MAX_TOKENS + 1)
        self.k1_scratch = {s: buf(k1_bytes(s)) for s in widths}
        self.k2_scratch = {s: buf(k2_bytes(s)) for s in widths}
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        # collective: every rank builds the model, so every rank gets here
        self.peers = PeerBuffer(
            peer_bytes, tp.cpu_group, tp.rank_in_group, tp.world_size, dev
        )
        self.peers.bytes.zero_()
        self.warm_up(dev)
        logger.info_once(
            "%s mono decode on for %s steps of <= %d rows",
            self.LABEL,
            "verify" if self.spec else "decode",
            self.MAX_TOKENS,
        )

    def refusals(self, vllm_config) -> list[str]:
        """Every reason the kernels cannot run this deployment: an explicit
        opt-in that cannot run is an error."""
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            GDN_AITER_TRITON_AVAILABLE,
        )
        from vllm.platforms.rocm import get_cdna_version

        pc = vllm_config.parallel_config
        why = []
        if get_cdna_version() not in CDNA_VERSIONS:
            why.append(f"needs CDNA{'/'.join(map(str, CDNA_VERSIONS))}")
        if get_tp_group().world_size not in self.TP_SIZES:
            why.append(
                f"needs tensor parallel size {'/'.join(map(str, self.TP_SIZES))}"
            )
        if get_pp_group().world_size != 1:
            why.append("needs no pipeline parallelism")
        if pc.enable_expert_parallel or pc.enable_eplb or pc.data_parallel_size > 1:
            why.append("needs tensor parallelism only: no expert or data parallelism")
        if pc.decode_context_parallel_size > 1 or pc.prefill_context_parallel_size > 1:
            why.append("needs no context parallelism")
        if vllm_config.lora_config is not None:
            why.append("needs no LoRA")
        spec = vllm_config.speculative_config
        if spec is not None:
            if not self.MTP:
                why.append("needs no speculative decoding")
            elif spec.method != "mtp":
                why.append("needs MTP or no speculative decoding")
            elif spec.num_speculative_tokens + 1 > self.MAX_TOKENS:
                why.append(f"needs at most {self.MAX_TOKENS - 1} MTP draft tokens")
            elif not GDN_AITER_TRITON_AVAILABLE:
                why.append("MTP needs AITER's Triton GDN kernels")
        if vllm_config.kv_transfer_config is not None:
            # a KV connector's copies on another stream can hold CUs a resident
            # grid spin-waits on
            why.append("needs no KV connector")
        cus = current_platform.num_compute_units(
            torch.accelerator.current_device_index()
        )
        if cus < self.BLOCKS:
            # every CTA of a launch stays resident: a partitioned GPU deadlocks
            why.append(f"needs {self.BLOCKS} compute units, the GPU has {cus}")
        return why

    def unsupported(self, model) -> str | None:
        """Why the kernels cannot serve this model variant, else None."""
        raise NotImplementedError

    def buffer_bytes(self):
        """(K1's scratch bytes at a width, K2's, the peer buffer's)."""
        raise NotImplementedError

    def warm_up(self, dev) -> None:
        """Compile the host-side helpers here: a decode graph's first call can
        be its capture."""

    def weights_unsupported(self) -> str | None:
        raise NotImplementedError

    def check_weights(self) -> None:
        """At the first decode step, after loading: every weight in the dtype
        and layout the kernels read, else an error."""
        if not self._weights_checked:
            why = self.weights_unsupported()
            if why is not None:
                raise ValueError(f"VLLM_ROCM_MONO_DECODE {why}.")
            self._weights_checked = True

    def eligible(self, input_ids, intermediate_tensors) -> bool:
        """A pure decode step (with MTP, a target-verify step) of at most
        ``MAX_TOKENS`` rows."""
        if not self.ok or intermediate_tensors is not None:
            return False
        # A piecewise step runs (or captures) the compiled stock model's graphs;
        # taking it here would leave them uncaptured for the prefills of that size.
        if in_piecewise_cudagraph():
            return False
        if input_ids is None or not 1 <= input_ids.size(0) <= self.MAX_TOKENS:
            return False
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return False
        m: Any = md.get(self._probe)
        if m is None or m.num_prefills != 0:
            return False
        if self.spec:
            if m.num_decodes + m.num_spec_decodes < 1:
                return False
        else:
            if m.spec_sequence_masks is not None:
                return False
            if m.num_spec_decodes != 0 or m.num_decodes < 1:
                return False
            idx = m.non_spec_state_indices_tensor
            # the graph's pad rows read NULL_BLOCK_ID entries: the indices cover
            # every row
            if idx is None or idx.dtype != torch.int32:
                return False
            if idx.numel() < input_ids.size(0) or not idx.is_contiguous():
                return False
        self.check_weights()
        return True

    def first_step(self, s: int) -> None:
        logger.info_once(
            "%s mono decode: first mono %s step (%d rows)",
            self.LABEL,
            "verify" if self.spec else "decode",
            s,
        )
