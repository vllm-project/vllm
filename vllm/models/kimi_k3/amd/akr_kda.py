# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The amd_kernels `kda` route: the K3 KDA spec-verify decode step (f_b GEMM + causal
conv + delta-rule recurrence + gated RMSNorm) as one launch, entry
`attention/kda_common.kda_decode_fb_gate_spec`.

Replaces, on a spec-only decode batch, `kimi_k3_kda_fb_proj` / `f_b_proj`
(KimiK3DeltaAttention.forward) and AITER `kda_fused_recurrent_spec_decode`
(KimiK3DeltaAttention._forward). A call the entry's
`kda_decode_fb_gate_spec_unsupported_reason` rejects keeps the stock chain: the caller
then computes g1 itself.
Sub-switch: VLLM_KIMI_AKR_KDA_DECODE=0 turns it off.
"""

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_config, akr_switch

logger = init_logger(__name__)


def _lib():
    from amd_kernels.attention import kda_common

    return kda_common


def layer_unsupported_reason(layer) -> str | None:
    """None when this KDA layer can run the kernel (static properties), else why not."""
    lib = _lib()
    if not lib.available():
        return "needs gfx950 with FlyDSL"
    key = (layer.local_num_heads, layer.head_dim)
    if key not in lib.kda_decode_fb_gate_spec_shapes():
        return f"no kda_decode_fb_gate_spec for (heads, head_dim)={key}"
    if layer.conv_size != 4:
        return f"needs conv width 4 (got {layer.conv_size})"
    if layer.decode_conv1d_weight is None:
        return "no width-major decode conv weight"
    if layer.gate_lower_bound is None:
        return "needs the safe-gate lower bound"
    if not layer.use_full_rank_gate:
        return "needs the full-rank gate"
    return None


def setup(layer) -> bool:
    """Whether `layer` (KimiK3DeltaAttention) runs the route; call once at
    construction."""
    if not akr_switch.on("KDA_DECODE"):
        return False
    if akr_config.unsupported_reason("kda", layer_unsupported_reason(layer)):
        return False
    logger.info_once(
        "amd_kernels kda: spec decode runs on kda_common.kda_decode_fb_gate_spec"
    )
    return True


def run(
    layer,
    proj: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    m,
    num_accepted_tokens: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> bool:
    """Run the kernel for a spec-only decode batch (`m` = GDNAttentionMetadata); False
    (nothing launched) when this call does not fit it. `proj` is the fused in_proj
    output [rows, >= 6288] (q|k|v|gate|f_a|beta); the results land in `core_attn_out`
    [1, rows, 12, 128] (gate, norm and all) and the conv / recurrent states in place."""
    spec_state_indices = m.spec_state_indices_tensor
    spec_cu = m.spec_query_start_loc
    n_seq = m.num_spec_decodes
    if (
        spec_state_indices is None
        or spec_cu is None
        or n_seq <= 0
        or num_accepted_tokens is None
    ):
        return False
    lib = _lib()
    n_rows = m.num_actual_tokens
    t = spec_state_indices.shape[1]
    uniform = m.uniform_spec_sequence_length == t and n_rows == n_seq * t
    args = (
        proj[:n_rows],  # (graph padding rows past the step's tokens are not read)
        layer.f_b_proj.weight,
        layer.A_log,
        layer.dt_bias,
        layer.gate_lower_bound,
        layer.decode_conv1d_weight,
        conv_state,
        recurrent_state,
        spec_cu[: n_seq + 1],
        spec_state_indices[:n_seq],
        num_accepted_tokens[:n_seq],
        layer.o_norm.weight,
        layer.o_norm.eps,
        core_attn_out[0, :n_rows],
    )
    reason = lib.kda_decode_fb_gate_spec_unsupported_reason(*args, uniform=uniform)
    if akr_config.call_unsupported("kda", reason):
        return False
    lib.kda_decode_fb_gate_spec(*args, uniform=uniform)
    return True


def warm_up(layer, tokens_per_seq: int) -> None:
    """Compile the kernel variants a spec decode step of `tokens_per_seq` rows per
    request can take (uniform / ragged) and launch each once on a padding sequence
    (state slot 0: nothing is read or written), outside any CUDA graph capture."""
    if not 1 <= tokens_per_seq <= 8:
        return
    lib = _lib()
    dev = layer.decode_conv1d_weight.device
    t = tokens_per_seq
    h, d = layer.local_num_heads, layer.head_dim
    rows, ncols = t, 6288 + (-6288 % 16)
    proj = torch.zeros(rows, ncols, dtype=torch.bfloat16, device=dev)
    conv = torch.zeros(2, t + 2, 3 * h * d, dtype=torch.bfloat16, device=dev).transpose(
        -1, -2
    )
    state = torch.zeros(2, h, d, d, dtype=torch.float32, device=dev)
    idx = torch.zeros(1, t, dtype=torch.int32, device=dev)
    acc = torch.ones(1, dtype=torch.int32, device=dev)
    cu = torch.tensor([0, t], dtype=torch.int32, device=dev)
    out = torch.empty(rows, h, d, dtype=torch.bfloat16, device=dev)
    for uniform in (False, True):
        lib.kda_decode_fb_gate_spec(
            proj,
            layer.f_b_proj.weight,
            layer.A_log,
            layer.dt_bias,
            layer.gate_lower_bound,
            layer.decode_conv1d_weight,
            conv,
            state,
            cu,
            idx,
            acc,
            layer.o_norm.weight,
            layer.o_norm.eps,
            out,
            uniform=uniform,
        )
    torch.accelerator.synchronize()
