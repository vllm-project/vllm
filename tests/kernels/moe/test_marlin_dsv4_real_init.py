# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin NVFP4 MoE INIT validation on dsv4's REAL stored weights (all archs).

The MarlinMoE clamp test proves the kernel math on weights we quantize ourselves.
This test closes the other gap: that prepare_moe_fp4_layer_for_marlin correctly
INITIALIZES from dsv4's ACTUAL stored format -- real packed fp4 weights, real
F8_E4M3 LINEAR block scales, real per-tensor weight_scale_2 -- loaded at runtime
from the mounted nvidia/DeepSeek-V4-Flash-NVFP4 checkpoint (not hardcoded).

It runs Marlin's real prepare + kernel at dsv4's clamp (swiglu_limit=10) on the
real expert tensors, and compares to a torch dequant of the SAME tensors with
dsv4's ground-truth clamp (SiluAndMulWithClamp alpha=1, beta=0 -- exactly
Expert.forward in the checkpoint's inference/model.py). Agreement validates that
Marlin reads dsv4's weight format correctly end to end.

dsv4 experts: w1=gate, w3=up (chunked), w2=down; k=4096, n=2048, group 16.

REFERENCE (what we WANT -> IMPLEMENT -> TEST):
  Model (quantized): nvidia/DeepSeek-V4-Flash-NVFP4
      https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4
  Base model: deepseek-ai/DeepSeek-V4-Flash
      https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
  Ground-truth SwiGLU clamp (reviewed + confirmed): the checkpoint's reference
  implementation inference/model.py, class Expert.forward, lines 596-606
  (gate=clamp(w1(x),max=L); up=clamp(w3(x),-L,L); F.silu(gate)*up; w2),
  swiglu_limit=10.0 (config.json) == vLLM SiluAndMulWithClamp(L, alpha=1, beta=0)
  (vllm/model_executor/layers/activation.py). Real weights loaded from this
  checkpoint at runtime; quant scheme/scales from hf_quant_config.json +
  safetensors.
"""

import os

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_moe
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.custom_op import CustomOp, op_registry
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(75):
    pytest.skip("Marlin requires compute capability 7.5+", allow_module_level=True)

# Opt-in: point DSV4_MODEL_DIR at a local nvidia/DeepSeek-V4-Flash-NVFP4
# checkpoint. Skips otherwise (so CI without the checkpoint skips cleanly).
_MODEL_DIR = os.environ.get("DSV4_MODEL_DIR")
if not _MODEL_DIR or not os.path.isdir(_MODEL_DIR):
    pytest.skip(
        "set DSV4_MODEL_DIR to a DeepSeek-V4-Flash-NVFP4 checkpoint to run this test",
        allow_module_level=True,
    )

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_BLOCK = 16
_SWIGLU_LIMIT = 10.0  # dsv4 config.swiglu_limit
_LAYER = 0
_E = 8  # subset of experts (init/format is per-expert identical)
_TOPK = 6  # dsv4 num_experts_per_tok

_CLAMP_OP_NAME = "dsv4_real_silu_clamp_10"
if _CLAMP_OP_NAME not in op_registry:

    @CustomOp.register(_CLAMP_OP_NAME)
    class _ClampOp(SiluAndMulWithClamp):
        custom_op_name = _CLAMP_OP_NAME

        def __init__(self, *, compile_native: bool = False) -> None:
            super().__init__(_SWIGLU_LIMIT, compile_native=compile_native)


SILU_WITH_CLAMP = op_registry[_CLAMP_OP_NAME]


def _load_real_experts(device):
    """Load real dsv4 layer-0 experts 0.._E-1 from the checkpoint."""
    import json

    from safetensors import safe_open

    base = _MODEL_DIR
    assert base is not None  # narrowed by the module-level skip guard
    with open(os.path.join(base, "model.safetensors.index.json")) as f:
        idx = json.load(f)["weight_map"]
    handles = {}

    def get(name):
        shard = idx[name]
        if shard not in handles:
            handles[shard] = safe_open(
                os.path.join(base, shard), framework="pt", device=str(device)
            )
        return handles[shard].get_tensor(name)

    w13_q, w2_q, w13_s, w2_s, w13_g, w2_g = [], [], [], [], [], []
    for i in range(_E):
        p = f"layers.{_LAYER}.ffn.experts.{i}"
        w1, w3 = get(f"{p}.w1.weight"), get(f"{p}.w3.weight")
        w1s, w3s = get(f"{p}.w1.weight_scale"), get(f"{p}.w3.weight_scale")
        # chunked w13 = [gate(w1); up(w3)] along output rows
        w13_q.append(torch.cat([w1, w3], dim=0))
        w13_s.append(torch.cat([w1s, w3s], dim=0))
        w2_q.append(get(f"{p}.w2.weight"))
        w2_s.append(get(f"{p}.w2.weight_scale"))
        # w1 and w3 share the same per-tensor weight_scale_2 in dsv4
        w13_g.append(get(f"{p}.w1.weight_scale_2").reshape(()))
        w2_g.append(get(f"{p}.w2.weight_scale_2").reshape(()))
    return (
        torch.stack(w13_q),
        torch.stack(w2_q),
        torch.stack(w13_s),
        torch.stack(w2_s),
        torch.stack(w13_g).float(),
        torch.stack(w2_g).float(),
    )


def _linear_dequant(packed_u8, block_scale_f8, gs2, dtype):
    """Dequant dsv4's LINEAR-layout NVFP4 weights: fp4 * block_scale * gs2."""
    rows, pk = packed_u8.shape
    cols = pk * 2
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    nib = torch.stack([low, high], dim=-1).reshape(rows, cols).long()
    sign = torch.where((nib & 0x08) > 0, -1.0, 1.0)
    levels = torch.tensor(_E2M1, device=packed_u8.device)
    vals = (levels[nib & 0x07] * sign).reshape(rows, cols // _BLOCK, _BLOCK)
    bs = block_scale_f8.float()
    out = (vals * (bs * gs2).unsqueeze(-1)).reshape(rows, cols)
    return out.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_marlin_dsv4_real_init(dtype, workspace_init):
    set_random_seed(7)
    device = torch.device("cuda")
    w13_q, w2_q, w13_s, w2_s, w13_g, w2_g = _load_real_experts(device)
    e = _E
    two_n, half_k = w13_q.shape[1], w13_q.shape[2]
    n, k = two_n // 2, half_k * 2
    topk, m = _TOPK, 64
    assert (n, k) == (2048, 4096), f"unexpected dsv4 dims n={n} k={k}"

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        # Activations sized so the gemm1 outputs reach the clamp (engage it).
        a = torch.randn((m, k), device=device, dtype=dtype) * 3.0
        score = torch.randn((m, e), device=device, dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        # --- Marlin: real prepare + kernel on the REAL stored tensors ---
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(w13_q, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_q, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_s, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_s, requires_grad=False)
        layer.w13_weight_scale_2 = torch.nn.Parameter(w13_g, requires_grad=False)
        layer.w2_weight_scale_2 = torch.nn.Parameter(w2_g, requires_grad=False)
        layer.params_dtype = dtype
        layer.moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            in_dtype=dtype,
        )
        prepare_moe_fp4_layer_for_marlin(layer)
        marlin_out = fused_marlin_moe(
            hidden_states=a,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            bias1=None,
            bias2=None,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=e,
            expert_map=None,
            global_scale1=layer.w13_weight_scale_2,
            global_scale2=layer.w2_weight_scale_2,
            workspace=layer.workspace,
            activation=MoEActivation.SILU,
            input_dtype=None,
            clamp_limit=_SWIGLU_LIMIT,
        )

        # --- torch reference: dequant the SAME real tensors + dsv4's clamp ---
        w13_dq = torch.stack(
            [_linear_dequant(w13_q[i], w13_s[i], w13_g[i], dtype) for i in range(e)]
        )
        w2_dq = torch.stack(
            [_linear_dequant(w2_q[i], w2_s[i], w2_g[i], dtype) for i in range(e)]
        )
        ref = torch_moe(a, w13_dq, w2_dq, score, topk, activation=SILU_WITH_CLAMP)

        mo = marlin_out.float().flatten()
        ro = ref.float().flatten()
        rel = (mo - ro).norm() / ro.norm().clamp_min(1e-6)
        print(
            f"[diag] dsv4-real-init marlin_norm={mo.norm():.4f} "
            f"ref_norm={ro.norm():.4f} rel_L2={rel:.5f}"
        )
        assert rel < 0.05, (
            f"Marlin diverges from a dequant of dsv4's REAL weights "
            f"(rel_L2={rel:.5f}); Marlin init/clamp on dsv4 format is wrong."
        )
