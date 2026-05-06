# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helper utilities for ``test_mla_aiter_qk_rope_kvcache_fusion.py``.

Kept separate from the test file so the tests stay focused on assertions and
this file can hold all the boilerplate for building a small DSV3-shaped MLA
layer + attention metadata.
"""
from __future__ import annotations

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.passes.fusion.mla_aiter_qk_rope_kvcache_fusion import (
    MLAAiterQkRopeKVCacheFusionPass,
)
from vllm.compilation.passes.fusion.mla_decode_q_prep_lift import (
    MLADecodeQPrepLiftPass,
)
from vllm.compilation.passes.fusion.mla_rope_kvcache_cat_fusion import (
    MLARoPEKVCacheCatFusionPass,
)
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.rotary_embedding import DeepseekScalingRotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum

FP8_DTYPE = current_platform.fp8_dtype()

# DSV3-Lite-shaped tiny config (small enough to run on modest GPUs).
NUM_HEADS = 16
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
BLOCK_SIZE = 16


def _ensure_dist_init() -> None:
    if torch.distributed.is_initialized():
        return
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    update_environment_variables(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "54321",
        }
    )
    init_distributed_environment()
    initialize_model_parallel()


def _make_vllm_config(
    kv_cache_dtype: str = "auto",
    fuse_aiter: bool = True,
    max_num_seqs: int = 128,
) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(
            model="deepseek-ai/DeepSeek-V2-Lite",
            dtype=torch.bfloat16,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            cache_dtype=kv_cache_dtype,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            # max_num_batched_tokens must be >= max_num_seqs for the
            # rocm_aiter_mla builder's pre-allocations to fit decode-only
            # batches that span the full request budget.
            max_num_batched_tokens=max(2048, max_num_seqs * 8),
            max_model_len=4096,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(
                fuse_rope_kvcache_cat_mla=True,
                fuse_aiter_qk_rope_kvcache_mla=fuse_aiter,
                eliminate_noops=True,
            ),
        ),
    )


class _MLATestModel(torch.nn.Module):
    """Tiny DSV3-shaped MLA test model: q-prep + RoPE + KV-cache write +
    MLAAttention.forward (lift target). Mirrors the structure of the PR
    #40392 test model but adds the attention call so the lift pass and AITER
    fusion can fire."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        is_neox: bool,
        device: torch.device,
        prefix: str = "model.layers.0.self_attn.attn",
    ):
        super().__init__()
        self.dtype = torch.bfloat16
        self.device = device
        self.layer_name = prefix
        self.qk_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
        self.head_size = KV_LORA_RANK + QK_ROPE_HEAD_DIM
        self.scale = self.qk_head_dim**-0.5

        self.rotary_emb = DeepseekScalingRotaryEmbedding(
            head_size=QK_ROPE_HEAD_DIM,
            rotary_dim=QK_ROPE_HEAD_DIM,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=is_neox,
            scaling_factor=1.0,
            dtype=self.dtype,
        )

        self.q_b_proj = ColumnParallelLinear(
            Q_LORA_RANK,
            NUM_HEADS * self.qk_head_dim,
            bias=False,
            prefix=f"{prefix}.q_b_proj",
        ).to(device)
        self.kv_b_proj = ColumnParallelLinear(
            KV_LORA_RANK,
            NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
            bias=False,
            prefix=f"{prefix}.kv_b_proj",
        ).to(device)
        with torch.no_grad():
            torch.nn.init.normal_(self.q_b_proj.weight, std=0.02)
            torch.nn.init.normal_(self.kv_b_proj.weight, std=0.02)

        # The actual attention layer that will be lifted around.
        self.mla_attn = MLAAttention(
            num_heads=NUM_HEADS,
            scale=self.scale,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            kv_b_proj=self.kv_b_proj,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
            attn_backend=AttentionBackendEnum.ROCM_AITER_MLA.get_class(),
        )
        self.attn_backend: type[AttentionBackend] = self.mla_attn.get_attn_backend()
        self.mla_attn._k_scale = self.mla_attn._k_scale.to(device)
        self.mla_attn._v_scale = self.mla_attn._v_scale.to(device)
        self.mla_attn._q_scale = self.mla_attn._q_scale.to(device)
        self.mla_attn.process_weights_after_loading(self.dtype)

        self.kv_cache_dtype_str = vllm_config.cache_config.cache_dtype
        self.kv_cache_dtype = (
            FP8_DTYPE if self.kv_cache_dtype_str.startswith("fp8") else self.dtype
        )

        self.builder = self.attn_backend.get_builder_cls()(
            kv_cache_spec=self.mla_attn.get_kv_cache_spec(vllm_config),
            layer_names=[self.mla_attn.layer_name],
            vllm_config=vllm_config,
            device=device,
        )

    def build_attn_metadata(self, batch_size: int) -> CommonAttentionMetadata:
        batch_spec = BatchSpec(
            seq_lens=[1] * batch_size, query_lens=[1] * batch_size
        )
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, BLOCK_SIZE, self.device, arange_block_indices=True
        )
        max_blocks = (max(batch_spec.seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = batch_size * max_blocks
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, BLOCK_SIZE, 1, self.head_size
        )
        try:
            kv_cache_stride_order = self.attn_backend.get_kv_cache_stride_order()
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
        kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]
        raw_tensor = torch.zeros(
            num_blocks * BLOCK_SIZE * self.head_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
        )
        kv_cache = raw_tensor.view(kv_cache_shape).permute(*inv_order)
        self.mla_attn.kv_cache = kv_cache
        return self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def forward(
        self, qkv_lora: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        qkv_lora = qkv_lora.clone()
        q_c, kv_lora = qkv_lora.split(
            [Q_LORA_RANK, KV_LORA_RANK + QK_ROPE_HEAD_DIM], dim=-1
        )
        q = self.q_b_proj(q_c)[0]
        kv_c, k_pe = kv_lora.split([KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)

        q = q.view(-1, NUM_HEADS, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)

        q[..., QK_NOPE_HEAD_DIM:], k_pe = self.rotary_emb(
            positions, q[..., QK_NOPE_HEAD_DIM:], k_pe
        )

        T = qkv_lora.shape[0]
        out = torch.empty(T, NUM_HEADS * V_HEAD_DIM, dtype=self.dtype, device=self.device)
        return self.mla_attn(
            q, kv_c, k_pe, output_shape=(T, NUM_HEADS * V_HEAD_DIM)
        )


def build_test_layer_and_inputs(T: int) -> tuple[MLAAttention, torch.Tensor]:
    """Build a tiny MLA layer and a random ``q`` tensor of shape
    ``[T, NUM_HEADS, qk_nope + qk_rope]`` for unit testing."""
    device = torch.device("cuda")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    vllm_config = _make_vllm_config(kv_cache_dtype="fp8")
    with vllm.config.set_current_vllm_config(vllm_config):
        _ensure_dist_init()
        model = _MLATestModel(vllm_config, is_neox=True, device=device)
    layer = model.mla_attn
    q = torch.randn(T, NUM_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM, device=device)
    return layer, q


def _setup_aiter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()


def run_parity_check(
    T: int,
    kv_cache_dtype: str,
    is_neox: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_aiter_env(monkeypatch)
    device = torch.device("cuda")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    vllm_config = _make_vllm_config(kv_cache_dtype=kv_cache_dtype)

    with vllm.config.set_current_vllm_config(vllm_config):
        _ensure_dist_init()
        model = _MLATestModel(vllm_config, is_neox=is_neox, device=device)

        lift_pass = MLADecodeQPrepLiftPass(vllm_config)
        fusion_pass = MLAAiterQkRopeKVCacheFusionPass(vllm_config)
        rope_kvcache_pass = MLARoPEKVCacheCatFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            rope_kvcache_pass,
            lift_pass,
            fusion_pass,
            PostCleanupPass(vllm_config),
            FixFunctionalizationPass(vllm_config),
        ]
        backend = TestBackend(*passes)

        qkv_lora = torch.randn(
            T,
            Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
        )
        pos = torch.arange(T, dtype=torch.long)

        # Unfused
        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            out_unfused = model(qkv_lora.clone(), pos.clone())

        # Fused (compiled)
        torch._dynamo.mark_dynamic(qkv_lora, 0)
        torch._dynamo.mark_dynamic(pos, 0)
        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(T)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            compiled = torch.compile(model, backend=backend)
            out_fused = compiled(qkv_lora, pos)

    # The AITER fusion should have fired exactly once.
    assert fusion_pass.matched_count == 1, (
        f"Expected fusion_pass.matched_count == 1, got "
        f"{fusion_pass.matched_count}. Lift / RoPE+KVCache passes may have "
        "rewritten the graph in an unexpected way."
    )

    atol, rtol = (1e-2, 1e-2) if kv_cache_dtype.startswith("fp8") else (5e-3, 5e-3)
    torch.testing.assert_close(out_unfused, out_fused, atol=atol, rtol=rtol)


def run_cuda_graph_capture_replay(
    compile_size: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Compile + capture + replay at a single fixed bucket size.

    The test uses ``compile_sizes=[compile_size]`` so dynamo specializes
    ``q.size(0) == compile_size`` exactly. INVARIANT 1 + the lift-pass gate
    together guarantee the fake_impl shape ``[compile_size, ...]`` is honest
    and CUDA-graph-replayable.
    """
    _setup_aiter_env(monkeypatch)
    device = torch.device("cuda")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    vllm_config = _make_vllm_config(
        kv_cache_dtype="fp8",
        max_num_seqs=max(128, compile_size),
    )
    # Force the lift / fusion gate to admit this compile_size.
    vllm_config.compilation_config.pass_config.aiter_qk_rope_kvcache_fusion_max_token_num = max(
        compile_size, 256
    )

    with vllm.config.set_current_vllm_config(vllm_config):
        _ensure_dist_init()
        model = _MLATestModel(vllm_config, is_neox=True, device=device)

        lift_pass = MLADecodeQPrepLiftPass(vllm_config)
        fusion_pass = MLAAiterQkRopeKVCacheFusionPass(vllm_config)
        rope_kvcache_pass = MLARoPEKVCacheCatFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            rope_kvcache_pass,
            lift_pass,
            fusion_pass,
            PostCleanupPass(vllm_config),
            FixFunctionalizationPass(vllm_config),
        ]
        backend = TestBackend(*passes)

        qkv_lora = torch.randn(
            compile_size,
            Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
        )
        pos = torch.arange(compile_size, dtype=torch.long)

        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(compile_size)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            compiled = torch.compile(model, backend=backend, dynamic=False)
            # Warmup pass — this is where the discarded PR's fault occurred.
            out1 = compiled(qkv_lora.clone(), pos.clone())
            # Replay: same shape, fresh data.
            out2 = compiled(
                qkv_lora.clone() + 0.01,
                pos.clone(),
            )

    # Sanity: outputs are finite, shapes match.
    assert torch.isfinite(out1).all(), "Captured-graph output contains NaN/Inf."
    assert torch.isfinite(out2).all(), "Replayed-graph output contains NaN/Inf."
    assert out1.shape == out2.shape

    # No inductor "shape mismatch" / "re-record" log spam.
    suspicious = ("shape mismatch", "re-record", "size mismatch")
    for record in caplog.records:
        msg = record.getMessage().lower()
        for s in suspicious:
            assert s not in msg, (
                f"Found suspicious log line during CUDA-graph capture/replay: "
                f"{record.getMessage()!r}. This is the symptom class of the "
                "discarded PR's (4682, 16384) fault."
            )
