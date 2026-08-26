# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from importlib.metadata import version

import pytest
import torch
from packaging.version import Version

import vllm._aiter_ops as aiter_ops_module
import vllm.compilation.passes.fusion.qk_norm_rope_kvcache_fusion as fusion_module
import vllm.config
import vllm.v1.attention.backends.rocm_aiter_unified_attn as rocm_aiter_backend
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    dense_kv_cache_views,
)
from vllm._aiter_ops import (
    is_aiter_found_and_supported,
    is_aiter_mrope_strided_kv_cache_supported,
    rocm_aiter_ops,
)
from vllm.compilation.passes.fusion.matcher_utils import MROPE_OP, ROTARY_OP
from vllm.compilation.passes.fusion.qk_norm_rope_kvcache_fusion import (
    QkNormRopeKvCacheFusionPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.compilation.passes.utility.scatter_split_replace import (
    ScatterSplitReplacementPass,
)
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

pytestmark = pytest.mark.skip_global_cleanup

if not is_aiter_found_and_supported():
    pytest.skip(
        "ROCm with supported AITER is required",
        allow_module_level=True,
    )

INDEX_SELECT_OP = torch.ops.aten.index.Tensor
FP8_DTYPE = current_platform.fp8_dtype()


@pytest.fixture(scope="module", autouse=True)
def module_global_cleanup():
    # Cleanup once at the end of the module
    from vllm.distributed import cleanup_dist_env_and_memory

    yield
    cleanup_dist_env_and_memory()


class QKNormRoPEKVCacheTestModel(torch.nn.Module):
    """Minimal model that reproduces the QK-norm + RoPE + KV cache update
    pattern matched by QkNormRopeKvCacheFusionPass:

        q, k, v = split(qkv)
        q = rms_norm(q.view(heads, dim), q_weight).view(flat)
        k = rms_norm(k.view(heads, dim), k_weight).view(flat)
        q, k = rotary_emb(positions, q, k)
        q = q.view(num_heads, head_dim)
        k = k.view(num_kv_heads, head_dim)
        v = v.view(num_kv_heads, head_dim)
        dummy = unified_kv_cache_update(k, v, layer_name)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        attn_backend: AttentionBackendEnum,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        is_neox: bool,
        rms_norm_eps: float,
        dtype: torch.dtype,
        device: torch.device,
        rotary_dim: int | None = None,
        prefix: str = "model.layers.0.self_attn.attn",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.rotary_dim = rotary_dim if rotary_dim is not None else head_size
        self.block_size = vllm_config.cache_config.block_size
        self.q_size = num_heads * head_size
        self.kv_size = num_kv_heads * head_size
        self.is_neox = is_neox
        self.dtype = dtype
        self.device = device
        self.layer_name = prefix

        self.q_norm = RMSNorm(head_size, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_size, eps=rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(
            head_size,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=is_neox,
            dtype=self.dtype,
        )

        self.enable_rope_custom_op = self.rotary_emb.enabled()

        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 / head_size**0.5,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
            attn_backend=attn_backend.get_class(),
        )
        self.attn_backend: type[AttentionBackend] = self.attn.get_attn_backend()
        assert not self.attn_backend.forward_includes_kv_cache_update, (
            f"Attention backend {self.attn_backend} does not support "
            "fuse_qk_norm_rope_kvcache."
        )
        kv_cache_dtype_str = vllm_config.cache_config.cache_dtype
        self.kv_cache_dtype = (
            FP8_DTYPE if kv_cache_dtype_str.startswith("fp8") else self.dtype
        )

        if self.kv_cache_dtype != self.dtype:
            self.attn._k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
            self.attn._v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
            self.attn._k_scale_float = 1.0
            self.attn._v_scale_float = 1.0
        else:
            self.attn._k_scale = self.attn._k_scale.to(device)
            self.attn._v_scale = self.attn._v_scale.to(device)

        # Mirror the worker spec-collection loop: the backend publishes the
        # packing it needs.
        self.kv_cache_spec = self.attn.attn_backend.customize_spec(
            AttentionSpec(
                block_size=self.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=head_size,
                dtype=self.kv_cache_dtype,
            )
        )
        self.builder = self.attn.attn_backend.get_builder_cls()(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=[self.attn.layer_name],
            vllm_config=vllm_config,
            device=device,
        )

    def build_attn_metadata(
        self, batch_size: int, layout: KVCacheLayout
    ) -> CommonAttentionMetadata:
        batch_spec = BatchSpec(seq_lens=[1] * batch_size, query_lens=[1] * batch_size)
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )

        max_blocks = (max(batch_spec.seq_lens) + self.block_size - 1) // self.block_size
        num_blocks = batch_size * max_blocks

        raw_tensor = torch.zeros(
            num_blocks * self.kv_cache_spec.page_size_bytes,
            dtype=torch.int8,
            device=self.device,
        )
        kv_cache = dense_kv_cache_views(
            raw_tensor,
            self.kv_cache_spec,
            num_blocks,
            num_layers=1,
            layout=layout,
        )[0]

        # Store as a bare tensor (not wrapped in a list) to match production
        # `bind_kv_cache` behavior.  `get_attention_context` returns this
        # attribute directly to the fused/unfused `do_kv_cache_update` impls,
        # which call `kv_cache.unbind(0)` and therefore require a tensor.
        self.attn.kv_cache = kv_cache

        attn_metadata = self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

        return attn_metadata

    def forward(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = qkv.clone()
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-norm: RMSNorm on per-head Q and K
        q = q.view(-1, self.num_heads, self.head_size)
        q = self.q_norm(q)
        q = q.view(-1, self.q_size)

        k = k.view(-1, self.num_kv_heads, self.head_size)
        k = self.k_norm(k)
        k = k.view(-1, self.kv_size)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Mirror Attention.forward: quant-query impls consume an fp8 q.
        if (
            self.kv_cache_dtype != self.dtype
            and self.attn.impl.supports_quant_query_input
        ):
            q_fp8 = torch.empty_like(q, dtype=FP8_DTYPE)
            torch.ops.vllm.rocm_aiter_per_tensor_quant(
                q_fp8, q, self.attn._q_scale, False
            )
            q = q_fp8

        # Final views + KV cache update
        q = q.view(-1, self.num_heads, self.head_size)
        k = k.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size)
        kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
            k, v, _encode_layer_name(self.layer_name)
        )
        return q, k, v, kv_cache_dummy_dep

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        ops: list[torch._ops.OpOverload] = []
        # RoPE is not yet IR-migrated, so its custom op still surfaces
        # directly in the graph based on `enable_rope_custom_op`.
        if self.enable_rope_custom_op:
            if rocm_aiter_ops.is_triton_rotary_embed_enabled():
                ops.append(torch.ops.vllm.rocm_aiter_triton_rotary_embedding.default)
            else:
                ops.append(ROTARY_OP)
        else:
            ops.append(INDEX_SELECT_OP)
        ops.append(torch.ops.vllm.unified_kv_cache_update.default)
        return ops

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.vllm.fused_qk_norm_rope_and_unified_kv_cache_update.default]


class QKNormMRoPEKVCacheTestModel(QKNormRoPEKVCacheTestModel):
    def __init__(
        self,
        *,
        mrope_section: tuple[int, int, int],
        mrope_interleaved: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rotary_emb = MRotaryEmbedding(
            self.head_size,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=self.is_neox,
            dtype=self.dtype,
            mrope_section=list(mrope_section),
            mrope_interleaved=mrope_interleaved,
        )
        self.enable_rope_custom_op = self.rotary_emb.enabled()

    def forward(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, kv_cache_dummy_dep = super().forward(qkv, positions)
        output = torch.empty(
            q.shape[0],
            self.num_heads * self.head_size,
            device=q.device,
            dtype=self.dtype,
        )
        torch.ops.vllm.unified_attention_with_output(
            q,
            k,
            v,
            output,
            _encode_layer_name(self.layer_name),
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
        # Production decoder attention consumes K from the cache rather than
        # its direct K argument. Returning `output` keeps that production user
        # in the graph without exposing K to an unsupported consumer.
        return q, output, v, kv_cache_dummy_dep

    def ops_in_model_before(self) -> list[torch._ops.OpOverload]:
        return [MROPE_OP, torch.ops.vllm.unified_kv_cache_update.default]

    def ops_in_model_after(self) -> list[torch._ops.OpOverload]:
        return [torch.ops.vllm.fused_qk_norm_mrope_and_unified_kv_cache_update.default]


class QKNormMRoPEObservableKTestModel(QKNormMRoPEKVCacheTestModel):
    def forward(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return QKNormRoPEKVCacheTestModel.forward(self, qkv, positions)


def _run_qk_norm_rope_kvcache_fusion_test(
    *,
    attn_backend: AttentionBackendEnum,
    enable_aiter_triton_rope: bool,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: int,
    block_size: int,
    is_neox: bool,
    use_shuffle_kv_layout: str,
    kv_layout: KVCacheLayout,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    rms_norm_eps: float,
    custom_op: str,
    monkeypatch: pytest.MonkeyPatch,
    mrope_section: tuple[int, int, int] | None = None,
    mrope_interleaved: bool = False,
    observe_mrope_k: bool = False,
    expect_fusion: bool = True,
    force_mrope_support: bool = False,
) -> None:
    device = os.environ.get("VLLM_TEST_CUDA_DEVICE", "cuda")
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        cache_config=CacheConfig(
            block_size=block_size,
            cache_dtype=kv_cache_dtype,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=[custom_op],
            pass_config=PassConfig(
                fuse_qk_norm_rope_kvcache=True,
                eliminate_noops=True,
            ),
        ),
    )

    with vllm.config.set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv(
            "VLLM_ROCM_USE_AITER_TRITON_ROPE",
            "1" if enable_aiter_triton_rope else "0",
        )
        m.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", use_shuffle_kv_layout)
        rocm_aiter_ops.refresh_env_variables()
        if force_mrope_support:
            m.setattr(
                rocm_aiter_backend,
                "is_aiter_mrope_strided_kv_cache_supported",
                lambda: True,
            )

        model_kwargs = {
            "vllm_config": vllm_config,
            "attn_backend": attn_backend,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "rotary_dim": rotary_dim,
            "is_neox": is_neox,
            "rms_norm_eps": rms_norm_eps,
            "dtype": dtype,
            "device": torch.get_default_device(),
        }
        if mrope_section is None:
            model = QKNormRoPEKVCacheTestModel(**model_kwargs)
        else:
            alternate_section = (8, 8, 8)
            m.setattr(
                fusion_module,
                "_discover_mrope_configs",
                lambda _: (
                    (mrope_section, mrope_interleaved),
                    (alternate_section, not mrope_interleaved),
                ),
            )
            model_cls = (
                QKNormMRoPEObservableKTestModel
                if observe_mrope_k
                else QKNormMRoPEKVCacheTestModel
            )
            model = model_cls(
                **model_kwargs,
                mrope_section=mrope_section,
                mrope_interleaved=mrope_interleaved,
            )

        fusion_pass = QkNormRopeKvCacheFusionPass(vllm_config)
        passes = [
            NoOpEliminationPass(vllm_config),
            SplitCoalescingPass(vllm_config),
            ScatterSplitReplacementPass(vllm_config),
            fusion_pass,
            PostCleanupPass(vllm_config),
        ]
        backend = TestBackend(*passes)

        qkv = torch.randn(
            num_tokens,
            num_heads * head_size + 2 * num_kv_heads * head_size,
            dtype=dtype,
        )
        if mrope_section is None:
            pos = torch.arange(num_tokens, dtype=torch.long)
        else:
            # Real Qwen-VL position IDs can be a strided [3, N] view.
            pos = torch.randint(0, 4096, (3, 2 * num_tokens), dtype=torch.long)[:, ::2]
            assert not pos.is_contiguous()

        qkv_unfused = qkv.clone()
        pos_unfused = pos.clone()

        # Run unfused (eager) forward
        with set_forward_context(None, vllm_config):
            forward_context = get_forward_context()
            attn_metadata = model.build_attn_metadata(num_tokens, kv_layout)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_unfused, k_unfused, v_unfused, dummy = model(qkv_unfused, pos_unfused)
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_unfused = attn_layer.kv_cache
        del dummy

        # Run fused (compiled) forward
        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(pos, 0 if mrope_section is None else 1)
        with set_forward_context(None, vllm_config):
            model_fused = torch.compile(model, backend=backend)
            forward_context = get_forward_context()
            attn_metadata = model_fused.build_attn_metadata(num_tokens, kv_layout)
            forward_context.slot_mapping = {
                model.layer_name: attn_metadata.slot_mapping
            }
            q_fused, k_fused, v_fused, dummy = model_fused(qkv, pos)
            attn_layer = forward_context.no_compile_layers[model.layer_name]
            kv_cache_fused = attn_layer.kv_cache
        del dummy

        assert fusion_pass.matched_count == int(expect_fusion)

        if expect_fusion:
            backend.check_before_ops(model.ops_in_model_before())
            backend.check_after_ops(model.ops_in_model_after())
        else:
            for op in model.ops_in_model_before():
                assert backend.op_count(op, before=True) > 0
                assert backend.op_count(op) > 0

        # Sweep-backed (18.2k pts, PR #42749): native-rope ref worst 7.7e-3 -> 1e-2;
        # AITER-triton-rope ref is itself approximate (plateau 1.28e-2) -> 2e-2.
        ATOL, RTOL = (
            (2e-2, 2e-2)
            if enable_aiter_triton_rope or mrope_section is not None
            else (1e-2, 1e-2)
        )
        is_fp8_cache = model.kv_cache_dtype != dtype

        if q_fused.dtype == FP8_DTYPE:
            # Quant-query path: both q are fp8; compare dequant within 1 fp8 ULP.
            torch.testing.assert_close(
                q_unfused.float(), q_fused.float(), atol=1.25e-1, rtol=1.25e-1
            )
        else:
            torch.testing.assert_close(q_unfused, q_fused, atol=ATOL, rtol=RTOL)

        if not expect_fusion or (not is_fp8_cache and mrope_section is None):
            # The AITER PTS kernel populates k_out only for non-FP8 caches.
            # With FP8, the kernel writes quantized K directly to the cache
            # and may leave k_out uninitialised.  In production this is fine
            # because downstream attention reads K from the cache.
            torch.testing.assert_close(k_unfused, k_fused, atol=ATOL, rtol=RTOL)

        # Should be bit exact since no processing had been done on v for both paths
        torch.testing.assert_close(v_unfused, v_fused, atol=0.0, rtol=0.0)

        # Fused and unfused arithmetic can straddle an FP8 rounding boundary.
        # Allow one E4M3 quantization step while still comparing every block.
        if is_fp8_cache:
            cache_atol = cache_rtol = 1.25e-1
        else:
            cache_atol, cache_rtol = ATOL, RTOL

        torch.testing.assert_close(
            kv_cache_unfused.float(),
            kv_cache_fused.float(),
            atol=cache_atol,
            rtol=cache_rtol,
        )


_FUSION_CONFIGS = [
    # Full rotary, both neox styles (the original coverage).
    pytest.param(64, 8, 64, 64, True, id="full-neox"),
    pytest.param(64, 8, 64, 64, False, id="full-non_neox"),
    # GLM-4.5/4.6/4.7 (glm4_moe.py:275 partial_rotary_factor=0.5, neox-style)
    pytest.param(32, 8, 128, 64, True, id="glm4_moe"),
    # GLM-4 dense (glm4.py:97,124 partial_rotary_factor=0.5, non-neox)
    pytest.param(32, 2, 128, 64, False, id="glm4_dense"),
    # Moondream3-style small head (head_size=64, rotary_dim=32)
    pytest.param(16, 2, 64, 32, True, id="partial_small_head"),
]


@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size, rotary_dim, is_neox",
    _FUSION_CONFIGS,
)
@pytest.mark.parametrize(
    "attn_backend",
    [
        AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
    ],
)
@pytest.mark.parametrize("num_tokens", [5, 2048])
@pytest.mark.parametrize(
    "kv_layout",
    [
        pytest.param(KVCacheLayout.LBHNC, id="head_major"),
        pytest.param(KVCacheLayout.LBNHC, id="token_major"),
    ],
)
@pytest.mark.parametrize("enable_aiter_triton_rope", [True, False])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("rms_norm_eps", [1e-6])
@pytest.mark.parametrize("custom_op", ["+rotary_embedding", "+rms_norm"])
# The fused_qk_norm_rope_cache kernel used by this test aborts on AITER < 0.1.20
# (fused_qk_norm_rope_cache_quant.cu: "k_cache/v_cache must be contiguous within
# a block")
@pytest.mark.skipif(
    not Version(version("amd_aiter")) >= Version("0.1.20"),
    reason=(
        "Requires AITER >= 0.1.20; older kernels abort on "
        "fused_qk_norm_rope_cache (k_cache/v_cache contiguity)"
    ),
)
def test_qk_norm_rope_kvcache_fusion(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: int,
    is_neox: bool,
    attn_backend: AttentionBackendEnum,
    enable_aiter_triton_rope: bool,
    kv_layout: KVCacheLayout,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    rms_norm_eps: float,
    custom_op: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _run_qk_norm_rope_kvcache_fusion_test(
        attn_backend=attn_backend,
        enable_aiter_triton_rope=enable_aiter_triton_rope,
        num_tokens=num_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rotary_dim=rotary_dim,
        block_size=block_size,
        is_neox=is_neox,
        use_shuffle_kv_layout="0",
        kv_layout=kv_layout,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        rms_norm_eps=rms_norm_eps,
        custom_op=custom_op,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize(
    "mrope_section",
    [
        pytest.param((16, 24, 24), id="full_rotary"),
        pytest.param((4, 6, 6), id="partial_rotary"),
    ],
)
@pytest.mark.parametrize("mrope_interleaved", [False, True])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize(
    "kv_layout",
    [
        pytest.param(KVCacheLayout.LBHNC, id="head_major"),
        pytest.param(KVCacheLayout.LBNHC, id="token_major"),
        pytest.param(KVCacheLayout.LHBNC, id="block_major"),
    ],
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.skipif(
    not is_aiter_mrope_strided_kv_cache_supported(),
    reason=("Requires AITER v0.1.20.dev1+ or v0.1.21.dev0+ containing ROCm/aiter#4531"),
)
def test_qk_norm_mrope_kvcache_fusion(
    mrope_section: tuple[int, int, int],
    mrope_interleaved: bool,
    is_neox: bool,
    kv_layout: KVCacheLayout,
    kv_cache_dtype: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _run_qk_norm_rope_kvcache_fusion_test(
        attn_backend=AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
        enable_aiter_triton_rope=False,
        num_tokens=5,
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        rotary_dim=2 * sum(mrope_section),
        block_size=16,
        is_neox=is_neox,
        use_shuffle_kv_layout="0",
        kv_layout=kv_layout,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        rms_norm_eps=1e-6,
        custom_op="+rotary_embedding",
        monkeypatch=monkeypatch,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )


def test_mrope_fusion_rejects_observable_k(monkeypatch: pytest.MonkeyPatch):
    _run_qk_norm_rope_kvcache_fusion_test(
        attn_backend=AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
        enable_aiter_triton_rope=False,
        num_tokens=5,
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        rotary_dim=128,
        block_size=16,
        is_neox=True,
        use_shuffle_kv_layout="0",
        kv_layout=KVCacheLayout.LBHNC,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        rms_norm_eps=1e-6,
        custom_op="+rotary_embedding",
        monkeypatch=monkeypatch,
        mrope_section=(16, 24, 24),
        observe_mrope_k=True,
        expect_fusion=False,
        force_mrope_support=True,
    )


def test_mrope_fusion_rejects_unaligned_rotary_dim(
    monkeypatch: pytest.MonkeyPatch,
):
    _run_qk_norm_rope_kvcache_fusion_test(
        attn_backend=AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN,
        enable_aiter_triton_rope=False,
        num_tokens=5,
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        rotary_dim=6,
        block_size=16,
        is_neox=True,
        use_shuffle_kv_layout="0",
        kv_layout=KVCacheLayout.LBHNC,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        rms_norm_eps=1e-6,
        custom_op="+rotary_embedding",
        monkeypatch=monkeypatch,
        mrope_section=(1, 1, 1),
        expect_fusion=False,
        force_mrope_support=True,
    )


@pytest.mark.parametrize(
    ("aiter_version", "expected"),
    [
        ("0.1.19", False),
        ("0.1.20.dev0", False),
        ("0.1.20.dev1", True),
        ("0.1.20.dev2", True),
        ("0.1.20", False),
        ("0.1.20.post1", False),
        ("0.1.21.dev0", True),
        ("0.1.21", True),
    ],
)
def test_mrope_strided_cache_version_gate(
    monkeypatch: pytest.MonkeyPatch,
    aiter_version: str,
    expected: bool,
):
    monkeypatch.setattr(aiter_ops_module, "IS_AITER_FOUND", True)
    monkeypatch.setattr(aiter_ops_module, "version", lambda _: aiter_version)
    assert is_aiter_mrope_strided_kv_cache_supported() is expected


def _call_fused_mrope_impl(q_out: torch.Tensor) -> None:
    device = q_out.device
    fusion_module.fused_qk_norm_mrope_and_unified_kv_cache_update_impl(
        q_out=q_out,
        qkv=torch.zeros(2, 16, device=device),
        positions=torch.arange(2, device=device),
        q_weight=torch.ones(4, device=device),
        k_weight=torch.ones(4, device=device),
        rms_norm_eps=1e-6,
        cos_sin_cache=torch.zeros(16, 4, device=device),
        is_neox=True,
        mrope_section_t=1,
        mrope_section_h=1,
        mrope_section_w=0,
        is_interleaved=False,
        rotary_dim=4,
        layer_name="model.layers.0.self_attn.attn",
    )


def test_mrope_profiling_pass_initializes_query(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        fusion_module,
        "get_attention_context",
        lambda _: (None, None, None, None),
    )
    q_out = torch.full((2, 8), torch.nan, device="cpu")

    _call_fused_mrope_impl(q_out)

    torch.testing.assert_close(q_out, torch.zeros_like(q_out))


def test_mrope_attention_context_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_context(_):
        raise RuntimeError("missing attention context")

    monkeypatch.setattr(fusion_module, "get_attention_context", fail_context)

    with pytest.raises(RuntimeError, match="missing attention context"):
        _call_fused_mrope_impl(torch.empty(2, 8, device="cpu"))
