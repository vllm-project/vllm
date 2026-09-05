# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

import vllm.config
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    dense_kv_cache_views,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.passes.fusion.rope_kvcache_fusion import (
    RopeKVCacheFusionPass,
)
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ModelConfig,
    PassConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models.llama import LlamaAttention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheLayout

_LAYER_NAME = "model.layers.0.self_attn.attn"


class _IdentityProjection(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states, None


class _IdentityRotary(torch.nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key


class _ManualPathMarker(Attention):
    rope_kvcache_fusion_max_token_num = 256

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward_with_fused_rope_kvcache(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rotary_emb: torch.nn.Module,
    ) -> torch.Tensor:
        return query + 1

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return query - 1


@support_torch_compile(
    dynamic_arg_dims={
        "positions": {0: "num_tokens"},
        "hidden_states": {0: "num_tokens"},
    }
)
class _CompiledLlamaAttentionCallSite(torch.nn.Module):
    forward = LlamaAttention.forward

    def __init__(self, vllm_config: VllmConfig | None = None) -> None:
        super().__init__()
        self.q_size = 1
        self.kv_size = 1
        self.qkv_proj = _IdentityProjection()
        self.rotary_emb = _IdentityRotary()
        self.attn = _ManualPathMarker()
        self.o_proj = _IdentityProjection()
        self._use_fused_rope_kvcache = True


def test_manual_fusion_requires_matching_activation_and_cache_dtype(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config: VllmConfig,
) -> None:
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    with vllm.config.set_current_vllm_config(default_vllm_config):
        rotary_emb = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=128,
            base=10000,
            is_neox_style=True,
            dtype=torch.float16,
        )
    layer = SimpleNamespace(
        _rope_kvcache_fusion_enabled=True,
        _fuse_attn_quant=False,
        attn_type=AttentionType.DECODER,
        attn_backend=SimpleNamespace(forward_includes_kv_cache_update=False),
        kv_sharing_target_layer_name=None,
        head_size=64,
        head_size_v=64,
        dtype=torch.float16,
        kv_cache_torch_dtype=torch.float16,
        query_quant=None,
        impl=SimpleNamespace(fused_rope_kvcache_q_out_supported=lambda: True),
    )

    assert Attention.manual_rope_kvcache_fusion_supported(layer, rotary_emb)
    layer.kv_cache_torch_dtype = torch.bfloat16
    assert not Attention.manual_rope_kvcache_fusion_supported(layer, rotary_emb)


def test_missing_slot_mapping_rotates_query_without_materializing_key(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm import _custom_ops

    calls = []
    monkeypatch.setattr(
        _custom_ops,
        "rotary_embedding",
        lambda _positions, _query, key, head_size, *_args, **_kwargs: (
            calls.append((key, head_size))
        ),
    )

    query = torch.randn(1, 4 * 64)
    key = torch.randn(1, 2 * 64)
    value = torch.randn_like(key)
    layer = SimpleNamespace(
        impl=SimpleNamespace(fused_rope_kvcache_q_out_supported=lambda: True),
        rope_kvcache_fusion_max_token_num=256,
        head_size=64,
    )
    query_out = torch.empty_like(query, memory_format=torch.contiguous_format)
    Attention._rope_and_kv_cache_update_q_out(
        layer,
        query,
        key,
        value,
        query_out,
        torch.empty(1, dtype=torch.int64),
        torch.empty(1, 1),
        True,
        torch.empty(1),
        None,
    )

    torch.testing.assert_close(query_out, query)
    assert query_out.data_ptr() != query.data_ptr()
    assert calls == [(None, 64)]


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [(256, 1.0), (257, -1.0)],
)
def test_llama_manual_rope_eager_call_site_keeps_token_threshold(
    num_tokens: int,
    expected: float,
) -> None:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.NONE)
    )
    model = _CompiledLlamaAttentionCallSite(vllm_config=vllm_config)
    positions = torch.arange(num_tokens)
    hidden_states = torch.zeros(num_tokens, 3)

    output = LlamaAttention.forward(model, positions, hidden_states)

    torch.testing.assert_close(output, torch.full_like(output, expected))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA.")
@pytest.mark.parametrize(
    "token_counts",
    [
        (512, 2),
        (2, 512),
        (257, 256),
        (256, 257),
    ],
)
def test_llama_manual_rope_call_site_is_stable_when_guards_are_dropped(
    token_counts: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
    disable_vllm_compile_cache,
) -> None:
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")
    monkeypatch.setenv("VLLM_USE_BYTECODE_HOOK", "0")
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=512,
            max_num_seqs=1,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            pass_config=PassConfig(
                fuse_rope_kvcache=True,
                rope_kvcache_fusion_max_token_num=256,
            ),
            inductor_compile_config={"force_disable_caches": True},
        ),
    )
    assert vllm_config.compilation_config.compile_ranges_endpoints == [256, 512]

    with (
        torch.no_grad(),
        vllm.config.set_current_vllm_config(vllm_config),
        set_forward_context({}, vllm_config),
    ):
        model = _CompiledLlamaAttentionCallSite(vllm_config=vllm_config).cuda()
        for num_tokens in token_counts:
            positions = torch.arange(num_tokens, device="cuda")
            hidden_states = torch.zeros(num_tokens, 3, device="cuda")
            output = model(positions, hidden_states)
            torch.testing.assert_close(output, torch.ones_like(output))


class _FunctionalRoPEAttention(torch.nn.Module):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_size = 4, 2, 64
        self.qkv_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        self.qkv_proj = torch.nn.Linear(
            self.qkv_size, self.qkv_size, bias=False, dtype=torch.float16
        )
        self.rotary_emb = RotaryEmbedding(
            self.head_size,
            rotary_dim=self.head_size,
            max_position_embeddings=128,
            base=10000,
            is_neox_style=True,
            dtype=torch.float16,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.head_size**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=_LAYER_NAME,
            attn_backend=AttentionBackendEnum.FLASH_ATTN.get_class(),
        )
        self.attn._k_scale = self.attn._k_scale.to(device)
        self.attn._v_scale = self.attn._v_scale.to(device)
        self.backend = self.attn.get_attn_backend()

    def _split_qkv(self, hidden_states: torch.Tensor):
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        qkv = self.qkv_proj(hidden_states)
        return qkv.split([q_size, kv_size, kv_size], dim=-1)

    def incumbent(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        query, key, value = self._split_qkv(qkv)
        query, key = self.rotary_emb(positions, query, key)
        return self.attn(query, key, value)

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        query, key, value = self._split_qkv(qkv)
        return self.attn.forward_with_fused_rope_kvcache(
            positions, query, key, value, self.rotary_emb
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA.")
def test_q_out_rope_kvcache_stays_before_attention_with_graph_owned_output(
    mocker, disable_vllm_compile_cache, tmp_path
):
    from vllm.compilation.backends import VllmBackend

    dtype = torch.float16
    device = torch.device("cuda")
    num_tokens = 2
    model_dir = tmp_path / "model"
    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(model_dir)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(model_dir), tokenizer=str(model_dir), dtype=dtype
        ),
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=num_tokens,
            max_num_seqs=num_tokens,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            use_inductor_graph_partition=False,
            inductor_compile_config={"force_disable_caches": True},
            pass_config=PassConfig(
                fuse_rope_kvcache=True,
                fuse_attn_quant=False,
            ),
        ),
    )
    vllm_config.cache_config.kv_cache_layout = KVCacheLayout.LBNHC.name
    assert "vllm::unified_attention_with_output" in (
        vllm_config.compilation_config.splitting_ops or []
    )

    with (
        torch.device(device),
        set_default_torch_dtype(dtype),
        vllm.config.set_current_vllm_config(vllm_config),
    ):
        torch.manual_seed(0)
        model = _FunctionalRoPEAttention(vllm_config, device)
        assert model.attn.manual_rope_kvcache_fusion_supported(model.rotary_emb)
        qkv = torch.randn(
            num_tokens,
            (model.num_heads + 2 * model.num_kv_heads) * model.head_size,
            dtype=dtype,
        )
        positions = torch.arange(num_tokens, dtype=torch.long)
        common_metadata = create_common_attn_metadata(
            BatchSpec([num_tokens], [num_tokens]),
            block_size=16,
            device=device,
            arange_block_indices=True,
        )
        cache_spec = model.attn.get_kv_cache_spec(vllm_config)
        assert cache_spec is not None
        builder = model.backend.get_builder_cls()(
            cache_spec,
            [_LAYER_NAME],
            vllm_config,
            device,
        )
        metadata = builder.build(0, common_metadata)
        cache_storage = torch.zeros(
            cache_spec.page_size_bytes, dtype=torch.int8, device=device
        )
        cache = dense_kv_cache_views(
            cache_storage,
            cache_spec,
            num_blocks=1,
            num_layers=1,
            layout=KVCacheLayout.LBNHC,
        )[0]

        def run(call):
            model.attn.kv_cache = cache.clone()
            with set_forward_context(
                metadata,
                vllm_config,
                slot_mapping={_LAYER_NAME: metadata.slot_mapping},
            ):
                output = call(qkv, positions)
            return output, model.attn.kv_cache.clone()

        incumbent_output, incumbent_cache = run(model.incumbent)
        fused_update = mocker.spy(model.attn.impl, "do_rope_and_kv_cache_update_q_out")
        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(positions, 0)
        backend = VllmBackend(vllm_config)
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        fused_output, fused_cache = run(compiled)

    # CUDA owns this fusion at the model call site, so the legacy ROCm graph
    # pass must not also be registered.
    fusion_passes = [
        pass_
        for pass_ in backend.pass_manager.passes
        if isinstance(pass_, RopeKVCacheFusionPass)
    ]
    assert fusion_passes == []
    call_nodes = [
        node for node in backend.graph.graph.nodes if node.op == "call_function"
    ]
    fused_nodes = [
        node
        for node in call_nodes
        if node.target is torch.ops.vllm.fused_rope_and_unified_kv_cache_update_q_out
    ]
    attention_nodes = [
        node
        for node in call_nodes
        if node.target is torch.ops.vllm.unified_attention_with_output
    ]
    assert len(fused_nodes) == len(attention_nodes) == 1
    assert fused_nodes[0].args[-1] is attention_nodes[0].args[4]
    assert attention_nodes[0].args[1:3] == (None, None)
    fused_update.assert_called_once()
    torch.testing.assert_close(incumbent_output, fused_output, atol=2e-3, rtol=2e-3)
    assert torch.count_nonzero(fused_cache).item() > 0
    torch.testing.assert_close(incumbent_cache, fused_cache, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test on CUDA.")
@pytest.mark.parametrize("token_counts", [(3, 2), (2, 3)])
def test_compiled_manual_rope_runtime_threshold_tracks_dynamic_tokens(
    token_counts: tuple[int, int],
    mocker,
    disable_vllm_compile_cache,
    tmp_path,
) -> None:
    from vllm.compilation.backends import VllmBackend

    dtype = torch.float16
    device = torch.device("cuda")
    threshold = 2
    block_size = 16
    model_dir = tmp_path / "model"
    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(model_dir)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(model_dir), tokenizer=str(model_dir), dtype=dtype
        ),
        cache_config=CacheConfig(block_size=block_size, cache_dtype="auto"),
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=max(token_counts),
            max_num_seqs=max(token_counts),
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            use_inductor_graph_partition=False,
            inductor_compile_config={"force_disable_caches": True},
            pass_config=PassConfig(
                fuse_rope_kvcache=True,
                fuse_attn_quant=False,
                rope_kvcache_fusion_max_token_num=threshold,
            ),
        ),
    )
    vllm_config.cache_config.kv_cache_layout = KVCacheLayout.LBNHC.name

    with (
        torch.device(device),
        set_default_torch_dtype(dtype),
        vllm.config.set_current_vllm_config(vllm_config),
    ):
        torch.manual_seed(0)
        model = _FunctionalRoPEAttention(vllm_config, device)
        cache_spec = model.attn.get_kv_cache_spec(vllm_config)
        assert cache_spec is not None
        builder = model.backend.get_builder_cls()(
            cache_spec,
            [_LAYER_NAME],
            vllm_config,
            device,
        )
        fused_update = mocker.spy(model.attn.impl, "do_rope_and_kv_cache_update_q_out")
        fallback_update = mocker.spy(model.attn.impl, "do_kv_cache_update")
        backend = VllmBackend(vllm_config)
        compiled = torch.compile(model, backend=backend, fullgraph=True)

        def run(call, qkv: torch.Tensor, positions: torch.Tensor):
            num_tokens = qkv.shape[0]
            common_metadata = create_common_attn_metadata(
                BatchSpec([num_tokens], [num_tokens]),
                block_size=block_size,
                device=device,
                arange_block_indices=True,
            )
            metadata = builder.build(0, common_metadata)
            num_blocks = (num_tokens + block_size - 1) // block_size
            cache_storage = torch.zeros(
                num_blocks * cache_spec.page_size_bytes,
                dtype=torch.int8,
                device=device,
            )
            model.attn.kv_cache = dense_kv_cache_views(
                cache_storage,
                cache_spec,
                num_blocks=num_blocks,
                num_layers=1,
                layout=KVCacheLayout.LBNHC,
            )[0]
            with set_forward_context(
                metadata,
                vllm_config,
                slot_mapping={_LAYER_NAME: metadata.slot_mapping},
            ):
                output = call(qkv, positions)
            return output, model.attn.kv_cache.clone()

        for index, num_tokens in enumerate(token_counts):
            qkv = torch.randn(num_tokens, model.qkv_size, dtype=dtype, device=device)
            positions = torch.arange(num_tokens, dtype=torch.long, device=device)
            expected_output, expected_cache = run(model.incumbent, qkv, positions)
            fused_update.reset_mock()
            fallback_update.reset_mock()
            if index == 0:
                torch._dynamo.mark_dynamic(qkv, 0)
                torch._dynamo.mark_dynamic(positions, 0)
            actual_output, actual_cache = run(compiled, qkv, positions)

            assert fused_update.call_count == int(num_tokens <= threshold)
            assert fallback_update.call_count == int(num_tokens > threshold)
            torch.testing.assert_close(
                actual_output, expected_output, atol=2e-3, rtol=2e-3
            )
            torch.testing.assert_close(
                actual_cache, expected_cache, atol=2e-3, rtol=2e-3
            )
