# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import Qwen3Config

import vllm.config
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
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
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

_LAYER_NAME = "model.layers.0.self_attn.attn"


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
    assert "vllm::unified_attention_with_output" in (
        vllm_config.compilation_config.splitting_ops or []
    )

    with torch.device(device), vllm.config.set_current_vllm_config(vllm_config):
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
        builder = model.backend.get_builder_cls()(
            model.attn.get_kv_cache_spec(vllm_config),
            [_LAYER_NAME],
            vllm_config,
            device,
        )
        metadata = builder.build(0, common_metadata)
        cache_shape = model.backend.get_kv_cache_shape(
            1, 16, model.num_kv_heads, model.head_size
        )
        order = model.backend.get_kv_cache_stride_order()
        cache = torch.zeros(
            tuple(cache_shape[index] for index in order), dtype=dtype
        ).permute(*(order.index(index) for index in range(len(order))))

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
