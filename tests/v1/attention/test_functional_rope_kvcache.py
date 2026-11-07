# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoPE + KV-cache fusion owned by `Attention` on CUDA FlashAttention."""

import pytest
import torch
from transformers import Qwen3Config

from tests.models.utils import check_logprobs_close
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    dense_kv_cache_views,
)
from vllm.compilation.backends import VllmBackend
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ModelConfig,
    PassConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding import (
    Phi3LongRoPEScaledRotaryEmbedding,
    RotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheLayout

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA FlashAttention only"
)

LAYER_NAME = "model.layers.0.self_attn.attn"
FUSED_KERNEL = "_C_cache_ops::fused_rope_and_reshape_cache_flash_q_out"


@pytest.mark.parametrize(
    ("backend", "cache_dtype", "fused"),
    [
        (AttentionBackendEnum.FLASH_ATTN, "auto", True),
        (AttentionBackendEnum.FLASH_ATTN, "bfloat16", False),
        (AttentionBackendEnum.TRITON_ATTN, "auto", False),
    ],
)
def test_fusion_needs_fused_writer_and_matching_cache_dtype(
    backend: AttentionBackendEnum,
    cache_dtype: str,
    fused: bool,
    default_vllm_config: VllmConfig,
) -> None:
    default_vllm_config.compilation_config.pass_config.fuse_rope_kvcache = True
    with set_default_torch_dtype(torch.float16):
        layer = Attention(
            num_heads=4,
            head_size=64,
            scale=0.125,
            num_kv_heads=2,
            cache_config=CacheConfig(cache_dtype=cache_dtype),
            prefix=LAYER_NAME,
            attn_backend=backend.get_class(),
        )
    assert layer._fuse_rope_kvcache is fused


class _RoPEAttention(torch.nn.Module):
    """Llama's attention call site: one qkv projection, RoPE and attention."""

    num_heads, num_kv_heads, head_size = 4, 2, 64

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__()
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        self.sizes = [q_size, kv_size, kv_size]
        self.qkv_proj = torch.nn.Linear(sum(self.sizes), sum(self.sizes))
        self.rotary_emb: torch.nn.Module = RotaryEmbedding(
            self.head_size, self.head_size, 128, 10000, True, torch.float16
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.head_size**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=LAYER_NAME,
            attn_backend=AttentionBackendEnum.FLASH_ATTN.get_class(),
        )

    def unfused(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(qkv).split(self.sizes, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return self.attn(q, k, v)

    def forward(self, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(qkv).split(self.sizes, dim=-1)
        return self.attn(q, k, v, positions=positions, rotary_emb=self.rotary_emb)


def _vllm_config(
    tmp_path,
    num_tokens: int,
    cudagraph_mode: CUDAGraphMode,
    threshold: int = 256,
    **pass_config,
) -> VllmConfig:
    model_dir = tmp_path / "model"
    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(model_dir)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(model_dir), tokenizer=str(model_dir), dtype=torch.float16
        ),
        cache_config=CacheConfig(block_size=16),
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=num_tokens, max_num_seqs=num_tokens
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=cudagraph_mode,
            inductor_compile_config={"force_disable_caches": True},
            pass_config=PassConfig(
                fuse_attn_quant=False,
                rope_kvcache_fusion_max_token_num=threshold,
                **pass_config,
            ),
        ),
    )
    config.cache_config.kv_cache_layout = KVCacheLayout.LBNHC.name
    return config


def _run(model, call, vllm_config, qkv, positions):
    """Run `call` on fresh KV-cache pages and return the output, the cache
    contents and the native ops that ran."""
    num_tokens = qkv.shape[0]
    spec = model.attn.get_kv_cache_spec(vllm_config)
    num_blocks = -(-num_tokens // spec.block_size)
    metadata = (
        model.attn.get_attn_backend()
        .get_builder_cls()(spec, [LAYER_NAME], vllm_config, qkv.device)
        .build(
            0,
            create_common_attn_metadata(
                BatchSpec([num_tokens], [num_tokens]),
                block_size=spec.block_size,
                device=qkv.device,
                arange_block_indices=True,
            ),
        )
    )
    storage = torch.zeros(
        num_blocks * spec.page_size_bytes, dtype=torch.int8, device=qkv.device
    )
    model.attn.kv_cache = dense_kv_cache_views(
        storage, spec, num_blocks=num_blocks, num_layers=1, layout=KVCacheLayout.LBNHC
    )[0]
    with (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof,
        set_forward_context(
            metadata, vllm_config, slot_mapping={LAYER_NAME: metadata.slot_mapping}
        ),
    ):
        output = call(qkv, positions)
    return output, model.attn.kv_cache.clone(), {e.name for e in prof.events()}


@pytest.mark.parametrize("path", ["fused", "disabled", "longrope"])
def test_compiled_attention_rope_matches_unfused(
    path: str, disable_vllm_compile_cache, tmp_path
) -> None:
    """The fused op writes Q and the cache before attention, which then gets
    no K/V; disabled fusion and RoPE without a rotation keep the old graph."""
    vllm_config = _vllm_config(
        tmp_path, 2, CUDAGraphMode.PIECEWISE, fuse_rope_kvcache=path != "disabled"
    )
    assert "vllm::unified_attention_with_output" in (
        vllm_config.compilation_config.splitting_ops or []
    )
    with (
        torch.device("cuda"),
        set_default_torch_dtype(torch.float16),
        set_current_vllm_config(vllm_config),
    ):
        torch.manual_seed(0)
        model = _RoPEAttention(vllm_config)
        if path == "longrope":
            model.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                64, 64, 256, 128, 10000, True, torch.float16, [1.0] * 32, [2.0] * 32
            )
        qkv = torch.randn(2, sum(model.sizes))
        positions = torch.arange(2)
        expected, expected_cache, _ = _run(
            model, model.unfused, vllm_config, qkv, positions
        )
        torch._dynamo.mark_dynamic(qkv, 0)
        torch._dynamo.mark_dynamic(positions, 0)
        backend = VllmBackend(vllm_config)
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        output, cache, ops = _run(model, compiled, vllm_config, qkv, positions)

    nodes = [n for n in backend.graph.graph.nodes if n.op == "call_function"]
    fused = [
        n
        for n in nodes
        if n.target is torch.ops.vllm.fused_rope_and_unified_kv_cache_update_q_out
    ]
    (attention,) = [
        n for n in nodes if n.target is torch.ops.vllm.unified_attention_with_output
    ]
    if path == "fused":
        (fused_node,) = fused
        assert all(a.meta["example_value"].ndim == 3 for a in fused_node.args[:4])
        assert attention.args[4] is fused_node.args[-1]
        assert attention.args[1:3] == (None, None)
    else:
        assert fused == []
        assert all(arg is not None for arg in attention.args[1:3])
    assert (FUSED_KERNEL in ops) is (path == "fused")
    torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(cache, expected_cache, atol=2e-3, rtol=2e-3)
    assert cache.count_nonzero() > 0


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("token_counts", [(3, 2), (2, 3)])
def test_token_threshold_selects_fused_kernel_at_runtime(
    compiled: bool, token_counts: tuple[int, int], disable_vllm_compile_cache, tmp_path
) -> None:
    """VLLM drops Dynamo shape guards, so one compiled graph serves both sides
    of the threshold and the op itself must choose the kernel."""
    threshold = 2
    vllm_config = _vllm_config(
        tmp_path,
        max(token_counts),
        CUDAGraphMode.NONE,
        threshold,
        fuse_rope_kvcache=True,
    )
    with (
        torch.device("cuda"),
        set_default_torch_dtype(torch.float16),
        set_current_vllm_config(vllm_config),
    ):
        torch.manual_seed(0)
        model = _RoPEAttention(vllm_config)
        call = (
            torch.compile(model, backend=VllmBackend(vllm_config), fullgraph=True)
            if compiled
            else model
        )
        for index, num_tokens in enumerate(token_counts):
            qkv = torch.randn(num_tokens, sum(model.sizes))
            positions = torch.arange(num_tokens)
            expected, expected_cache, _ = _run(
                model, model.unfused, vllm_config, qkv, positions
            )
            if index == 0:
                torch._dynamo.mark_dynamic(qkv, 0)
                torch._dynamo.mark_dynamic(positions, 0)
            output, cache, ops = _run(model, call, vllm_config, qkv, positions)

            assert (FUSED_KERNEL in ops) is (num_tokens <= threshold)
            torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(cache, expected_cache, atol=2e-3, rtol=2e-3)


def _all_layers_fused(model: torch.nn.Module) -> bool:
    return all(layer.self_attn.attn._fuse_rope_kvcache for layer in model.model.layers)


@pytest.mark.parametrize("breakable_cudagraph", [False, True])
def test_llama_fused_rope_matches_unfused(
    vllm_runner,
    example_prompts,
    monkeypatch: pytest.MonkeyPatch,
    enable_pickle,
    breakable_cudagraph: bool,
) -> None:
    """Captured graphs replay the fused op with new positions and slots."""
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", str(int(breakable_cudagraph)))
    outputs = []
    for fused in (False, True):
        with vllm_runner(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dtype="bfloat16",
            max_model_len=512,
            gpu_memory_utilization=0.5,
            attention_config={"backend": "FLASH_ATTN"},
            compilation_config={"pass_config": {"fuse_rope_kvcache": fused}},
        ) as llm:
            assert llm.apply_model(_all_layers_fused) == [fused]
            outputs.append(
                llm.generate_greedy_logprobs(
                    example_prompts, max_tokens=32, num_logprobs=5
                )
            )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="unfused",
        name_1="fused",
    )
