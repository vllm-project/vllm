# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    required_sm120_sparse_topk,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert required_sm120_sparse_topk(causal, 128) == 128
    assert required_sm120_sparse_topk(dspark, 128) == 192


@torch.inference_mode()
def _check_sm120_sparse_dcp(
    config, rope_dim: int, block_size: int, interleave: int, tokens: int = 4
):
    """Real backend + packed KV + collective; compare to unsharded BF16 output."""
    from dataclasses import replace

    from vllm.config import ParallelConfig
    from vllm.distributed import get_dcp_group, get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseMetadata,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        FlashInferMLASparseSM120Impl,
    )
    from vllm.v1.attention.ops.dcp import dcp_a2a_lse_reduce

    group = get_dcp_group()
    rank, world = group.rank_in_group, group.world_size
    heads, length, topk = 64, 4096, 2048
    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.manual_seed(17)
    rows = torch.zeros(length, 528 + 2 * rope_dim, dtype=torch.uint8, device=device)
    rows[:, :512] = (
        (torch.randn(length, 512, device=device) * 16)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    glm = config.model_config.hf_text_config.model_type.startswith("glm")
    scales = torch.full((length, 4), 0.035 if glm else 0.03125, device=device)
    rows[:, 512:528] = scales.view(torch.uint8)
    rows[:, 528:] = (
        (torch.randn(length, rope_dim, device=device) * 0.25)
        .to(torch.bfloat16)
        .view(torch.uint8)
    )
    q = (torch.randn(tokens, heads, 512 + rope_dim, device=device) * 0.25).to(
        torch.bfloat16
    )
    # Offset into a shared index buffer, with unequal local counts and a row
    # that is empty on all ranks. Row 0 belongs only to rank 0.
    logical = torch.full((tokens + 1, topk), -1, dtype=torch.int32, device=device)
    logical[1, 0] = 0
    ids = torch.arange(length, device=device)
    if world > 1:
        # A 1:7 split exposes an incorrect logarithm base in the LSE merge.
        logical[2, :8] = torch.cat(
            (
                ids[(ids // interleave) % world == 0][:1],
                ids[(ids // interleave) % world == 1][:7],
            )
        ).int()
    else:
        logical[2, :8] = ids[:8].int()
    logical[3:tokens] = torch.randperm(length, device=device)[:topk].int()
    # Exact 1:7 attention: one value is 1, the other seven are 0, and
    # zero queries make every score equal. The result is exactly 1/8;
    # shard-local FP8 probability rounding cannot obscure the LSE-base check.
    exact_ids = logical[2, :8].long()
    rows[exact_ids, :512] = 0
    rows[exact_ids[0], :512] = torch.ones(
        512, dtype=torch.float8_e4m3fn, device=device
    ).view(torch.uint8)
    rows[exact_ids, 512:528] = torch.ones(8, 4, device=device).view(torch.uint8)
    q[1].zero_()

    def make_impl(current_config, num_heads, indices):
        with set_current_vllm_config(current_config):
            proj = ColumnParallelLinear(
                512,
                num_heads * get_tensor_model_parallel_world_size() * 256,
                bias=False,
                params_dtype=torch.bfloat16,
            )
            return FlashInferMLASparseSM120Impl(
                num_heads=num_heads,
                head_size=512 + rope_dim,
                scale=128**-0.5,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="fp8_ds_mla",
                logits_soft_cap=None,
                attn_type="decoder",
                kv_sharing_target_layer_name=None,
                q_lora_rank=None,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=rope_dim,
                qk_head_dim=128 + rope_dim,
                v_head_dim=128,
                kv_b_proj=proj,
                topk_indices_buffer=indices,
            )

    impl = make_impl(config, heads // world, logical)
    reference_config = replace(
        config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=world,
            decode_context_parallel_size=1,
            disable_custom_all_reduce=True,
            distributed_executor_backend="external_launcher",
        ),
    )
    reference = make_impl(reference_config, heads, logical[1:])
    owned = ids[(ids // interleave) % world == rank]
    nblocks = owned.numel() // block_size
    # Two layers interleaved at block granularity (BLHNC layout).
    storage = torch.full(
        (nblocks, 2, block_size, rows.shape[-1]), 255, dtype=torch.uint8, device=device
    )
    cache = storage[:, 0]
    blocks = torch.randperm(nblocks, device=device).int()
    local = torch.arange(owned.numel(), device=device)
    cache[blocks[local // block_size].long(), local % block_size] = rows[owned]
    metadata = FlashInferMLASparseMetadata(
        num_reqs=1,
        max_query_len=tokens,
        max_seq_len=length,
        num_actual_tokens=tokens,
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        slot_mapping=torch.full((tokens,), -1, dtype=torch.int64, device=device),
        num_decodes=int(tokens == 4),
        num_decode_tokens=tokens if tokens == 4 else 0,
        num_prefills=int(tokens != 4),
        block_table=blocks[None],
        req_id_per_token=torch.zeros(tokens, dtype=torch.int32, device=device),
        block_size=block_size,
        topk_tokens=topk,
        topk_start=1,
        cp_kv_cache_interleave_size=interleave,
    )
    ref_meta = replace(
        metadata,
        topk_start=0,
        block_size=64,
        block_table=torch.arange(length // 64, dtype=torch.int32, device=device)[None],
    )
    full_cache = rows.view(-1, 64, rows.shape[-1])

    def run():
        return impl.forward_mqa((q[..., :512], q[..., 512:]), cache, metadata, None)

    def check(output, lse):
        expected, _ = reference.forward_mqa(q, full_cache, ref_meta, None)
        if world > 1:
            assert lse is not None and lse.shape == (tokens, heads)
            # Empty rows must have an exact neutral value before communication.
            assert torch.count_nonzero(output[-1]) == 0
            assert torch.isneginf(lse[-1]).all()
            if rank:
                assert torch.count_nonzero(output[0]) == 0
                assert torch.isneginf(lse[0]).all()
            output = dcp_a2a_lse_reduce(
                output, lse, group, is_lse_base_on_e=impl.lse_base_on_e
            )
            expected = expected[
                :, rank * (heads // world) : (rank + 1) * (heads // world)
            ]
        else:
            assert lse is None
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)

    with set_current_vllm_config(config):
        output, lse = run()
        check(output, lse)
        saved_output, saved_lse = output.clone(), None if lse is None else lse.clone()
        # Another launch must not overwrite results retained for the DCP merge.
        q.mul_(0.5)
        run()
        torch.testing.assert_close(output, saved_output, atol=0, rtol=0)
        if lse is not None:
            torch.testing.assert_close(lse, saved_lse, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_lse = run()
        q.mul_(0.5)
        logical[3:tokens].add_(interleave).remainder_(length)
        graph.replay()
        eager, eager_lse = run()
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)
        if captured_lse is not None:
            torch.testing.assert_close(captured_lse, eager_lse, atol=0, rtol=0)
        check(captured, captured_lse)


def _sm120_sparse_config(path, model_kind, world_size):
    from transformers import DeepseekV32Config, Glm5NextConfig, GlmMoeDsaConfig

    from vllm.config import KernelConfig, ModelConfig, ParallelConfig, VllmConfig

    config_cls, architecture, rope_dim = {
        "deepseek": (DeepseekV32Config, "DeepseekV32ForCausalLM", 64),
        "glm": (GlmMoeDsaConfig, "GlmMoeDsaForCausalLM", 64),
        "glm_nope": (Glm5NextConfig, "Glm5NextForCausalLM", 0),
    }[model_kind]
    config_cls(architectures=[architecture], qk_rope_head_dim=rope_dim).save_pretrained(
        path
    )
    return VllmConfig(
        model_config=ModelConfig(
            model=str(path),
            dtype="bfloat16",
            max_model_len=4096,
            skip_tokenizer_init=True,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            decode_context_parallel_size=world_size,
            disable_custom_all_reduce=True,
            distributed_executor_backend="external_launcher",
        ),
        kernel_config=KernelConfig(enable_jit_warmup=False),
    ), rope_dim


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120/SM121 kernel"
)
@pytest.mark.parametrize("model_kind", ["deepseek", "glm", "glm_nope"])
@pytest.mark.parametrize("block_size,interleave", [(64, 1), (256, 64)])
@pytest.mark.parametrize("tokens", [4, 128])
def test_sm120_sparse_packed_cache_and_graph(
    tmp_path, dist_init, workspace_init, model_kind, block_size, interleave, tokens
):
    config, rope_dim = _sm120_sparse_config(tmp_path, model_kind, 1)
    _check_sm120_sparse_dcp(config, rope_dim, block_size, interleave, tokens)


def _sm120_dcp_worker(rank, world_size, init_method, local_rank=None):
    from tempfile import TemporaryDirectory

    from vllm.config import ParallelConfig, VllmConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    local_rank = rank if local_rank is None else local_rank
    torch.accelerator.set_device_index(local_rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            decode_context_parallel_size=world_size,
            disable_custom_all_reduce=True,
            distributed_executor_backend="external_launcher",
        )
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            decode_context_model_parallel_size=world_size,
        )
        init_workspace_manager(torch.device("cuda", local_rank))
    try:
        for kind in ("deepseek", "glm", "glm_nope"):
            with TemporaryDirectory() as path:
                config, rope_dim = _sm120_sparse_config(path, kind, world_size)
                for block_size, interleave in ((64, 1), (256, 64)):
                    for tokens in (4, 128):
                        _check_sm120_sparse_dcp(
                            config, rope_dim, block_size, interleave, tokens
                        )
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120)
    or torch.accelerator.device_count() < 2,
    reason="Two SM120/SM121 GPUs required for the real DCP collective",
)
def test_sm120_sparse_dcp(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_sm120_dcp_worker, args=(2, f"file://{tmp_path / 'dist'}"), nprocs=2)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120)
    or torch.accelerator.device_count() < 2,
    reason="Two SM120/SM121 GPUs required for DCP generation",
)
def test_sm120_sparse_dcp_generation(tmp_path, vllm_runner):
    """Exercise model dispatch, chunked prefill and decode through LLM.generate."""
    from vllm import SamplingParams

    _sm120_sparse_config(tmp_path, "deepseek", 2)
    overrides = dict(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=64,
        num_key_value_heads=64,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        vocab_size=256,
        max_position_embeddings=1024,
        first_k_dense_replace=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=128,
        num_nextn_predict_layers=0,
    )
    prompts = [
        {"prompt_token_ids": [2 + (j * 17 + i * 13) % 250 for j in range(n)]}
        for i, n in enumerate((33, 257))
    ]
    sampling = SamplingParams(
        temperature=0, max_tokens=8, ignore_eos=True, logprobs=256
    )
    results = []
    for dcp in (1, 2):
        with vllm_runner(
            str(tmp_path),
            hf_overrides=overrides,
            load_format="dummy",
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tensor_parallel_size=2,
            decode_context_parallel_size=dcp,
            distributed_executor_backend="mp",
            disable_custom_all_reduce=True,
            attention_config={"backend": "FLASHINFER_MLA_SPARSE_SM120"},
            kernel_config={
                "enable_flashinfer_autotune": False,
                "enable_jit_warmup": False,
            },
            kv_cache_dtype="fp8",
            block_size=64,
            max_model_len=512,
            max_num_seqs=2,
            max_num_batched_tokens=64,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            seed=42,
            kv_cache_memory_bytes=256 * 1024 * 1024,
            gpu_memory_utilization=0.1,
            max_logprobs=256,
        ) as runner:
            outputs = runner.llm.generate(prompts, sampling, use_tqdm=False)
            assert all(len(r.outputs[0].token_ids) == 8 for r in outputs)
            results.append(
                [
                    (list(r.outputs[0].token_ids), "", r.outputs[0].logprobs)
                    for r in outputs
                ]
            )
    for reference, actual in zip(*results):
        assert reference[0] == actual[0]
        # Compare every vocabulary entry; near-ties at a top-k cutoff should
        # not turn small BF16 numerical differences into a set mismatch.
        for ref_step, step in zip(reference[2], actual[2]):
            assert len(step) == len(ref_step) == 256
            torch.testing.assert_close(
                torch.tensor([step[t].logprob for t in range(256)]),
                torch.tensor([ref_step[t].logprob for t in range(256)]),
                atol=0.01,
                rtol=0,
            )
