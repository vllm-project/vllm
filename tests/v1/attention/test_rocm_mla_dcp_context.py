# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.utils import multi_gpu_only
from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import context_row_indices


@pytest.mark.parametrize(
    "padded,lengths,starts,total,expected",
    [
        ([3, 2], [[7, 6], [2, 1]], [5, 0], 6, [0, 1, 5, 3, 4, 8]),
        ([3], [[2, 9]], [4], 3, [3, 4, 5]),
        ([3], [[1, 1]], [0], 3, None),
    ],
    ids=["padding_and_chunk_starts", "empty_rank", "inconsistent_total"],
)
def test_dcp_context_indices(padded, lengths, starts, total, expected):
    chunk = SimpleNamespace(
        padded_local_seq_lens=padded,
        local_context_lens_allranks=lengths,
        local_starts=starts,
        num_local_context_tokens=sum(padded),
        num_context_tokens=total,
    )
    if expected is None:
        with pytest.raises(AssertionError):
            context_row_indices(chunk, torch.device("cpu"))
    else:
        rows = context_row_indices(chunk, torch.device("cpu"))
        assert rows.tolist() == expected
        assert rows.dtype == torch.int32


def test_compressed_gather_preserves_bytes_and_workspace_partition(monkeypatch):
    from vllm.v1.attention.ops import rocm_aiter_mla_prefill as prefill

    workspace = torch.full((12, 4), -1, dtype=torch.bfloat16)
    cache = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
    chunk = SimpleNamespace(
        num_local_context_tokens=2,
        local_context_lens_allranks=[[2, 2]],
        padded_local_cu_seq_lens=torch.tensor([0, 2]),
        num_requests=1,
        starts=torch.tensor([0]),
    )

    def local_gather(src, dst, *args):
        assert src.dtype == dst.dtype == torch.uint8
        dst.copy_(src)

    def allgather(dst, src):
        assert dst.dtype == src.dtype == torch.uint8
        assert dst.shape == (4, 4)
        assert src.shape == (2, 4)
        dst[:2].copy_(src)
        dst[2:].copy_(src + 16)

    monkeypatch.setattr(prefill.ops, "cp_gather_cache", local_gather)
    gathered = prefill.gather_compressed_context(
        cache, workspace, torch.empty(1, 1), chunk, allgather, torch.float8_e4m3fn
    )
    torch.testing.assert_close(
        gathered.view(torch.uint8), torch.cat((cache, cache + 16))
    )
    # Retain the original logical capacity even though the byte view is larger.
    assert gathered.data_ptr() - workspace.data_ptr() == 4 * 4
    torch.testing.assert_close(workspace.view(torch.uint8).reshape(-1, 4)[:2], cache)


@pytest.mark.parametrize(
    "unsupported",
    [
        "packed_cache",
        "e5m2",
        "bf16_cache",
        "quantized_weight",
        "bias",
        "no_indices",
        "query_dtype",
        "prefill_dtype",
        "weight_dtype",
        "weight_layout",
        "cache_layout",
        "dimensions",
        "no_aiter",
    ],
)
def test_dcp_prefill_unsupported_formats_use_original_path(monkeypatch, unsupported):
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        pytest.skip("ROCm MLA backend")
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        AiterMLAImpl,
        MLACommonImpl,
    )

    if unsupported == "no_aiter":
        monkeypatch.setattr(
            "vllm.v1.attention.backends.mla.rocm_aiter_mla."
            "is_aiter_found_and_supported",
            lambda: False,
        )

    impl = object.__new__(AiterMLAImpl)
    impl.kv_cache_dtype = {
        "packed_cache": "fp8_ds_mla",
        "e5m2": "fp8_e5m2",
        "bf16_cache": "auto",
    }.get(unsupported, "fp8")
    impl.kv_lora_rank = 512
    if unsupported == "dimensions":
        impl.kv_lora_rank = 256
    impl.qk_nope_head_dim = impl.v_head_dim = 128
    impl.qk_rope_head_dim = 64
    impl.num_heads = 1
    impl.kv_b_proj = SimpleNamespace(
        weight=torch.empty(256, 512, dtype=torch.bfloat16),
        weight_scale=torch.ones(1) if unsupported == "quantized_weight" else None,
        bias=torch.ones(256) if unsupported == "bias" else None,
    )
    if unsupported == "weight_dtype":
        impl.kv_b_proj.weight = impl.kv_b_proj.weight.half()
    elif unsupported == "weight_layout":
        impl.kv_b_proj.weight = torch.empty(512, 256, dtype=torch.bfloat16).T
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            q_data_type=torch.float16
            if unsupported == "prefill_dtype"
            else torch.bfloat16
        ),
        dcp_context_row_indices=None if unsupported == "no_indices" else [],
    )
    expected = (torch.ones(1), torch.zeros(1))

    def fused_mla_kv_concat(kv_nope, k_pe, use_fp8_prefill):
        raise AssertionError("The fallback should receive the callback")

    def original_path(*args):
        assert args[-1] is fused_mla_kv_concat
        return expected

    monkeypatch.setattr(
        MLACommonImpl,
        "_context_parallel_compute_prefill_context",
        original_path,
    )
    q = torch.empty(1, 1, 192, dtype=torch.bfloat16)
    cache = torch.empty(1, 1536, 576, dtype=torch.uint8)
    if unsupported == "query_dtype":
        q = q.half()
    elif unsupported == "cache_layout":
        cache = torch.empty(1, 1536, 1152, dtype=torch.uint8)[..., ::2]
    assert (
        impl._context_parallel_compute_prefill_context(
            q,
            cache,
            metadata,
            torch.ones(1),
            8,
            fused_mla_kv_concat_fn=fused_mla_kv_concat,
        )
        is expected
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm MLA backend")
@multi_gpu_only(num_gpus=2)
@pytest.mark.parametrize("interleave", [1, 64])
@pytest.mark.parametrize(
    "context_lens,workspace_size",
    [([145, 37], 1024), ([545, 0, 3, 0], 256)],
    ids=["single_chunk", "continuations_and_empty_context"],
)
def test_dcp_prefill_matches_full_attention(
    interleave, context_lens, workspace_size, tmp_path
):
    """Real DCP gather, fused expansion and suffix/chunk merges match dense SDPA."""
    import torch.multiprocessing as mp

    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("AITER on CDNA 3 or newer is required")
    mp.spawn(
        _dcp_prefill_worker,
        args=(
            f"file://{tmp_path / 'dist_init'}",
            interleave,
            context_lens,
            workspace_size,
        ),
        nprocs=2,
    )


@torch.inference_mode()
def _dcp_prefill_worker(rank, init_method, interleave, context_lens, workspace_size):
    import torch.nn.functional as F

    from tests.v1.attention.test_mla_backends import MockMLAAttentionLayer
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.attention.backends.mla.prefill.aiter_flash_attn import (
        AiterFlashAttnPrefillBackend,
    )
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        AiterMLAImpl,
        AiterMLAMetadataBuilder,
    )
    from vllm.v1.attention.ops.dcp import MLADCPManager
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}")
    init_workspace_manager(device)
    world_size, num_heads, block_size = 2, 12, 64
    config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        dtype=torch.bfloat16,
        tensor_parallel_size=world_size,
        max_model_len=2048,
        block_size=block_size,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
    )
    config.cache_config.cache_dtype = "fp8"
    config.parallel_config.decode_context_parallel_size = world_size
    config.parallel_config.cp_kv_cache_interleave_size = interleave
    config.attention_config.use_prefill_query_quantization = False
    config.kernel_config.enable_jit_warmup = False
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            decode_context_model_parallel_size=world_size,
        )
        try:
            projection = ColumnParallelLinear(
                512,
                num_heads * world_size * 256,
                bias=False,
                params_dtype=torch.bfloat16,
            ).to(device)
            torch.manual_seed(101 + rank)
            projection.weight.copy_(torch.randn_like(projection.weight) * 0.02)
            dims = dict(
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            )
            impl = AiterMLAImpl(
                num_heads=num_heads,
                head_size=576,
                scale=192**-0.5,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="fp8",
                logits_soft_cap=None,
                attn_type="decoder",
                kv_sharing_target_layer_name=None,
                q_lora_rank=None,
                qk_head_dim=192,
                kv_b_proj=projection,
                **dims,
            )
            layer = MockMLAAttentionLayer(
                impl=impl,
                num_heads=num_heads,
                device=device,
                kv_b_proj=projection,
                q_scale=1.0,
                k_scale=0.07,
                **dims,
            )
            layer.prefill_backend = AiterFlashAttnPrefillBackend(
                num_heads=num_heads,
                scale=impl.scale,
                vllm_config=config,
                **dims,
            )
            layer.dcp_manager = MLADCPManager(
                config,
                device,
                num_heads,
                576,
                512,
                torch.bfloat16,
                torch.bfloat16,
                None,
                True,
                False,
            )
            config.compilation_config.static_forward_context["mla"] = layer
            spec = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=576,
                dtype=current_platform.fp8_dtype(),
                cache_dtype_str="fp8",
            )
            builder = AiterMLAMetadataBuilder(spec, ["mla"], config, device)
            builder.chunked_prefill_workspace_size = workspace_size
            builder.chunked_prefill_workspace = torch.empty(
                (workspace_size + workspace_size // world_size, 576),
                dtype=torch.bfloat16,
                device=device,
            )
            layer.dcp_manager.init_kv_gather(
                builder.chunked_prefill_workspace, workspace_size
            )
            query_lens = [i + 2 for i in range(len(context_lens))]
            batch = BatchSpec(
                seq_lens=[c + q for c, q in zip(context_lens, query_lens)],
                query_lens=query_lens,
            )
            common = create_common_attn_metadata(
                batch, block_size, device, arange_block_indices=True
            )
            metadata = builder.build(0, common)
            assert metadata.prefill is not None
            chunked = metadata.prefill.chunked_context
            assert chunked is not None and metadata.dcp_context_row_indices is not None
            if 0 in context_lens:
                assert chunked.empty_token_slices
                assert any(c.is_continuation for c in chunked.chunks)
            else:
                assert len(chunked.chunks) == 1
            blocks_per_req = common.block_table_tensor.shape[1]
            cache = torch.zeros(
                (len(context_lens) * blocks_per_req, block_size, 576),
                dtype=torch.bfloat16,
                device=device,
            ).to(current_platform.fp8_dtype())
            queries, suffixes, references = [], [], []
            # All ranks reconstruct the same global cache before sharding it.
            torch.manual_seed(42)
            for request, (context_len, query_len) in enumerate(
                zip(context_lens, query_lens)
            ):
                latent = torch.randn(
                    context_len, 576, device=device, dtype=torch.bfloat16
                )
                quantized = (latent.float() / layer._k_scale).to(cache.dtype)
                positions = torch.arange(context_len, device=device)
                shard = quantized.view(torch.uint8)[
                    (positions // interleave) % world_size == rank
                ]
                cache.view(len(context_lens), -1, 576).view(torch.uint8)[
                    request, : len(shard)
                ].copy_(shard)
                suffix = torch.randn(
                    query_len, 576, device=device, dtype=torch.bfloat16
                )
                q = (
                    torch.randn(
                        query_len, num_heads, 192, device=device, dtype=torch.bfloat16
                    )
                    * 0.5
                )
                full_latent = torch.cat(
                    (
                        (quantized.float() * layer._k_scale).to(torch.bfloat16),
                        suffix,
                    )
                )
                kv = F.linear(full_latent[:, :512], projection.weight).view(
                    -1, num_heads, 256
                )
                k_nope, v = kv.split(128, dim=-1)
                k = torch.cat(
                    (k_nope, full_latent[:, None, 512:].expand(-1, num_heads, -1)),
                    dim=-1,
                )
                mask = torch.arange(context_len + query_len, device=device)[
                    None, :
                ] <= (context_len + torch.arange(query_len, device=device)[:, None])
                references.append(
                    F.scaled_dot_product_attention(
                        q.transpose(0, 1).float(),
                        k.transpose(0, 1).float(),
                        v.transpose(0, 1).float(),
                        attn_mask=mask,
                        scale=impl.scale,
                    ).transpose(0, 1)
                )
                queries.append(q)
                suffixes.append(suffix)
            q, suffix = torch.cat(queries), torch.cat(suffixes)
            output = torch.empty(
                q.shape[0], num_heads * 128, device=device, dtype=torch.bfloat16
            )
            # Ensure this test cannot silently succeed through the fallback.
            with pytest.MonkeyPatch.context() as patch:

                def unexpected_fallback(*args, **kwargs):
                    raise AssertionError("Expected the compressed DCP path")

                patch.setattr(
                    "vllm.model_executor.layers.attention.mla_attention."
                    "MLACommonImpl._context_parallel_compute_prefill_context",
                    unexpected_fallback,
                )
                impl.forward_mha(
                    q,
                    suffix[:, :512],
                    suffix[:, None, 512:],
                    cache,
                    metadata,
                    layer._k_scale,
                    output,
                )
            torch.testing.assert_close(
                output.view(-1, num_heads, 128).float(),
                torch.cat(references),
                rtol=0.03,
                atol=0.003,
            )
        finally:
            cleanup_dist_env_and_memory()
