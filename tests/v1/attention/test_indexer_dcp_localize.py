# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.dcp import CPTritonContext, correct_attn_out


def _sparse_dcp_parity_case(
    config, rank, world, next_n, seed, q_scale, k_scale, backend
):
    from vllm import _custom_ops as ops
    from vllm.distributed import get_dcp_group
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseImpl,
        FlashMLASparseMetadata,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.attention.ops.dcp import MLADCPManager
    from vllm.v1.attention.ops.flashmla import get_mla_metadata
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    is_flashmla = backend == "flashmla"
    impl_cls = FlashMLASparseImpl if is_flashmla else FlashInferMLASparseImpl

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(seed)
    # GLM-5.3 dimensions. MTP3 verifies the current token plus three drafts.
    heads, index_heads, index_dim, topk = 64, 32, 128, 2048
    block_size, interleave, kv_rank, rope_dim = 64, 64, 512, 64
    lengths = [5, 65, 129, 193, 257, 2049, 4097, 8193]
    if next_n == 1:
        lengths = [1, 63, 64, 65, 255, 256, 257, 2049, 8193]
    batch, tokens = len(lengths), len(lengths) * next_n
    capacity = ((max(lengths) + 255) // 256) * 256
    bounds = (
        torch.tensor(lengths, device=device, dtype=torch.int32)[:, None]
        - next_n
        + 1
        + torch.arange(next_n, device=device)
    ).to(torch.int32)

    # Generate all inputs before any rank/layout-dependent allocation or RNG use.
    q = torch.randn(
        tokens, heads, kv_rank + rope_dim, device=device, dtype=torch.bfloat16
    )
    index_q = torch.randn(
        tokens, index_heads, index_dim, device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    weights = torch.rand(tokens, index_heads, device=device) / index_heads**0.5
    index_k = torch.randn(
        batch, capacity, index_dim, device=device, dtype=torch.bfloat16
    )
    kv = torch.randn(
        batch, capacity, kv_rank + rope_dim, device=device, dtype=torch.bfloat16
    )
    if is_flashmla:
        # DS-MLA has per-tile cache scales and BF16 queries, rather than
        # FlashInfer's per-layer FP8 scales. Exercise two input magnitudes.
        q = (q * q_scale).to(q.dtype)
        kv = (kv * k_scale).to(kv.dtype)
    input_hashes = {
        name: hashlib.sha256(
            tensor.view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        for name, tensor in (
            ("query", q),
            ("index_query", index_q),
            ("index_weights", weights),
            ("index_keys", index_k),
            ("mla_kv", kv),
        )
    }
    if is_flashmla:
        q_input = q
    else:
        q_input, _ = ops.scaled_fp8_quant(
            q.flatten(1), torch.tensor(q_scale, device=device, dtype=torch.float32)
        )
        q_input = q_input.view_as(q)
    positions = torch.arange(capacity, device=device)
    owned = positions[(positions // interleave) % world == rank]
    local_capacity = owned.numel()
    blocks_per_req = local_capacity // block_size
    num_blocks = batch * blocks_per_req + 1
    # Unused page plus shuffled physical blocks expose logical/physical mixups.
    block_table = (
        torch.randperm(num_blocks - 1, device=device)
        .add(1)
        .reshape(batch, blocks_per_req)
        .to(torch.int32)
    )
    local_positions = torch.arange(local_capacity, device=device)
    slots = (
        block_table[:, local_positions // block_size].long() * block_size
        + local_positions % block_size
    ).flatten()
    index_cache = torch.zeros(
        num_blocks, block_size, index_dim + 4, dtype=torch.uint8, device=device
    )
    ops.indexer_k_quant_and_cache(
        index_k[:, owned].reshape(-1, index_dim), index_cache, slots, 128, "ue8m0"
    )
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        kv_rank + 16 + 2 * rope_dim if is_flashmla else kv_rank + rope_dim,
        dtype=torch.uint8 if is_flashmla else torch.float8_e4m3fn,
        device=device,
    )
    local_kv = kv[:, owned].reshape(-1, kv_rank + rope_dim)
    ops.concat_and_cache_mla(
        local_kv[:, :kv_rank],
        local_kv[:, kv_rank:],
        kv_cache,
        slots,
        "fp8_ds_mla" if is_flashmla else "fp8",
        torch.tensor(k_scale, device=device, dtype=torch.float32),
    )
    local_bounds = get_dcp_local_seq_lens(bounds, world, rank, interleave)
    # Construct the real builder so reintroducing the interleave guard fails
    # before attention runs. No model identity or backend allowlist is supplied.
    builder_config = copy.copy(config)
    builder_config.model_config = SimpleNamespace(max_model_len=capacity)
    if next_n > 1:
        builder_config.speculative_config = SimpleNamespace(
            num_speculative_tokens=next_n - 1, enable_adaptive_verification=False
        )
    builder = DeepseekV32IndexerMetadataBuilder(
        MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=index_dim + 4,
            dtype=torch.uint8,
        ),
        ["indexer"],
        builder_config,
        device,
        block_table_width=blocks_per_req,
    )
    query_starts = torch.arange(batch + 1, dtype=torch.int32) * next_n
    common = CommonAttentionMetadata(
        query_start_loc=query_starts.to(device),
        query_start_loc_cpu=query_starts,
        seq_lens=bounds[:, -1].contiguous(),
        seq_lens_cpu_upper_bound=torch.tensor(lengths, dtype=torch.int32),
        num_reqs=batch,
        num_actual_tokens=tokens,
        max_query_len=next_n,
        max_seq_len=max(lengths),
        block_table_tensor=block_table,
        slot_mapping=torch.full((tokens,), -1, dtype=torch.int64, device=device),
        dcp_local_seq_lens=local_bounds[:, -1] if world > 1 else None,
    )
    metadata = builder.build(0, common)
    assert metadata.decode is not None
    torch.testing.assert_close(
        metadata.decode.seq_lens.flatten(), local_bounds.flatten()
    )
    if world > 1:
        torch.testing.assert_close(
            metadata.decode.global_seq_lens.flatten(), bounds.flatten()
        )
    indices = torch.empty(tokens, topk, dtype=torch.int32, device=device)
    with set_forward_context({"indexer": metadata}, config):
        sparse_indexer.sparse_attn_indexer(
            hidden_states=q,
            k_cache_prefix="indexer",
            kv_cache=index_cache,
            q_quant=index_q,
            q_scale=None,
            k=None,
            weights=weights,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=topk,
            head_dim=index_dim,
            max_model_len=local_capacity,
            total_seq_lens=batch * local_capacity,
            topk_indices_buffer=indices,
            skip_k_cache_insert=True,
            use_pcp=False,
            dense_mha_metadata_layer_name="",
            dcp_rank=rank,
            dcp_world_size=world,
            cp_kv_cache_interleave_size=interleave,
        )
    valid = indices >= 0
    assert (indices[valid] < bounds.flatten()[:, None].expand_as(indices)[valid]).all()
    torch.testing.assert_close(valid.sum(1), bounds.flatten().clamp_max(topk).long())
    manager = None
    if world > 1:
        manager = MLADCPManager(
            config,
            device,
            heads // world,
            kv_rank + rope_dim,
            kv_rank,
            q_input.dtype,
            torch.bfloat16,
            None,
            impl_cls.lse_base_on_e,
            False,
        )
        query = manager.query_gather(q_input.chunk(world, dim=1)[rank].contiguous())
        torch.testing.assert_close(query.view(torch.uint8), q_input.view(torch.uint8))
    else:
        query = q_input

    # Set up the kernel-facing layer, with no projection weights or model load.
    impl = object.__new__(impl_cls)
    impl.scale = (192 + rope_dim) ** -0.5
    impl.qk_nope_head_dim, impl.kv_lora_rank = 192, kv_rank
    impl.qk_rope_head_dim, impl.kv_cache_dtype = rope_dim, "fp8"
    impl.topk_indices_buffer, impl.dcp_world_size, impl.dcp_rank = indices, world, rank
    impl._workspace_buffer = None
    impl.bmm1_scale = impl.bmm2_scale = None
    impl.is_nope_mla, impl.need_to_return_lse_for_decode = False, True
    attn_metadata = SimpleNamespace(
        req_id_per_token=torch.arange(
            batch, device=device, dtype=torch.int32
        ).repeat_interleave(next_n),
        block_table=block_table,
        block_size=block_size,
        cp_kv_cache_interleave_size=interleave,
    )
    if is_flashmla:
        impl.kv_cache_dtype = "fp8_ds_mla"
        impl.softmax_scale, impl.num_heads = impl.scale, heads // world
        impl.fp8_decode_padded_heads = 64
        scheduler, _ = get_mla_metadata(
            cache_seqlens=torch.full((1,), topk, device=device, dtype=torch.int32),
            num_q_tokens_per_head_k=tokens * heads,
            topk=topk,
            num_heads_q=heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        attn_metadata.fp8_use_mixed_batch = True
        attn_metadata.fp8_extra_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=scheduler,
            cache_lens=torch.full((1,), capacity, device=device, dtype=torch.int32),
            dummy_block_table=torch.empty((1, 1), device=device, dtype=torch.int32),
        )
    partial, lse = impl.forward_mqa(
        query,
        kv_cache,
        attn_metadata,
        SimpleNamespace(_q_scale_float=q_scale, _k_scale_float=k_scale),
    )
    assert lse is not None
    # Store LSE in natural-log units for comparison with the FP32 oracle.
    lse_natural = lse if impl.lse_base_on_e else lse * math.log(2)
    assert torch.isfinite(partial).all()
    empty = local_bounds.flatten() == 0
    assert (partial[empty] == 0).all() and torch.isneginf(lse[empty]).all()
    if manager is not None:
        out = manager.combine(partial, lse)
        out = get_dcp_group().all_gather(out, dim=1)
        lse_natural = (
            get_dcp_group().all_gather(lse_natural.unsqueeze(0), dim=0).logsumexp(0)
        )
        # The real indexer collective must produce the same set on every rank.
        gathered_ids = get_dcp_group().all_gather(indices.unsqueeze(0), dim=0)
        assert (gathered_ids.sort(-1).values == indices.sort(-1).values).all()
    else:
        out = partial
    result = {
        "input_hashes": input_hashes,
        "out": out.cpu(),
        "lse": lse_natural.cpu(),
        "indices": indices.cpu(),
        "causal_bounds": bounds.cpu(),
        "query_bytes": q_input.view(torch.uint8).cpu(),
    }
    if world == 1:
        # FP32 SDPA with the indexer's selected tokens checks shared attention
        # kernel errors, especially missing FP8 scales; it does not check top-k.
        if is_flashmla:
            raw = kv_cache.view(-1, kv_cache.shape[-1])[slots].contiguous()
            latent = raw[:, :kv_rank].contiguous().view(torch.float8_e4m3fn)
            scales = raw[:, kv_rank : kv_rank + 16].contiguous().view(torch.float32)
            # SM100 FlashMLA truncates tile scales to E8M0 before dequantizing.
            scales = torch.exp2(scales.log2().floor())
            latent = (latent.float().view(-1, 4, 128) * scales[..., None]).flatten(1)
            rope = raw[:, kv_rank + 16 :].contiguous().view(torch.bfloat16)
            canonical_kv = torch.cat((latent, rope.float()), -1).reshape(
                batch, capacity, kv_rank + rope_dim
            )
        else:
            canonical_kv = (
                kv_cache.view(-1, kv_rank + rope_dim)
                .float()[slots]
                .reshape(batch, capacity, kv_rank + rope_dim)
                * k_scale
            )
        reference = torch.empty(tokens, heads, kv_rank, device=device)
        reference_lse = torch.empty(tokens, heads, device=device)
        for row in range(tokens):
            selected = canonical_kv[row // next_n, indices[row, valid[row]].long()]
            query_scale = 1.0 if is_flashmla else q_scale
            scores = (q_input[row].float() * query_scale) @ selected.T * impl.scale
            reference[row] = scores.softmax(-1) @ selected[:, :kv_rank]
            reference_lse[row] = scores.logsumexp(-1)
        result.update(sdpa=reference.cpu(), sdpa_lse=reference_lse.cpu())
        if is_flashmla:
            # TP1's 64 local heads select the separate decode path in production.
            # Retain mixed-path LSE for diagnostics, but compare its real output.
            cache_lens = torch.full((batch,), topk, device=device, dtype=torch.int32)
            scheduler, _ = get_mla_metadata(
                cache_seqlens=cache_lens,
                num_q_tokens_per_head_k=next_n * heads,
                topk=topk,
                num_heads_q=heads,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )
            extra_cls = FlashMLASparseMetadata.FP8SeparatePrefillDecode
            attn_metadata.fp8_use_mixed_batch = False
            attn_metadata.fp8_extra_metadata = extra_cls(
                num_decodes=batch,
                num_decode_tokens=tokens,
                decode=extra_cls.Decode(
                    seq_lens=bounds[:, -1],
                    decode_query_len=next_n,
                    kernel_metadata=FlashMLASparseMetadata.FP8KernelMetadata(
                        scheduler_metadata=scheduler,
                        cache_lens=cache_lens,
                        dummy_block_table=torch.empty(
                            (batch, 1), device=device, dtype=torch.int32
                        ),
                    ),
                ),
            )
            separate_out, _ = impl.forward_mqa(query, kv_cache, attn_metadata, None)
            torch.testing.assert_close(separate_out, out, rtol=0.065, atol=0.05)
            result["out"] = separate_out.cpu()
    return result


def _mla_dcp_parity_worker(rank, world, directory, backend):
    from datetime import timedelta

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    torch.accelerator.set_device_index(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world,
            decode_context_parallel_size=world,
            cp_kv_cache_interleave_size=64,
            dcp_comm_backend="a2a" if backend == "flashinfer" else "ag_rs",
        )
    )
    directory = Path(directory)
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=world,
                rank=rank,
                local_rank=rank,
                backend="nccl",
                distributed_init_method=f"file://{directory / f'store-{world}'}",
                timeout=timedelta(minutes=5),
            )
            initialize_model_parallel(world, decode_context_model_parallel_size=world)
            init_workspace_manager(torch.device(f"cuda:{rank}"))
            if backend == "trtllm_ragged":
                from tests.v1.attention.test_mla_backends import (
                    _trtllm_ragged_dcp_parity_case,
                )

                for seed in (0, 42):
                    for query_len in (4, 65):
                        for k_scale in (1.0, 0.25):
                            result = _trtllm_ragged_dcp_parity_case(
                                config, rank, world, query_len, seed, k_scale
                            )
                            if rank == 0:
                                name = f"s{seed}-n{query_len}-k{k_scale}"
                                torch.save(result, directory / f"tp{world}-{name}.pt")
                return
            for seed in (0, 42):
                for next_n in (1, 4):
                    for q_scale, k_scale in ((1.0, 1.0), (0.5, 0.25)):
                        result = _sparse_dcp_parity_case(
                            config, rank, world, next_n, seed, q_scale, k_scale, backend
                        )
                        if rank == 0:
                            name = f"s{seed}-n{next_n}-q{q_scale}-k{k_scale}"
                            torch.save(result, directory / f"tp{world}-{name}.pt")
                            print(
                                f"ATTENTION_PARITY captured TP{world} {name}",
                                flush=True,
                            )
    finally:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="FlashInfer TRTLLM sparse MLA requires SM100",
)
@pytest.mark.parametrize("backend", ["flashinfer", "flashmla"])
def test_sparse_dcp4_interleave64_attention_matches_tp1(tmp_path, backend):
    """Real sparse decode/MTP outputs with DCP interleave64 match unsharded TP1.

    Requires four SM100 GPUs. No fake logits, selected indices, attention
    kernels, or collectives. Artifacts retain the actual output tensors.
    Output comparisons use the existing sparse-backend FP8 tolerance;
    indices are exact and natural-log LSE is checked at 1e-4.
    This kernel-facing test also exercises multi-token attention below the
    separate backend MTP compatibility checks; it does not change those checks.
    Sparse prefill kernels and fused model-level cache population are not
    covered. The FP32 oracle checks attention given the indexer's selections.
    """
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four GPUs for real TP4/DCP4 collectives")
    for world in (1, 4):
        torch.multiprocessing.spawn(
            _mla_dcp_parity_worker,
            args=(world, str(tmp_path), backend),
            nprocs=world,
            join=True,
        )
    metrics = []
    for ref_path in sorted(tmp_path.glob("tp1-*.pt")):
        ref = torch.load(ref_path, weights_only=True)
        candidate = torch.load(
            ref_path.with_name(ref_path.name.replace("tp1-", "tp4-")), weights_only=True
        )
        delta = candidate["out"].float() - ref["out"].float()
        metrics.append(
            {
                "case": ref_path.stem.removeprefix("tp1-"),
                "max_abs": delta.abs().max().item(),
                "relative_l2": (delta.norm() / ref["out"].float().norm()).item(),
                "lse_max_abs": (candidate["lse"] - ref["lse"]).abs().max().item(),
                "tp1_sdpa_lse_max_abs": (ref["lse"] - ref["sdpa_lse"])
                .abs()
                .max()
                .item(),
                "max_row_relative_l2": (
                    delta.flatten(1).norm(dim=1)
                    / ref["out"].float().flatten(1).norm(dim=1)
                )
                .max()
                .item(),
                "index_set_mismatches": (
                    candidate["indices"].sort(-1).values
                    != ref["indices"].sort(-1).values
                )
                .any(-1)
                .sum()
                .item(),
            }
        )
    assert len(metrics) == 8
    (tmp_path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    for ref_path in sorted(tmp_path.glob("tp1-*.pt")):
        ref = torch.load(ref_path, weights_only=True)
        candidate = torch.load(
            ref_path.with_name(ref_path.name.replace("tp1-", "tp4-")), weights_only=True
        )
        torch.testing.assert_close(
            candidate["query_bytes"], ref["query_bytes"], rtol=0, atol=0
        )
        assert candidate["input_hashes"] == ref["input_hashes"]
        torch.testing.assert_close(candidate["causal_bounds"], ref["causal_bounds"])
        torch.testing.assert_close(
            candidate["indices"].sort(-1).values,
            ref["indices"].sort(-1).values,
            rtol=0,
            atol=0,
        )
        for actual in (ref["out"], candidate["out"]):
            # FP8 softmax rounding and cancellation can amplify individual
            # coordinates. Check the independent FP32 oracle per query;
            # the actual TP1/DCP4 comparison below remains elementwise.
            oracle_relative_error = (actual.float() - ref["sdpa"]).flatten(1).norm(
                dim=1
            ) / ref["sdpa"].flatten(1).norm(dim=1)
            assert (oracle_relative_error < 0.065).all(), ref_path.name
        # Sharding changes the softmax intermediate's FP8 rounding, even with
        # identical quantized Q/K/V. Use the sparse backend's FP8 budget here too.
        torch.testing.assert_close(candidate["out"], ref["out"], rtol=0.065, atol=0.05)
        torch.testing.assert_close(candidate["lse"], ref["lse"], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ref["lse"], ref["sdpa_lse"], rtol=1e-4, atol=1e-4)
    # Check each query separately so exact short rows cannot dilute an error.
    assert all(case["max_row_relative_l2"] < 0.065 for case in metrics)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("first_causal_len", [63, 255])
@pytest.mark.parametrize("dcp_rank", range(4))
def test_flashinfer_sparse_dcp_mtp_keeps_per_token_causal_pages(
    monkeypatch, first_causal_len: int, dcp_rank: int
):
    """MTP rows crossing an interleave boundary retain individual causal masks.

    Exercise the real owner filtering and physical page conversion. Only the
    FlashInfer launcher is replaced: it must receive one query per row, not a
    multi-query local causal triangle, including rows with no local candidates.
    """
    from types import SimpleNamespace

    import flashinfer.decode

    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    device = torch.device("cuda")
    num_tokens, num_heads, kv_lora_rank = 4, 2, 512
    block_size, interleave, world, topk = 64, 64, 4, 512
    causal_lens = list(range(first_causal_len, first_causal_len + num_tokens))
    global_ids = torch.arange(topk, device=device, dtype=torch.int32)
    topk_indices = torch.where(
        global_ids.unsqueeze(0) < torch.tensor(causal_lens, device=device).unsqueeze(1),
        global_ids,
        -1,
    )
    # Non-identity pages catch confusing local token IDs with physical slots.
    block_ids = [3, 1]
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        block_table=torch.tensor([block_ids], dtype=torch.int32, device=device),
        block_size=block_size,
        cp_kv_cache_interleave_size=interleave,
    )
    expected_slots = []
    for length in causal_lens:
        owned = [
            pos for pos in range(length) if (pos // interleave) % world == dcp_rank
        ]
        local = [
            (pos // (world * interleave)) * interleave + pos % interleave
            for pos in owned
        ]
        expected_slots.append(
            sorted(
                block_ids[pos // block_size] * block_size + pos % block_size
                for pos in local
            )
        )
    expected_counts = torch.tensor(
        [len(slots) for slots in expected_slots], dtype=torch.int32, device=device
    )

    impl = object.__new__(FlashInferMLASparseImpl)
    impl.scale = 1.0
    impl.qk_nope_head_dim = 192
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_rope_head_dim = 64
    impl.kv_cache_dtype = "fp8"
    impl.topk_indices_buffer = topk_indices
    impl.dcp_world_size = world
    impl.dcp_rank = dcp_rank
    impl._workspace_buffer = torch.empty(1, device=device, dtype=torch.int8)
    impl.bmm1_scale = None
    impl.bmm2_scale = None
    impl.is_nope_mla = False
    impl.need_to_return_lse_for_decode = True
    q = torch.zeros(
        num_tokens,
        num_heads,
        kv_lora_rank + 64,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    launches = 0

    def fake_flashinfer(**kwargs):
        nonlocal launches
        launches += 1
        assert kwargs["query"].shape == (num_tokens, 1, num_heads, kv_lora_rank + 64)
        pages = kwargs["block_tables"]
        assert pages.shape == (num_tokens, 1, topk)
        torch.testing.assert_close(kwargs["seq_lens"], expected_counts)
        for row, slots in enumerate(expected_slots):
            count = len(slots)
            assert sorted(pages[row, 0, :count].tolist()) == slots
            assert (pages[row, 0, count:] == -1).all()

        out = torch.ones(num_tokens, 1, num_heads, kv_lora_rank, device=device)
        lse = torch.ones(num_tokens, num_heads, 1, device=device)
        # Empty sparse rows can have undefined kernel partials. They must not
        # contaminate the later cross-rank LSE reduction, even at zero weight.
        out[expected_counts == 0] = float("nan")
        lse[expected_counts == 0] = float("nan")
        return out, lse

    monkeypatch.setattr(
        flashinfer.decode, "trtllm_batch_decode_with_kv_cache_mla", fake_flashinfer
    )
    out, lse = impl.forward_mqa(
        q,
        torch.zeros(4, block_size, kv_lora_rank + 64, device=device),
        metadata,
        SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0),
    )
    assert launches == 1
    assert lse is not None
    empty = expected_counts == 0
    assert (out[empty] == 0).all()
    assert torch.isneginf(lse[empty]).all()
    assert (out[~empty] == 1).all()
    assert (lse[~empty] == 1).all()


def _local_count(length: int, rank: int, world: int, interleave: int) -> int:
    return sum(1 for pos in range(length) if (pos // interleave) % world == rank)


def _global_to_local_indices(
    global_indices: torch.Tensor,
    rank: int,
    world: int,
    interleave: int,
) -> torch.Tensor:
    valid = global_indices >= 0
    global_i64 = global_indices.to(torch.int64).clamp_min(0)
    owner = (global_i64 // interleave) % world
    local = (global_i64 // (world * interleave)) * interleave + global_i64 % interleave
    return torch.where(valid & (owner == rank), local, -1).to(torch.int64)


def _local_to_global_indices(
    local_indices: torch.Tensor,
    rank: int,
    world: int,
    interleave: int,
) -> torch.Tensor:
    valid = local_indices >= 0
    local = local_indices.to(torch.int64).clamp_min(0)
    global_indices = (
        (local // interleave) * (world * interleave)
        + rank * interleave
        + local % interleave
    )
    return torch.where(valid, global_indices, -1).to(torch.int64)


def _ref_stable_topk_from_candidates_fp64(
    candidate_scores: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Pure-PyTorch reference for the CuteDSL stable-topk selector order
    (score desc, then lowest global token id). Selects the same SET as the
    kernel; only the set is compared in tests."""
    num_rows, num_candidates = candidate_scores.shape
    device = candidate_scores.device
    select_k = min(k, num_candidates)
    valid = candidate_token_ids >= 0
    bits = (
        candidate_scores.to(torch.float32).view(torch.int32).to(torch.int64)
        & 0xFFFFFFFF
    )
    sign = (bits >> 31) & 1
    score_key = (
        torch.where(sign.bool(), bits ^ 0xFFFFFFFF, bits ^ 0x80000000) & 0xFFFFFFFF
    )
    id_key = (~candidate_token_ids.to(torch.int64)) & 0xFFFFFFFF
    key = (score_key << 32) | id_key
    key = torch.where(valid, key, torch.zeros_like(key))
    topk_key = key ^ torch.iinfo(torch.int64).min
    _, topk_pos = topk_key.topk(select_k, dim=-1)

    selected = candidate_token_ids.gather(1, topk_pos).to(torch.int32)
    selected_valid = valid.gather(1, topk_pos)
    selected = torch.where(selected_valid, selected, selected.new_full((), -1))
    if select_k == k:
        return selected
    pad = torch.full((num_rows, k - select_k), -1, dtype=torch.int32, device=device)
    return torch.cat((selected, pad), dim=1)


def _attention_from_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = indices >= 0
    safe_indices = indices.clamp_min(0)
    selected_k = k[safe_indices]
    selected_v = v[safe_indices]
    scores = torch.einsum("td,tkd->tk", q, selected_k)
    scores = scores.masked_fill(~valid, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).masked_fill(~valid, 0.0)
    out = torch.einsum("tk,tkd->td", probs, selected_v)
    empty_rows = ~valid.any(dim=-1)
    out[empty_rows] = 0
    lse[empty_rows] = float("-inf")
    return out, lse


def _dcp_lse_merge(
    local_outs: list[torch.Tensor],
    local_lses: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    outs = torch.stack(local_outs, dim=0)
    lses = torch.stack(local_lses, dim=0)
    merged_lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - merged_lse.unsqueeze(0))
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    merged_out = (outs * weights.unsqueeze(-1)).sum(dim=0)
    return merged_out, merged_lse


class _FakeDCPGroup:
    """Single-process stand-in: ``all_gather`` returns the pre-built
    concatenation of every rank's packed ``(score, global_id)`` candidates,
    mirroring the one packed all-gather the merge issues."""

    def __init__(self, gathered_packed: torch.Tensor) -> None:
        self.gathered_packed = gathered_packed

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 1
        return self.gathered_packed.clone()


def _run_decode_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    topk: int,
) -> torch.Tensor:
    indices = torch.empty(
        (logits.shape[0], topk), dtype=torch.int32, device=logits.device
    )
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk,
    )
    return indices


def _run_persistent_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    max_seq_len: int,
) -> torch.Tensor:
    indices = torch.empty(
        (logits.shape[0], topk), dtype=torch.int32, device=logits.device
    )
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=logits.device)
    torch.ops._C.persistent_topk(
        logits,
        seq_lens,
        indices,
        workspace,
        topk,
        max_seq_len,
    )
    return indices


def _dcp_attention_from_local_topks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    local_topks: list[torch.Tensor],
    world: int,
    interleave: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_outs = []
    local_lses = []
    for rank, local_topk in enumerate(local_topks):
        owned = [
            pos for pos in range(k.shape[0]) if (pos // interleave) % world == rank
        ]
        local_out, local_lse = _attention_from_indices(
            q, k[owned], v[owned], local_topk.to(torch.int64)
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)
    return _dcp_lse_merge(local_outs, local_lses)


def _merge_local_topks_global_with_fake_dcp(
    local_logits: list[torch.Tensor],
    local_topks: list[torch.Tensor],
    topk: int,
    world: int,
    interleave: int,
    row_starts: list[torch.Tensor | None] | None = None,
) -> list[torch.Tensor]:
    """Run ``_merge_dcp_topk_global`` for every rank against a faked
    all-gather and return each rank's global-index result (all ranks should
    agree). The fake pre-builds the packed candidate concatenation the merge's
    single ``all_gather`` would return."""
    # The merge is now CuteDSL-only (no PyTorch fallback), so it runs the real
    # Triton pack + CuteDSL selector kernels even behind the faked all-gather.
    if not current_platform.is_cuda() or not has_cutedsl():
        pytest.skip("DCP merge requires CUDA and CuteDSL")
    packed_per_rank = []
    for rank, (logits, indices) in enumerate(zip(local_logits, local_topks)):
        score_indices = indices.clamp_min(0).to(torch.long)
        if row_starts is not None:
            rs = row_starts[rank]
            assert rs is not None
            score_indices = score_indices + rs.to(torch.long).view(-1, 1)
        if logits.shape[1] == 0:
            scores = torch.full_like(indices, float("-inf"), dtype=torch.float32)
        else:
            score_indices = score_indices.clamp_max(logits.shape[1] - 1)
            scores = logits.gather(1, score_indices).masked_fill(
                indices < 0, float("-inf")
            )
        global_ids = _local_to_global_indices(indices, rank, world, interleave)
        packed_per_rank.append(
            torch.stack((scores.float(), global_ids.to(torch.float32)), dim=-1)
        )

    fake_group = _FakeDCPGroup(torch.cat(packed_per_rank, dim=1).contiguous())
    original_get_dcp_group = sparse_indexer.get_dcp_group
    sparse_indexer.get_dcp_group = lambda: fake_group
    try:
        merged = []
        for rank, (logits, indices) in enumerate(zip(local_logits, local_topks)):
            rank_indices = indices.clone()
            result = sparse_indexer._merge_dcp_topk_global(
                logits,
                rank_indices,
                topk,
                rank,
                world,
                interleave,
                row_starts=None if row_starts is None else row_starts[rank],
            )
            assert result is None
            merged.append(rank_indices)
        return merged
    finally:
        sparse_indexer.get_dcp_group = original_get_dcp_group


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 2, 4, 64])
def test_get_dcp_local_seq_lens_matches_naive(world: int, interleave: int):
    seq_lens = torch.tensor(
        [*range(33), 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513],
        dtype=torch.int32,
    )

    for rank in range(world):
        actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        expected = torch.tensor(
            [
                _local_count(int(seq_len), rank, world, interleave)
                for seq_len in seq_lens
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_can_localize_per_token_bounds():
    seq_lens = torch.tensor([0, 1, 2, 3, 4, 7, 8, 17], dtype=torch.int32)
    world = 4
    interleave = 2

    for rank in range(world):
        actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        expected = torch.tensor(
            [
                _local_count(int(seq_len), rank, world, interleave)
                for seq_len in seq_lens
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("interleave", [1, 64])
def test_get_dcp_local_seq_lens_preserves_mtp_bounds_shape(interleave: int):
    seq_lens = torch.tensor(
        [[8, 9, 10], [63, 64, 65], [255, 256, 257]], dtype=torch.int32
    )
    world = 2
    rank = 1

    actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
    expected = torch.tensor(
        [
            [_local_count(int(seq_len), rank, world, interleave) for seq_len in row]
            for row in seq_lens
        ],
        dtype=torch.int32,
    )

    assert actual.shape == seq_lens.shape
    torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_must_run_after_decode_expansion():
    world = 2
    rank = 1
    interleave = 1
    expanded_bounds = torch.tensor([8, 9, 10], dtype=torch.int32)

    localized_after_expansion = get_dcp_local_seq_lens(
        expanded_bounds, world, rank, interleave
    )
    localized_request_len_minus_offsets = get_dcp_local_seq_lens(
        torch.tensor([10], dtype=torch.int32), world, rank
    ) - torch.tensor([2, 1, 0], dtype=torch.int32)

    assert not torch.equal(
        localized_after_expansion, localized_request_len_minus_offsets
    )
    torch.testing.assert_close(
        localized_after_expansion, torch.tensor([4, 4, 5], dtype=torch.int32)
    )


@pytest.mark.parametrize("interleave", [1, 2])
def test_sparse_dcp_attention_matches_global_topk_attention(interleave: int):
    torch.manual_seed(0)
    world = 2
    topk = 3
    num_queries = 4
    max_seq_len = 13
    head_dim = 8

    q = torch.randn(num_queries, head_dim)
    k = torch.randn(max_seq_len, head_dim)
    v = torch.randn(max_seq_len, head_dim)
    seq_lens = torch.tensor([6, 8, 11, 13], dtype=torch.int64)

    scores = q @ k.T
    global_topk = torch.full((num_queries, topk), -1, dtype=torch.int64)
    for row, seq_len in enumerate(seq_lens.tolist()):
        global_topk[row, :topk] = scores[row, :seq_len].topk(topk).indices

    ref_out, ref_lse = _attention_from_indices(q, k, v, global_topk)

    local_outs = []
    local_lses = []
    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        k_local = k[owned]
        v_local = v[owned]
        local_topk = _global_to_local_indices(global_topk, rank, world, interleave)
        local_topks.append(local_topk)
        local_out, local_lse = _attention_from_indices(q, k_local, v_local, local_topk)
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)

    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)

    gathered_global = torch.cat(
        [
            _local_to_global_indices(local_topks[rank], rank, world, interleave)
            for rank in range(world)
        ],
        dim=1,
    )
    assert set(gathered_global[gathered_global >= 0].tolist()) == set(
        global_topk.flatten().tolist()
    )


def test_local_topk_union_is_not_equivalent_to_global_topk_attention():
    world = 2
    interleave = 1
    topk = 2
    q = torch.tensor([[1.0]])
    k = torch.tensor(
        [
            [1.00],
            [0.90],
            [0.95],
            [0.85],
        ]
    )
    v = torch.tensor([[0.0], [1000.0], [0.0], [1000.0]])

    scores = q @ k.T
    global_topk = scores.topk(topk, dim=-1).indices
    ref_out, ref_lse = _attention_from_indices(q, k, v, global_topk)

    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(k.shape[0]) if (pos // interleave) % world == rank
        ]
        local_topks.append(scores[:, owned].topk(topk, dim=-1).indices)

    local_union_out, local_union_lse = _dcp_attention_from_local_topks(
        q, k, v, local_topks, world, interleave
    )

    assert not torch.allclose(local_union_out, ref_out)
    assert not torch.allclose(local_union_lse, ref_lse)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("interleave", [1, 64])
def test_sparse_decode_dcp_persistent_topk_matches_non_dcp(interleave: int):
    torch.manual_seed(3)
    device = torch.device("cuda")
    world = 2
    topk = 512
    num_rows = 2
    max_seq_len = 1025
    head_dim = 16

    q = torch.randn(num_rows, head_dim, device=device)
    k = torch.randn(max_seq_len, head_dim, device=device)
    v = torch.randn(max_seq_len, head_dim, device=device)
    logits = q @ k.T
    seq_lens = torch.tensor([[1024], [1025]], dtype=torch.int32, device=device)

    non_dcp_topk = torch.empty((num_rows, topk), dtype=torch.int64, device=device)
    for row, seq_len in enumerate(seq_lens.flatten().tolist()):
        non_dcp_topk[row] = logits[row, :seq_len].topk(topk).indices
    ref_out, ref_lse = _attention_from_indices(q, k, v, non_dcp_topk)

    local_logits = []
    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        rank_logits = logits[:, owned].contiguous()
        rank_seq_lens = get_dcp_local_seq_lens(
            seq_lens, world, rank, interleave
        ).contiguous()
        local_logits.append(rank_logits)
        local_topks.append(
            _run_persistent_topk(
                rank_logits,
                rank_seq_lens,
                topk,
                max_seq_len=rank_logits.shape[1],
            )
        )

    merged_global_topks = _merge_local_topks_global_with_fake_dcp(
        local_logits, local_topks, topk, world, interleave
    )
    # The radix top-K kernel selects a deterministic SET but writes it in
    # nondeterministic (atomicAdd) order; the production path is permutation-
    # invariant (compaction + softmax), so all ranks must agree on the set, not
    # the array order. (The fp64 fallback happens to return sorted order.)
    ref_topk = merged_global_topks[0]
    for rank_topk in merged_global_topks[1:]:
        for row in range(rank_topk.shape[0]):
            assert set(rank_topk[row].tolist()) == set(ref_topk[row].tolist())

    local_outs = []
    local_lses = []
    for rank, global_topk in enumerate(merged_global_topks):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        local_topk = _global_to_local_indices(
            global_topk.to(torch.int64),
            rank,
            world,
            interleave,
        )
        local_out, local_lse = _attention_from_indices(
            q, k[owned], v[owned], local_topk
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)
    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_cutedsl(),
    reason="This test requires CUDA and CuteDSL",
)
@pytest.mark.parametrize("use_row_starts", [False, True])
@pytest.mark.parametrize("interleave", [1, 64])
def test_cutedsl_dcp_candidate_pack_and_select_matches_reference(
    use_row_starts: bool, interleave: int
):
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    torch.manual_seed(13)
    device = torch.device("cuda")
    rows = 4
    valid_width = 1024
    width = valid_width + (8 if use_row_starts else 0)
    topk = 512
    world = 2
    row_starts = (
        torch.tensor([0, 2, 4, 1], device=device, dtype=torch.int32)
        if use_row_starts
        else None
    )
    row_offsets = (
        row_starts
        if row_starts is not None
        else torch.zeros(rows, device=device, dtype=torch.int32)
    )

    packed_by_rank = []
    for rank in range(world):
        logits = torch.randn((rows, width), device=device, dtype=torch.float32)
        local_topks = []
        for row in range(rows):
            start = int(row_offsets[row].item())
            local_topks.append(
                logits[row, start : start + valid_width].topk(topk).indices
            )
        topk_indices = torch.stack(local_topks).to(torch.int32)

        packed = torch.empty((rows, topk, 2), device=device, dtype=torch.float32)
        pack_dcp_topk_candidates_cutedsl(
            logits,
            topk_indices,
            packed,
            rank,
            world,
            interleave,
            row_starts,
        )

        expected_scores = logits.gather(
            1, topk_indices.to(torch.long) + row_offsets.to(torch.long).view(-1, 1)
        )
        expected_ids = _local_to_global_indices(
            topk_indices, rank, world, interleave
        ).to(torch.float32)
        torch.testing.assert_close(packed[..., 0], expected_scores)
        torch.testing.assert_close(packed[..., 1], expected_ids)
        packed_by_rank.append(packed)

    gathered = torch.cat(packed_by_rank, dim=1).contiguous()
    actual = torch.empty((rows, topk), device=device, dtype=torch.int32)
    returned = stable_topk_from_gathered_candidates_cutedsl(gathered, topk, out=actual)
    assert returned is actual
    expected = _ref_stable_topk_from_candidates_fp64(
        gathered[..., 0],
        gathered[..., 1].to(torch.int32),
        topk,
    )

    for row in range(rows):
        assert set(actual[row].cpu().tolist()) == set(expected[row].cpu().tolist())


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_sparse_prefill_dcp_metadata_localizes_causal_bounds():
    device = torch.device("cuda")
    seq_len = 8

    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    query_start_loc_cpu = torch.tensor([0, seq_len], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)

    def build(dcp_world_size, dcp_rank, interleave=1):
        chunk = build_prefill_chunk_metadata(
            start_idx=0,
            end_idx=1,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            uncompressed_seq_lens=seq_lens,
            compressed_seq_lens=seq_lens,
            compressed_seq_lens_cpu=seq_lens_cpu,
            block_table=block_table,
            compress_ratio=1,
            dcp_rank=dcp_rank,
            dcp_world_size=dcp_world_size,
            cp_kv_cache_interleave_size=interleave,
        )
        assert chunk is not None
        torch.accelerator.synchronize()
        return chunk

    # Non-DCP: local_cu_seq_lens aliases the global cu_seq_lens, and
    # cu_seqlen_ks/ke carry the global causal bounds.
    chunk = build(dcp_world_size=1, dcp_rank=0)
    assert chunk.local_cu_seq_lens is chunk.cu_seq_lens
    torch.testing.assert_close(
        chunk.cu_seqlen_ks.cpu(),
        torch.zeros(seq_len, dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ke.cpu(),
        torch.arange(1, seq_len + 1, dtype=torch.int32),
    )

    # DCP: cu_seqlen_ks/ke are localized in place to this rank's shard.
    chunk = build(dcp_world_size=4, dcp_rank=0)
    assert chunk.local_cu_seq_lens is not None
    torch.testing.assert_close(
        chunk.local_cu_seq_lens.cpu(),
        torch.tensor([0, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ks.cpu(),
        torch.zeros(seq_len, dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ke.cpu(),
        torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32),
    )

    # DCP with interleave=2: per-token causal bounds localize differently from
    # interleave=1 (groups of 2 consecutive tokens are owned together). For
    # world=4, rank=0, K=2, per-token global len L=1..8 -> local len
    # [1,2,2,2,2,2,2,2] (matches get_dcp_local_seq_lens).
    chunk = build(dcp_world_size=4, dcp_rank=0, interleave=2)
    assert chunk.local_cu_seq_lens is not None
    torch.testing.assert_close(
        chunk.cu_seqlen_ks.cpu(),
        torch.zeros(seq_len, dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ke.cpu(),
        torch.tensor([1, 2, 2, 2, 2, 2, 2, 2], dtype=torch.int32),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("interleave", [1, 64])
@pytest.mark.parametrize("query_slice", [None, slice(1, 5)])
def test_sparse_prefill_dcp_chunk_causal_bounds_at_block_boundaries(
    rank: int, interleave: int, query_slice: slice | None
):
    """Cached, batched queries retain rank-local causal bounds when sliced."""
    world = 4
    device = torch.device("cuda")
    seq_lens_cpu = torch.tensor([65, 257], dtype=torch.int32)
    query_start_loc_cpu = torch.tensor([0, 3, 6], dtype=torch.int32)
    seq_lens = seq_lens_cpu.to(device)
    chunk = build_prefill_chunk_metadata(
        start_idx=0,
        end_idx=2,
        query_start_loc=query_start_loc_cpu.to(device),
        query_start_loc_cpu=query_start_loc_cpu,
        uncompressed_seq_lens=seq_lens,
        compressed_seq_lens=seq_lens,
        compressed_seq_lens_cpu=seq_lens_cpu,
        block_table=torch.zeros((2, 5), dtype=torch.int32, device=device),
        compress_ratio=1,
        query_slice=query_slice,
        dcp_rank=rank,
        dcp_world_size=world,
        cp_kv_cache_interleave_size=interleave,
    )
    assert chunk is not None
    local_first_len = _local_count(65, rank, world, interleave)
    local_second_len = _local_count(257, rank, world, interleave)
    expected_starts = [0] * 3 + [local_first_len] * 3
    expected_ends = [
        start + _local_count(bound, rank, world, interleave)
        for start, bound in zip(expected_starts, [63, 64, 65, 255, 256, 257])
    ]
    query_slice = query_slice or slice(None)
    torch.testing.assert_close(
        chunk.cu_seqlen_ks.cpu(),
        torch.tensor(expected_starts[query_slice], dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ke.cpu(),
        torch.tensor(expected_ends[query_slice], dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.local_cu_seq_lens.cpu(),
        torch.tensor(
            [0, local_first_len, local_first_len + local_second_len],
            dtype=torch.int32,
        ),
    )
    assert chunk.local_total_seq_lens == local_first_len + local_second_len


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("block_stride_rows", [None, 12])
def test_dcp_filter_compacts_valid_slots_for_sparse_kernel(
    block_stride_rows: int | None,
):
    block_size = 4
    num_topk = 128
    dcp_size = 2
    req_id = torch.zeros(1, dtype=torch.int32, device="cuda")
    token_indices = torch.full((1, num_topk), -1, dtype=torch.int32, device="cuda")
    token_indices[0, :8] = torch.arange(8, dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[10]], dtype=torch.int32, device="cuda")

    out, valid_counts = triton_filter_and_convert_dcp_index(
        req_id,
        block_table,
        token_indices,
        dcp_size=dcp_size,
        dcp_rank=0,
        BLOCK_SIZE=block_size,
        BLOCK_STRIDE_ROWS=block_stride_rows,
        NUM_TOPK_TOKENS=num_topk,
        return_valid_counts=True,
    )

    valid = int(valid_counts.item())
    assert valid == 4
    assert (out[0, :valid] >= 0).all()
    assert (out[0, valid:] == -1).all()
    # In-kernel compaction packs valid slots to the front; prefix order is
    # unspecified, so compare as a set.
    block_stride_rows = block_stride_rows or block_size
    expected_base = 10 * block_stride_rows
    assert set(out[0, :valid].cpu().tolist()) == set(
        range(expected_base, expected_base + block_size)
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("interleave", [1, 2, 64])
@pytest.mark.parametrize("dcp_rank", [0, 1])
# 384 is not a power of two, so it exercises the multi-tile atomic allocator
# rather than the single-tile path 1024 takes.
@pytest.mark.parametrize("num_topk", [1024, 384])
def test_dcp_filter_compaction_matches_reference(
    interleave: int, dcp_rank: int, num_topk: int
):
    """In-kernel compaction must, for every row, produce exactly the rank-owned
    physical slots packed into [0, valid_count) with -1 in the tail -- the same
    SET a reference filter + sort/gather produces. Uses wide rows (> BLOCK_N
    valid slots), with interior -1 gaps."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    dcp_size = 2
    block_size = max(8, interleave)
    num_rows = 5
    max_blocks = 64
    seq = max_blocks * block_size

    req_id = torch.randint(0, 3, (num_rows,), dtype=torch.int32, device=device)
    block_table = torch.randint(
        0, 1000, (3, max_blocks), dtype=torch.int32, device=device
    )
    # Each row: a dense valid prefix of distinct global token ids, then -1 pad.
    token_indices = torch.full(
        (num_rows, num_topk), -1, dtype=torch.int32, device=device
    )
    for r in range(num_rows):
        # Ids are distinct, so a row holds at most `seq` valid entries.
        hi = min(num_topk * 3 // 4, seq)
        n_valid = int(torch.randint(num_topk // 4, hi, (1,)).item())
        perm = torch.randperm(seq, device=device)[:n_valid].to(torch.int32)
        token_indices[r, :n_valid] = perm

    out, valid_counts = triton_filter_and_convert_dcp_index(
        req_id,
        block_table,
        token_indices,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=interleave,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk,
        return_valid_counts=True,
    )

    for r in range(num_rows):
        toks = token_indices[r]
        toks = toks[toks >= 0]
        owner = (toks // interleave) % dcp_size
        owned = toks[owner == dcp_rank]
        local = (owned // (dcp_size * interleave)) * interleave + owned % interleave
        blk = local // block_size
        off = local % block_size
        expected = (
            block_table[int(req_id[r].item()), blk].to(torch.int64) * block_size + off
        )
        n = int(valid_counts[r].item())
        assert n == owned.numel()
        assert (out[r, :n] >= 0).all()
        assert (out[r, n:] == -1).all()
        assert set(out[r, :n].cpu().tolist()) == set(expected.cpu().tolist())


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("interleave", [1, 2, 4, 64])
def test_dcp_global_topk_physical_attention_matches_non_dcp(interleave: int):
    torch.manual_seed(2)
    device = torch.device("cuda")
    dcp_size = 2
    block_size = max(4, interleave)
    num_topk = 128
    selected_k = 8
    seq_len = 4 * block_size
    head_dim = 16
    num_queries = 3

    q = torch.randn(num_queries, head_dim, device=device)
    k_global = torch.randn(seq_len, head_dim, device=device)
    v_global = torch.randn(seq_len, head_dim, device=device)
    scores = q @ k_global.T
    global_topk = torch.full(
        (num_queries, num_topk), -1, dtype=torch.int32, device=device
    )
    global_topk[:, :selected_k] = scores.topk(selected_k, dim=-1).indices.to(
        torch.int32
    )

    ref_out, ref_lse = _attention_from_indices(
        q, k_global, v_global, global_topk[:, :selected_k].to(torch.int64)
    )

    local_outs = []
    local_lses = []
    for rank in range(dcp_size):
        block_ids = torch.tensor(
            [[rank * 10 + 1, rank * 10 + 2]], dtype=torch.int32, device=device
        )
        num_slots = int((block_ids.max().item() + 1) * block_size)
        k_cache = torch.zeros(num_slots, head_dim, device=device)
        v_cache = torch.zeros(num_slots, head_dim, device=device)
        for global_idx in range(seq_len):
            if (global_idx // interleave) % dcp_size != rank:
                continue
            local_idx = (
                global_idx // (dcp_size * interleave)
            ) * interleave + global_idx % interleave
            block = local_idx // block_size
            offset = local_idx % block_size
            slot = int(block_ids[0, block].item()) * block_size + offset
            k_cache[slot] = k_global[global_idx]
            v_cache[slot] = v_global[global_idx]

        slots, valid_counts = triton_filter_and_convert_dcp_index(
            torch.zeros(num_queries, dtype=torch.int32, device=device),
            block_ids,
            global_topk,
            dcp_size=dcp_size,
            dcp_rank=rank,
            cp_kv_cache_interleave_size=interleave,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=num_topk,
            return_valid_counts=True,
        )
        row_ids = torch.arange(num_queries, device=device)
        assert (slots[row_ids, valid_counts] == -1).all()
        local_out, local_lse = _attention_from_indices(
            q, k_cache, v_cache, slots.to(torch.int64)
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)
    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
def test_correct_attn_out_zeroes_empty_nan_partial(is_lse_base_on_e: bool):
    out = torch.full((1, 1, 4), float("nan"), device="cuda")
    lses = torch.tensor(
        [[[0.0]], [[float("-inf")]]],
        dtype=torch.float32,
        device="cuda",
    )

    corrected, final_lse = correct_attn_out(
        out,
        lses,
        cp_rank=1,
        ctx=CPTritonContext(),
        is_lse_base_on_e=is_lse_base_on_e,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(corrected, torch.zeros_like(corrected))
    torch.testing.assert_close(final_lse, torch.zeros_like(final_lse))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_decode_topk_pads_surplus_with_negative_one():
    """When a row's valid length < topk the top-k kernel must pad the surplus
    slots with -1: the DCP merge (`topk_indices >= 0`) and
    `triton_filter_and_convert_dcp_index` (`tok < 0`) both treat <0 as invalid,
    so a non-(-1) pad would be silently attended. This is the common case under
    DCP, where each rank's local seq_len = global / world is usually << topk.
    Checked on the *set* of valid indices (the kernel may order them by score,
    not position)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk = 8
    next_n = 1
    seq_lens_list = [0, 3, 5, 8]
    num_rows = len(seq_lens_list)
    logits = torch.randn(num_rows, 16, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device).view(
        num_rows, next_n
    )
    idx = _run_decode_topk(logits, seq_lens, next_n, topk)
    for r, sl in enumerate(seq_lens_list):
        valid = idx[r][idx[r] >= 0]
        # seq_len <= topk, so top-k selects exactly the whole valid range.
        assert valid.numel() == min(sl, topk)
        assert set(valid.tolist()) == set(range(sl))
        assert (idx[r] == -1).sum().item() == topk - min(sl, topk)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_persistent_topk_pads_surplus_with_negative_one():
    """Same surplus=-1 invariant for the persistent_topk kernel (k>=512)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk = 512  # persistent_topk requires k in {512, 1024, 2048}
    seq_lens_list = [100, 300, 512]
    num_rows = len(seq_lens_list)
    max_seq_len = 600
    logits = torch.randn(num_rows, max_seq_len, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device).view(
        num_rows, 1
    )
    idx = _run_persistent_topk(logits, seq_lens, topk, max_seq_len)
    for r, sl in enumerate(seq_lens_list):
        valid = idx[r][idx[r] >= 0]
        assert valid.numel() == min(sl, topk)
        assert set(valid.tolist()) == set(range(sl))
        assert (idx[r] == -1).sum().item() == topk - min(sl, topk)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 64])
def test_sparse_decode_dcp_short_context_matches_non_dcp(world: int, interleave: int):
    """End-to-end DCP decode where the global seq_len < topk (so every rank's
    local top-k is surplus-padded). Exercises the kernel surplus -> merge mask
    -> global top-k -> physical localize -> LSE merge chain for the common
    short-context decode case, vs the non-DCP reference."""
    torch.manual_seed(4)
    device = torch.device("cuda")
    topk = 512
    seq_lens_list = [63, 64, 65, 250, 255, 256, 257, 300]
    num_rows = len(seq_lens_list)
    max_seq_len = 300  # < topk -> surplus everywhere
    head_dim = 16

    q = torch.randn(num_rows, head_dim, device=device)
    k = torch.randn(max_seq_len, head_dim, device=device)
    v = torch.randn(max_seq_len, head_dim, device=device)
    logits = q @ k.T
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device).view(-1, 1)

    non_dcp_topk = torch.empty((num_rows, topk), dtype=torch.int64, device=device)
    for row, seq_len in enumerate(seq_lens.flatten().tolist()):
        sel = logits[row, :seq_len].topk(min(topk, seq_len)).indices
        non_dcp_topk[row, : sel.numel()] = sel
        non_dcp_topk[row, sel.numel() :] = -1
    ref_out, ref_lse = _attention_from_indices(q, k, v, non_dcp_topk)

    local_logits = []
    local_topks = []
    for rank in range(world):
        owned = [p for p in range(max_seq_len) if (p // interleave) % world == rank]
        rank_logits = logits[:, owned].contiguous()
        rank_seq_lens = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        local_logits.append(rank_logits)
        local_topks.append(
            _run_persistent_topk(
                rank_logits,
                rank_seq_lens.contiguous(),
                topk,
                max_seq_len=rank_logits.shape[1],
            )
        )

    merged_global_topks = _merge_local_topks_global_with_fake_dcp(
        local_logits, local_topks, topk, world, interleave
    )
    # The radix top-K kernel selects a deterministic SET but writes it in
    # nondeterministic (atomicAdd) order; the production path is permutation-
    # invariant (compaction + softmax), so all ranks must agree on the set, not
    # the array order. (The fp64 fallback happens to return sorted order.)
    ref_topk = merged_global_topks[0]
    for rank_topk in merged_global_topks[1:]:
        for row in range(rank_topk.shape[0]):
            assert set(rank_topk[row].tolist()) == set(ref_topk[row].tolist())

    local_outs = []
    local_lses = []
    for rank, global_topk in enumerate(merged_global_topks):
        owned = [p for p in range(max_seq_len) if (p // interleave) % world == rank]
        local_topk = _global_to_local_indices(
            global_topk.to(torch.int64), rank, world, interleave
        )
        local_out, local_lse = _attention_from_indices(
            q, k[owned], v[owned], local_topk
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)
    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)
