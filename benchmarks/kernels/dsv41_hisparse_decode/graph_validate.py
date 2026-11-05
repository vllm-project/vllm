# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation-only GPU shadow captured alongside the real decode CUDA graphs."""

import os
from pathlib import Path
from types import MethodType

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu_worker import Worker


@triton.jit
def _compare_rows(
    A,
    B,
    AI,
    BI,
    STATS,
    N: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    CHECK: tl.constexpr,
    BAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, byte = x // 584, x % 584
    ai = tl.load(AI + row, row < N, other=-1).to(tl.int64)
    bi = tl.load(BI + row, row < N, other=-1).to(tl.int64)
    valid = (row < N) & (ai >= 0) & (bi >= 0)
    a = tl.maximum(ai, 0)
    b = tl.maximum(bi, 0)
    ao = (a // AB) * AS + tl.where(
        byte < 576, (a % AB) * 576 + byte, AB * 576 + (a % AB) * 8 + byte - 576
    )
    bo = (b // BB) * BS + tl.where(
        byte < 576, (b % BB) * 576 + byte, BB * 576 + (b % BB) * 8 + byte - 576
    )
    av = tl.load(A + ao, valid, other=0)
    bv = tl.load(B + bo, valid, other=0)
    mismatch = (valid & (av != bv)) | ((row < N) & ((ai >= 0) != (bi >= 0)))
    checked = (row < N) & ((ai >= 0) | (bi >= 0))
    tl.atomic_add(STATS + CHECK, tl.sum(checked.to(tl.int64)))
    tl.atomic_add(STATS + BAD, tl.sum(mismatch.to(tl.int64)))


@triton.jit
def _compare_bytes(A, B, STATS, N: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + x, x < N, other=0)
    b = tl.load(B + x, x < N, other=0)
    tl.atomic_add(STATS + 6, tl.sum((x < N).to(tl.int64)))
    tl.atomic_add(STATS + 9, tl.sum(((x < N) & (a != b)).to(tl.int64)))


def compare_rows(a, b, ai, bi, stats, checked, bad):
    assert ai.numel() == bi.numel()
    _compare_rows[(triton.cdiv(ai.numel() * 584, 1024),)](
        a,
        b,
        ai,
        bi,
        stats,
        N=ai.numel(),
        AS=a.stride(0),
        BS=b.stride(0),
        AB=a.shape[1],
        BB=b.shape[1],
        CHECK=checked,
        BAD=bad,
        BLOCK=1024,
    )


class GraphValidator:
    def __init__(self, model, source_layer=20):
        from vllm.models.deepseek_v41 import compressor

        layers = {
            m.layer_id: m
            for m in model.modules()
            if type(m).__name__ == "DeepseekV4FlashMLAAttention"
        }
        self.source = layers[source_layer]
        self.source_layer = source_layer
        self.ratio = self.source.compress_ratio
        self.block_size = self.source.kv_cache.shape[1]
        pages = self.source.kv_cache.shape[0]
        stride = triton.cdiv(self.block_size * 584, 576) * 576
        self.shadow = torch.zeros(
            pages * stride, dtype=torch.uint8, device="cuda"
        ).as_strided((pages, self.block_size, 584), (stride, 584, 1))
        self.stats = torch.zeros(13, dtype=torch.int64, device="cuda")
        self.prefill_reuse_checks = 0
        self.prefill_attention_checks = 0
        self.host_copy_checks = 0
        self.writing = False
        original_publish = self.source.compressor.insert_cache

        def publish(owner, *args, **kwargs):
            self.writing = True
            try:
                return original_publish(*args, **kwargs)
            finally:
                self.writing = False

        self.source.compressor.insert_cache = MethodType(
            publish, self.source.compressor
        )
        original_insert = compressor.rope_quant_insert

        def insert(latent, positions, rope, cache, slots, ratio, **kwargs):
            result = original_insert(
                latent, positions, rope, cache, slots, ratio, **kwargs
            )
            if not self.writing:
                return result
            from vllm.forward_context import get_forward_context

            metadata = get_forward_context().attn_metadata[self.source.prefix]
            reference_slots = metadata.slot_mapping
            assert ratio == self.ratio
            original_insert(
                latent, positions, rope, self.shadow, reference_slots, ratio, **kwargs
            )
            written = ((positions + 1) % ratio == 0) & (slots >= 0)
            actual_rows = torch.where(written, slots, -1)
            reference_rows = torch.where(written, reference_slots, -1)
            compare_rows(
                cache, self.shadow, actual_rows, reference_rows, self.stats, 4, 7
            )
            self.stats[0].add_(1)
            return result

        compressor.rope_quant_insert = insert
        for index, layer in layers.items():
            if (
                index != source_layer
                and layer.compressed_cache_prefix != self.source.prefix
            ):
                continue
            original = layer._forward_decode
            original_prefill = layer._forward_prefill
            original_stage = layer._stage_hisparse_prefill

            def wrap_prefill(original_prefill):
                def prefill(attn, **kwargs):
                    result = original_prefill(**kwargs)
                    output = kwargs["output"]
                    actual = output.clone()
                    cache = self.source.hisparse_cache
                    try:
                        self.source.hisparse_cache = None
                        original_prefill(**dict(kwargs, compressed_k_cache=self.shadow))
                        a = actual.view(torch.uint8).reshape(-1)
                        b = output.view(torch.uint8).reshape(-1)
                        _compare_bytes[(triton.cdiv(a.numel(), 1024),)](
                            a, b, self.stats, N=a.numel(), BLOCK=1024
                        )
                        self.stats[2].add_(1)
                        self.prefill_attention_checks += 1
                    finally:
                        self.source.hisparse_cache = cache
                        output.copy_(actual)
                    return result

                return prefill

            layer._forward_prefill = MethodType(wrap_prefill(original_prefill), layer)

            def wrap_stage(original_stage):
                def stage(attn, host, table, metadata, swa, begin, end):
                    key = attn.compressed_cache_prefix, begin, end
                    reused = key in metadata.hisparse_prefill_caches
                    actual = original_stage(host, table, metadata, swa, begin, end)
                    cached = metadata.hisparse_prefill_caches.pop(key, None)
                    if cached is not None:
                        try:
                            reference = original_stage(
                                host, table, metadata, swa, begin, end
                            )
                        finally:
                            metadata.hisparse_prefill_caches[key] = cached
                        plan = metadata.hisparse_prefill_plans[key]
                        valid = plan.row_ids >= 0
                        if plan.gpu_row_ids is not None:
                            valid = valid | (plan.gpu_row_ids >= 0)
                        ids = torch.arange(
                            valid.numel(), dtype=torch.int64, device="cuda"
                        ).reshape_as(valid)
                        ids = torch.where(valid, ids, -1).reshape(-1)
                        compare_rows(
                            actual[0], reference[0], ids, ids, self.stats, 11, 12
                        )
                        # Check staging independently against native insertion,
                        # excluding unwritten rows after each request's tail.
                        lengths = swa.prefill_seq_lens[begin:end] // self.ratio
                        last = (
                            lengths.clamp_min(1).to(torch.int64) - 1
                        ) // self.block_size
                        last_blocks = table.gather(1, last[:, None]).reshape(-1)
                        tails = torch.where(
                            lengths > 0, (lengths - 1) % self.block_size + 1, 0
                        )
                        host_rows = plan.row_ids.reshape(-1)
                        unused = (
                            host_rows[None, :] // self.block_size
                            == last_blocks[:, None]
                        ) & (host_rows[None, :] % self.block_size >= tails[:, None])
                        active = (host_rows >= 0) & ~unused.any(dim=0)
                        staged_rows = torch.arange(
                            host_rows.numel(), dtype=torch.int64, device="cuda"
                        )
                        compare_rows(
                            actual[0],
                            self.shadow,
                            torch.where(active, staged_rows, -1),
                            torch.where(active, host_rows, -1),
                            self.stats,
                            11,
                            12,
                        )
                        self.stats[10].add_(1)
                        self.prefill_reuse_checks += int(reused)
                    return actual

                return stage

            layer._stage_hisparse_prefill = MethodType(
                wrap_stage(original_stage), layer
            )

            def wrap(original):
                def decode(attn, **kwargs):
                    from vllm.models.deepseek_v41.nvidia.flashmla import (
                        compute_global_topk_indices_and_lens,
                    )

                    result = original(**kwargs)
                    cache = self.source.hisparse_cache
                    metadata = kwargs["attn_metadata"]
                    swa = kwargs["swa_metadata"]
                    indices, _ = compute_global_topk_indices_and_lens(
                        attn.topk_indices_buffer[: swa.num_decode_tokens],
                        swa.token_to_req_indices,
                        metadata.block_table[: swa.num_decodes],
                        self.block_size,
                        swa.is_valid_token[: swa.num_decode_tokens],
                    )
                    hot_indices, _ = self.source._hisparse_topk
                    compare_rows(
                        cache.runtime.hot.attention_cache,
                        self.shadow,
                        hot_indices.reshape(-1),
                        indices.reshape(-1),
                        self.stats,
                        5,
                        8,
                    )
                    self.stats[1].add_(1)
                    if attn.is_index_source:
                        self.stats[3].add_(
                            cache.runtime.index_group.shared_topk.swap_counts[
                                : swa.num_decode_tokens
                            ].sum()
                        )
                    output = kwargs["output"]
                    actual = output.clone()
                    try:
                        self.source.hisparse_cache = None
                        original(**dict(kwargs, kv_cache=self.shadow))
                        a = actual.view(torch.uint8).reshape(-1)
                        b = output.view(torch.uint8).reshape(-1)
                        _compare_bytes[(triton.cdiv(a.numel(), 1024),)](
                            a, b, self.stats, N=a.numel(), BLOCK=1024
                        )
                        self.stats[2].add_(1)
                    finally:
                        self.source.hisparse_cache = cache
                        output.copy_(actual)
                    return result

                return decode

            layer._forward_decode = MethodType(wrap(original), layer)


class ValidatingWorker(Worker):
    def compile_or_warm_up_model(self):
        install_validators(self)
        return super().compile_or_warm_up_model()

    def execute_model(self, *args, **kwargs):
        result = super().execute_model(*args, **kwargs)
        destination = os.environ.get("DSV41_VALIDATION_DUMPS")
        if destination and getattr(self.model_runner, "validation_active", False):
            for validator in self.graph_validators:
                values = validator.stats.cpu().tolist()
                if any(values[7:10]) or values[12]:
                    dump_mismatch(self, validator, destination, values)
                    raise AssertionError((validator.source_layer, values))
        return result


def dump_mismatch(worker, validator, destination, values):
    from validate import records

    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group

    runner = worker.model_runner
    contexts = [runner.execute_model_state.attn_metadata]
    replay = runner.model_state.decoder_replay_layers.replay_batch
    if replay is not None:
        contexts.append(replay.forward_context.attn_metadata)
    snapshots = []
    resident_snapshots = []
    cache = validator.source.hisparse_cache
    for context in contexts:
        metadata = context.get(validator.source.prefix)
        if metadata is None:
            continue
        for row in range(metadata.num_reqs):
            state = int(cache.runtime.request_state_indices[row])
            if state < 0:
                continue
            host_blocks = metadata.block_table[row].to(torch.long)
            resident_blocks = cache.block_table[state].to(torch.long)
            count = min(host_blocks.numel(), resident_blocks.numel())
            present = (host_blocks[:count] > 0) & (resident_blocks[:count] > 0)
            offsets = torch.arange(validator.block_size, device="cuda")
            host_rows = (
                host_blocks[:count][present, None] * validator.block_size + offsets
            ).flatten()
            resident_rows = (
                resident_blocks[:count][present, None] * validator.block_size + offsets
            ).flatten()
            actual = records(cache.view.cache, resident_rows)
            expected = records(validator.shadow, host_rows)
            bad = (actual != expected).any(dim=1).nonzero().flatten()[:32]
            chosen = host_rows[bad].cpu()
            resident_snapshots.append(
                dict(
                    request_row=row,
                    host_rows=chosen,
                    actual=actual[bad].cpu(),
                    expected=expected[bad].cpu(),
                    host=records(validator.source.kv_cache, chosen),
                )
            )
        for key, plan in metadata.hisparse_prefill_plans.items():
            staged, _ = metadata.hisparse_prefill_caches[key]
            host_rows = plan.row_ids.reshape(-1)
            active = host_rows >= 0
            stage_rows = torch.arange(host_rows.numel(), device="cuda")[active]
            host_rows = host_rows[active].to(torch.long)
            actual = records(staged, stage_rows)
            expected = records(validator.shadow, host_rows)
            bad = (actual != expected).any(dim=1).nonzero().flatten()[:32]
            chosen = host_rows[bad].cpu()
            snapshots.append(
                dict(
                    key=key,
                    host_rows=chosen,
                    actual=actual[bad].cpu(),
                    expected=expected[bad].cpu(),
                    host=records(validator.source.kv_cache, chosen),
                    resident_rows=plan.gpu_row_ids.cpu(),
                    block_table=metadata.block_table.cpu(),
                )
            )
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            rank=worker.rank,
            source=validator.source_layer,
            stats=values,
            host_copies=validator.host_copy_checks,
            snapshots=snapshots,
            resident_snapshots=resident_snapshots,
            connector_metrics_calls=get_kv_transfer_group().connector_worker._metrics_calls,
        ),
        output / f"source{validator.source_layer}-rank{worker.rank}.pt",
    )


def install_validators(worker):
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
        HiSparseConnectorWorker,
    )

    model = worker.model_runner.get_model()
    sources = sorted(
        m.layer_id
        for m in model.modules()
        if type(m).__name__ == "DeepseekV4FlashMLAAttention"
        and m.hisparse_cache is not None
    )
    worker.graph_validators = [GraphValidator(model, source) for source in sources]
    original_copy = HiSparseConnectorWorker._copy_host_blocks

    def copy_blocks(connector, copies, previous_write):
        result = original_copy(connector, copies, previous_write)
        if copies:
            indices = torch.tensor(
                [(copy.src_block_id, copy.dst_block_id) for copy in copies],
                device="cuda",
            )
            src, dst = indices.unbind(dim=1)
            for validator in worker.graph_validators:
                validator.shadow[dst] = validator.shadow[src]
                validator.host_copy_checks += len(copies)
        return result

    HiSparseConnectorWorker._copy_host_blocks = copy_blocks


def _read_validator(worker, validator):
    stats = validator.stats
    values = stats.cpu().tolist()
    assert all(values[i] > 0 for i in range(3)), values
    assert not any(values[7:10]) and not values[12], (validator.source_layer, values)
    return dict(
        rank=worker.rank,
        passed=True,
        graph_checks=True,
        source_layer=validator.source_layer,
        write_checks=values[0],
        selected_byte_checks=values[1],
        attention_checks=values[2],
        host_miss_rows=values[3],
        checked_bytes=values[4:7],
        bad_bytes=values[7:10],
        prefill_staging_checks=values[10],
        prefill_checked_bytes=values[11],
        prefill_bad_bytes=values[12],
        prefill_reuse_checks=validator.prefill_reuse_checks,
        prefill_attention_checks=validator.prefill_attention_checks,
        host_copy_checks=validator.host_copy_checks,
    )


def control(worker, action):
    validators = worker.graph_validators
    if action == "reset":
        worker.model_runner.validation_active = True
        for validator in validators:
            validator.stats.zero_()
            validator.prefill_reuse_checks = 0
            validator.prefill_attention_checks = 0
            validator.host_copy_checks = 0
        torch.accelerator.synchronize()
        return dict(rank=worker.rank, reset=True)
    assert action == "read"
    records = [_read_validator(worker, v) for v in validators]
    result = dict(records[0])
    result.pop("source_layer")
    result["sources"] = {r["source_layer"]: r for r in records}
    for key in (
        "write_checks",
        "selected_byte_checks",
        "attention_checks",
        "host_miss_rows",
        "prefill_staging_checks",
        "prefill_bad_bytes",
        "prefill_reuse_checks",
        "prefill_attention_checks",
    ):
        result[key] = sum(r[key] for r in records)
    result["bad_bytes"] = [sum(r["bad_bytes"][i] for r in records) for i in range(3)]
    return result


def check_comparators():
    a_raw = torch.zeros(2 * 79488, dtype=torch.uint8, device="cuda")
    b_raw = torch.zeros(2 * 37440, dtype=torch.uint8, device="cuda")
    a = a_raw.as_strided((2, 64, 584), (79488, 584, 1))
    b = b_raw.as_strided((2, 64, 584), (37440, 584, 1))
    slots = torch.tensor([0, 63, 64, -1], dtype=torch.int64, device="cuda")
    stats = torch.zeros(10, dtype=torch.int64, device="cuda")
    compare_rows(a, b, slots, slots, stats, 4, 7)
    assert stats.cpu().tolist()[4:8:3] == [3 * 584, 0]
    b_raw[0] = 1
    b_raw[64 * 576 + 63 * 8 + 7] = 2
    compare_rows(a, b, slots, slots, stats, 4, 7)
    values = stats.cpu().tolist()
    assert values[4] == 6 * 584 and values[7] == 2, values
    other_slots = slots.clone()
    other_slots[-1] = 1
    compare_rows(a, b, slots, other_slots, stats, 4, 7)
    assert stats.cpu().tolist()[7] == 2 + 2 + 584
    aa = torch.zeros(123, dtype=torch.uint8, device="cuda")
    bb = aa.clone()
    bb[0], bb[-1] = 1, 2
    _compare_bytes[(triton.cdiv(aa.numel(), 1024),)](
        aa, bb, stats, N=aa.numel(), BLOCK=1024
    )
    assert stats.cpu().tolist()[9] == 2
    return dict(
        passed=True,
        injected_data_errors=2,
        invalid_row_bytes=584,
        injected_output_errors=2,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(check_comparators()))
