# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation-only native GPU shadow; never install during serving timings."""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import MethodType

import torch

ACTIVE = None


def records(cache, slots):
    block = cache.shape[1]
    values = cache.as_strided(
        (cache.shape[0], block, 576),
        (cache.stride(0), 576, 1),
        storage_offset=cache.storage_offset(),
    )
    scales = cache.as_strided(
        (cache.shape[0], block, 8),
        (cache.stride(0), 8, 1),
        storage_offset=cache.storage_offset() + block * 576,
    )
    return torch.cat(
        (values[slots // block, slots % block], scales[slots // block, slots % block]),
        dim=-1,
    )


class Validator:
    def __init__(self, model, max_tokens, concurrency, output):
        from vllm.models.deepseek_v41 import compressor

        self.layers = {
            m.layer_id: m
            for m in model.modules()
            if type(m).__name__ == "DeepseekV4FlashMLAAttention"
        }
        self.source = self.layers[20]
        self.output = Path(output)
        assert self.source.hisparse_cache is not None
        self.writing_source = False
        self.original_cache_insert = self.source.compressor.insert_cache

        def source_insert(compressor, *args, **kwargs):
            self.writing_source = True
            try:
                return self.original_cache_insert(*args, **kwargs)
            finally:
                self.writing_source = False

        self.source.compressor.insert_cache = MethodType(
            source_insert, self.source.compressor
        )
        self.compressor = compressor
        self.original_insert = compressor.rope_quant_insert
        self.original_decode = {}
        self.block = 64
        pages = 1 + concurrency * math.ceil(max_tokens / self.block)
        stride = math.ceil(self.block * 584 / 576) * 576
        self.shadow = torch.zeros(
            pages * stride, dtype=torch.uint8, device="cuda"
        ).as_strided((pages, self.block, 584), (stride, 584, 1))
        host_blocks = self.source.kv_cache.shape[0]
        self.mapping = torch.full((host_blocks,), -1, dtype=torch.int32, device="cuda")
        self.mapping[0] = 0
        self.pages = {0: 0}
        self.write_checks = 0
        self.written_rows = 0
        self.attention_checks = 0
        self.selected_checks = 0
        self.host_miss_rows = 0

        def insert(latent, positions, rope, cache, slots, ratio, **kwargs):
            result = self.original_insert(
                latent, positions, rope, cache, slots, ratio, **kwargs
            )
            resident = self.source.hisparse_cache.view.cache
            if not self.writing_source:
                return result
            assert cache.data_ptr() == resident.data_ptr() and ratio == 1
            from vllm.forward_context import get_forward_context

            metadata = get_forward_context().attn_metadata[self.source.prefix]
            original_slots = metadata.slot_mapping
            active = original_slots >= 0
            assert torch.equal(active, slots >= 0)
            ids = torch.unique(original_slots[active] // self.block).cpu().tolist()
            new = [i for i in ids if i not in self.pages]
            if new:
                first = len(self.pages)
                assert first + len(new) <= self.shadow.shape[0]
                for index, page in enumerate(new, first):
                    self.pages[page] = index
                self.mapping[torch.tensor(new, dtype=torch.long, device="cuda")] = (
                    torch.arange(
                        first, first + len(new), dtype=torch.int32, device="cuda"
                    )
                )
            translated = torch.where(
                active,
                self.mapping[original_slots.clamp_min(0) // self.block].to(torch.int64)
                * self.block
                + original_slots.clamp_min(0) % self.block,
                -1,
            )
            self.original_insert(
                latent, positions, rope, self.shadow, translated, ratio, **kwargs
            )
            assert torch.equal(
                records(cache, slots[active]), records(self.shadow, translated[active])
            )
            self.write_checks += 1
            self.written_rows += int(active.sum())
            return result

        compressor.rope_quant_insert = insert
        for layer_id in range(20, 40):
            layer = self.layers[layer_id]
            original = layer._forward_decode
            self.original_decode[layer_id] = original

            def wrap(original):
                def decode(attn, **kwargs):
                    result = original(**kwargs)
                    cache = self.source.hisparse_cache
                    metadata = kwargs["attn_metadata"]
                    swa = kwargs["swa_metadata"]
                    output = kwargs["output"]
                    actual = output.clone()
                    table = self.mapping[metadata.block_table.to(torch.long)]
                    from vllm.models.deepseek_v41.nvidia.flashmla import (
                        compute_global_topk_indices_and_lens,
                    )

                    reference_indices, _ = compute_global_topk_indices_and_lens(
                        attn.topk_indices_buffer[: swa.num_decode_tokens],
                        swa.token_to_req_indices,
                        table[: swa.num_decodes],
                        self.block,
                        swa.is_valid_token[: swa.num_decode_tokens],
                    )
                    hot_indices, _ = self.source._hisparse_topk
                    valid = reference_indices >= 0
                    assert torch.equal(hot_indices >= 0, valid)
                    actual_rows = records(
                        cache.runtime.hot.attention_cache,
                        hot_indices[valid].to(torch.long),
                    )
                    expected_rows = records(
                        self.shadow, reference_indices[valid].to(torch.long)
                    )
                    if not torch.equal(actual_rows, expected_rows):
                        from vllm.distributed import get_tp_group

                        host_indices, _ = compute_global_topk_indices_and_lens(
                            attn.topk_indices_buffer[: swa.num_decode_tokens],
                            swa.token_to_req_indices,
                            metadata.block_table[: swa.num_decodes],
                            self.block,
                            swa.is_valid_token[: swa.num_decode_tokens],
                        )
                        cold_rows = records(
                            self.source.kv_cache,
                            host_indices[valid].to("cpu", torch.long),
                        )
                        rank = get_tp_group().rank_in_group
                        torch.save(
                            dict(
                                layer=attn.layer_id,
                                selected_checks=self.selected_checks,
                                actual=actual_rows.cpu(),
                                expected=expected_rows.cpu(),
                                host=cold_rows,
                                hot_indices=hot_indices.cpu(),
                                reference_indices=reference_indices.cpu(),
                                host_indices=host_indices.cpu(),
                                logical_indices=attn.topk_indices_buffer[
                                    : swa.num_decode_tokens
                                ].cpu(),
                                resident_table=cache.block_table.cpu(),
                                swap_host=cache.runtime.index_group.shared_topk.swap_host_physical_rows.cpu(),
                                swap_device=cache.runtime.index_group.shared_topk.swap_device_physical_rows.cpu(),
                                swap_counts=cache.runtime.index_group.shared_topk.swap_counts.cpu(),
                            ),
                            self.output / f"mismatch-rank{rank:02d}.pt",
                        )
                        raise AssertionError(
                            f"Selected KV mismatch: layer={attn.layer_id}, "
                            f"passed checks={self.selected_checks}"
                        )
                    self.selected_checks += 1
                    if attn.is_index_source:
                        self.host_miss_rows += int(
                            cache.runtime.index_group.shared_topk.swap_counts.sum()
                        )
                    try:
                        self.source.hisparse_cache = None
                        reference = dict(
                            kwargs,
                            kv_cache=self.shadow,
                            attn_metadata=replace(metadata, block_table=table),
                        )
                        original(**reference)
                        assert torch.equal(
                            output.view(torch.uint8), actual.view(torch.uint8)
                        )
                        self.attention_checks += 1
                    finally:
                        self.source.hisparse_cache = cache
                        output.copy_(actual)
                    return result

                return decode

            layer._forward_decode = MethodType(wrap(original), layer)

    def finish(self):
        assert self.write_checks and self.attention_checks and self.selected_checks
        self.compressor.rope_quant_insert = self.original_insert
        self.source.compressor.insert_cache = self.original_cache_insert
        for i, original in self.original_decode.items():
            self.layers[i]._forward_decode = original
        return dict(
            passed=True,
            write_checks=self.write_checks,
            written_rows=self.written_rows,
            attention_checks=self.attention_checks,
            selected_byte_checks=self.selected_checks,
            reference_pages=len(self.pages) - 1,
            host_miss_rows=self.host_miss_rows,
        )


def control(model, action, output, max_tokens=0, concurrency=0):
    global ACTIVE
    from vllm.distributed import get_tp_group

    rank = get_tp_group().rank_in_group
    if action == "install":
        assert ACTIVE is None
        ACTIVE = Validator(model, max_tokens, concurrency, output)
        return dict(rank=rank, installed=True)
    assert action == "finish" and ACTIVE is not None
    result = dict(rank=rank, **ACTIVE.finish())
    ACTIVE = None
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"validation-rank{rank:02d}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def inventory(model, reset=False):
    import torch

    from vllm.distributed import get_tp_group

    if reset:
        torch.accelerator.reset_peak_memory_stats()
    torch.accelerator.synchronize()
    owners = {}
    indexers = {}
    for layer in model.modules():
        if type(layer).__name__ == "DeepseekV4IndexerCache":
            tensor = layer.kv_cache
            indexers[layer.prefix] = dict(
                device=str(tensor.device),
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
            )
        if (
            type(layer).__name__ != "DeepseekV4FlashMLAAttention"
            or not layer.is_kv_source
        ):
            continue
        tensor = layer.kv_cache
        owners[layer.layer_id] = dict(
            device=str(tensor.device),
            shape=list(tensor.shape),
            stride=list(tensor.stride()),
            storage_bytes=tensor.untyped_storage().nbytes(),
            hisparse=layer.hisparse_cache is not None,
        )
    return dict(
        rank=get_tp_group().rank_in_group,
        owners=owners,
        indexers=indexers,
        allocated_bytes=torch.accelerator.memory_allocated(),
        peak_allocated_bytes=torch.accelerator.max_memory_allocated(),
    )
