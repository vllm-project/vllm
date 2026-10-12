# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA regression for exact lengths and non-causal FlashInfer XQA metadata."""

import json
import math
from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends import flashinfer as fi
from vllm.v1.kv_cache_interface import FullAttentionSpec


class CopyAudit(TorchDispatchMode):
    def __init__(self, lengths):
        super().__init__()
        self.ptr = lengths.data_ptr()
        self.events = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func == torch.ops.aten._to_copy.default
            and args
            and isinstance(args[0], torch.Tensor)
        ):
            dev = kwargs.get("device")
            if args[0].is_cuda and dev is not None and torch.device(dev).type == "cpu":
                self.events.append(
                    {
                        "shape": list(args[0].shape),
                        "is_seq_lens": args[0].data_ptr() == self.ptr,
                    }
                )
        return func(*args, **kwargs)


def make_config(path, d, use_native):
    path.mkdir(exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 8 * d,
                "intermediate_size": 2048,
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": d,
                "vocab_size": 128,
                "max_position_embeddings": 65536,
                "torch_dtype": "bfloat16",
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-5,
            }
        )
    )
    mc = ModelConfig(
        model=str(path),
        tokenizer=str(path),
        dtype="bfloat16",
        seed=99173,
        max_model_len=65536,
    )
    cc = CacheConfig(block_size=16, cache_dtype="auto")
    cc.num_gpu_blocks = 65536
    cc.num_cpu_blocks = 0
    cc.kv_cache_layout = "LBHNC"
    vc = VllmConfig(
        model_config=mc,
        cache_config=cc,
        parallel_config=ParallelConfig(),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=1024,
            max_model_len=65536,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig(),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )
    vc.attention_config.use_trtllm_attention = False if use_native else None
    vc.speculative_config = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    return vc


def make_case(seq, qlens, causal, d, layout, upper_extra):
    dev = "cuda"
    hq, hk, page = 8, 2, 16
    nr = len(seq)
    nb = [(x + page - 1) // page for x in seq]
    nblocks = 1 + sum(nb)
    logical = (1, nblocks, hk, page, 2 * d)
    physical = tuple(logical[i] for i in layout.stride_order)
    inv = [layout.stride_order.index(i) for i in range(5)]
    # Poison padding so a CPU upper bound cannot silently replace exact lengths.
    kv = torch.full(physical, 9.0, dtype=torch.bfloat16, device=dev).permute(*inv)[0]
    table = torch.zeros(nr, max(nb), dtype=torch.int32, device=dev)
    slots = []
    keys = []
    values = []
    qs = []
    refs = []
    offset = 1
    for i, (s, q, n) in enumerate(zip(seq, qlens, nb)):
        k = torch.randn(s, hk, d, device=dev, dtype=torch.bfloat16) * 0.2
        v = torch.randn_like(k) * 0.2
        x = torch.randn(q, hq, d, device=dev, dtype=torch.bfloat16) * 0.2
        t = torch.arange(s, device=dev)
        kv[offset + t // page, :, t % page, :d] = k
        kv[offset + t // page, :, t % page, d:] = v
        table[i, :n] = torch.arange(offset, offset + n, dtype=torch.int32, device=dev)
        current = torch.arange(s - q, s, device=dev)
        slots.append((offset + current // page) * page + current % page)
        keys.append(k[s - q :])
        values.append(v[s - q :])
        qs.append(x)
        if q:
            kk = k.float().repeat_interleave(hq // hk, dim=1)
            vv = v.float().repeat_interleave(hq // hk, dim=1)
            scores = torch.einsum("qhd,khd->hqk", x.float(), kk) / math.sqrt(d)
            if causal:
                visible = (
                    torch.arange(s, device=dev)[None, :]
                    <= (s - q + torch.arange(q, device=dev))[:, None]
                )
                scores = scores.masked_fill(~visible[None], float("-inf"))
            refs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv))
        offset += n
    qcpu = torch.tensor(
        [0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32
    )
    exact = torch.tensor(seq, dtype=torch.int32, device=dev)
    upper = torch.tensor(seq, dtype=torch.int32) + torch.tensor(
        upper_extra, dtype=torch.int32
    )
    meta = CommonAttentionMetadata(
        query_start_loc=qcpu.to(dev),
        query_start_loc_cpu=qcpu,
        seq_lens=exact,
        seq_lens_cpu_upper_bound=upper,
        num_reqs=nr,
        num_actual_tokens=sum(qlens),
        max_query_len=max(qlens),
        max_seq_len=int(upper.max()),
        block_table_tensor=table,
        slot_mapping=torch.cat(slots).long(),
        causal=causal,
    )
    return meta, kv, torch.cat(qs), torch.cat(keys), torch.cat(values), torch.cat(refs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("query_lens", [(1, 1), (4, 4), (4, 64)])
def test_flashinfer_xqa_reads_exact_gpu_lengths(tmp_path, head_dim, causal, query_lens):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("dedicated SM120 XQA regression")
    torch.manual_seed(99173)
    config = make_config(tmp_path / "model", head_dim, False)
    with set_current_vllm_config(config):
        layer = Attention(
            8,
            head_dim,
            1 / math.sqrt(head_dim),
            num_kv_heads=2,
            cache_config=config.cache_config,
            prefix="layer.0",
            attn_backend=fi.FlashInferBackend,
        )
        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=head_dim, dtype=torch.bfloat16
        )
        builder = fi.FlashInferMetadataBuilder(
            spec, ["layer.0"], config, torch.device("cuda:0")
        )
        uses_xqa = getattr(
            builder, "use_xqa", getattr(builder, "use_dedicated_xqa", False)
        )
        if not uses_xqa:
            pytest.skip("XQA unavailable: native fallback is not changed-path coverage")
        common, kv, q, k, v, reference = make_case(
            [33, 257],
            list(query_lens),
            causal,
            head_dim,
            builder.kv_cache_layout,
            [3, 0],
        )
        original_lengths = common.seq_lens.clone()
        original_upper = common.seq_lens_cpu_upper_bound.clone()
        with CopyAudit(common.seq_lens) as audit:
            metadata = builder.build(0, common)
        length_copies = sum(event["is_seq_lens"] for event in audit.events)
        assert length_copies == int(metadata.num_prefills > 0)
        actual = torch.empty_like(q)
        layer.impl.forward(layer, q, k, v, kv, metadata, output=actual)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), reference, rtol=0.015, atol=0.003)
        assert torch.equal(common.seq_lens, original_lengths)
        assert torch.equal(common.seq_lens_cpu_upper_bound, original_upper)

        # Negative control: optimistic speculative lengths expose poisoned KV.
        # The original request must not silently attend to those slots.
        incorrect = replace(common, seq_lens=original_upper.to(q.device))
        bad_metadata = builder.build(0, incorrect)
        bad_output = torch.empty_like(q)
        layer.impl.forward(layer, q, k, v, kv, bad_metadata, output=bad_output)
        torch.cuda.synchronize()
        assert (
            bad_output[: query_lens[0]].float() - reference[: query_lens[0]]
        ).abs().max() > 0.05
