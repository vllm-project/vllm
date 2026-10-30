# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: E402
# Select the isolated runtime before importing vLLM.
import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import warnings
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--case", required=True)
p.add_argument("--repeat", type=int, default=30)
p.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parents[3])
p.add_argument("--expected-native-sha")
args = p.parse_args()
ROOT = args.runtime.resolve()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT))
import torch
import vllm._C_stable_libtorch as native
from chunk_observer import ChunkObserver
from fixture import make_inputs
from flashinfer.testing import bench_gpu_time_with_cupti
from kernel_trace import cupti_kernel_details
from page_repack import align_cache_pages, capacity_block_table, repack_page128
from scale_profile import apply_scale_profile

import vllm.model_executor.kernels.attention.dsa.candidate_blocks as cb
import vllm.model_executor.layers.sparse_attn_indexer as m
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.worker.workspace import init_workspace_manager


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    p.write_text(json.dumps(x, indent=2, default=str) + "\n")


def digest(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


if args.expected_native_sha:
    assert sha(native.__file__) == args.expected_native_sha
assert (
    Path(m.__file__).resolve()
    == ROOT / "vllm/model_executor/layers/sparse_attn_indexer.py"
)
assert torch.accelerator.device_count() == 1 and torch.cuda.get_device_capability() == (
    10,
    0,
)
warnings.filterwarnings("error", message="(?i).*falling back to cuda events.*")
torch.set_num_threads(8)
init_workspace_manager(torch.device("cuda", 0))
cases = {
    "smoke": dict(
        rows=96,
        width=1048576,
        heads=32,
        compression=1,
        context=4096,
        q_len=1,
        candidate=False,
        page=128,
        budgets=[0, 128],
    ),
    "c1_short": dict(
        rows=8192,
        width=1048576,
        heads=32,
        compression=1,
        context=8192,
        q_len=8,
        candidate=False,
        page=128,
    ),
    "c1_source": dict(
        rows=8192,
        width=1048576,
        heads=32,
        compression=1,
        context=32768,
        q_len=8,
        candidate=True,
        page=128,
    ),
    "c2": dict(
        rows=4096,
        width=524288,
        heads=32,
        compression=2,
        context=32768,
        q_len=8,
        candidate=False,
        page=64,
    ),
    "c4": dict(
        rows=2048,
        width=262144,
        heads=64,
        compression=4,
        context=65536,
        q_len=8,
        candidate=False,
        page=64,
    ),
    "tail_source": dict(
        rows=1025,
        width=1048576,
        heads=32,
        compression=1,
        context=32768,
        q_len=8,
        candidate=True,
        page=128,
    ),
}
c = cases[args.case]
R = c["rows"]
L = c["context"] // c["compression"]
K = 512
x = make_inputs(
    R, c["heads"], c["width"], c["compression"], c["context"], "mxfp4", c["q_len"], K
)
if c["page"] == 128:
    x = repack_page128(
        torch,
        x,
        c["q_len"],
        m.kv_cache_as_quant_view,
        get_paged_mqa_logits_metadata,
        get_num_sms(),
    )
x = align_cache_pages(torch, x, m.kv_cache_as_quant_view)
x = capacity_block_table(torch, x, c["width"])
apply_scale_profile(torch, x, "powers")
if args.case == "smoke":
    lens = torch.tensor(
        [0, 128, 512, 1024, 4095, 4096] * 16, device="cuda", dtype=torch.int32
    )[:, None]
    x["ctx"].copy_(lens)
    x["schedule"].copy_(
        get_paged_mqa_logits_metadata(
            x["ctx"], c["page"], get_num_sms(), indices=x["indices"]
        )
    )
x["candidates"] = (
    torch.full((R, 2048), -1, device="cuda", dtype=torch.int32)
    if c["candidate"]
    else None
)
prefix = "probe.indexer"
decode = DeepSeekV32IndexerDecodeMetadata(
    block_table=x["bt"],
    seq_lens=x["ctx"],
    decode_lens=torch.ones(R, device="cuda", dtype=torch.int32),
    requires_padding=False,
    schedule_metadata=x["schedule"],
    indices=x["indices"],
)
metadata = DeepseekV32IndexerMetadata(
    seq_lens=x["ctx"],
    max_seq_len=c["width"],
    slot_mapping=torch.zeros(R, device="cuda", dtype=torch.int64),
    num_decodes=math.ceil(R / c["q_len"]),
    num_decode_tokens=R,
    num_prefills=0,
    num_prefill_tokens=0,
    decode=decode,
)
context = ForwardContext(
    no_compile_layers={},
    attn_metadata={prefix: metadata},
    slot_mapping={prefix: metadata.slot_mapping},
)
hidden = torch.empty((R, 1), device="cuda", dtype=torch.bfloat16)


def invoke(mode=CUDAGraphMode.NONE):
    context.cudagraph_runtime_mode = mode
    with override_forward_context(context):
        return m.sparse_attn_indexer(
            hidden,
            prefix,
            x["kv"].squeeze(2),
            x["q"].reshape(R, c["heads"], 64),
            x["qs"].reshape(R, c["heads"]),
            None,
            x["weights"],
            128,
            "ue8m0",
            K,
            128,
            c["width"],
            R * L,
            x["out"],
            True,
            False,
            False,
            "",
            use_fp4_cache=True,
            candidate_blocks=x["candidates"],
            candidate_block_size=8 if c["candidate"] else 0,
            candidate_write=c["candidate"],
            topk_backend="auto",
        )


def gate(words, ends, ids, bits, k):
    h = hashlib.sha256()
    for i, n in enumerate(ends.tolist()):
        count = min(n, k)
        ii = ids[i][ids[i] >= 0].long()
        assert (
            len(ii) == count and len(ii.unique()) == count and bool((ii < n).all())
        ), (args.case, i, n, ii)
        assert bool((ids[i][ids[i] < 0] == -1).all())
        raw = words[i, :n].long() & ((1 << bits) - 1)
        keys = torch.where(
            raw & (1 << (bits - 1)) != 0,
            (~raw) & ((1 << bits) - 1),
            raw ^ (1 << (bits - 1)),
        )
        got = keys[ii].sort().values
        ref = keys.topk(count).values.sort().values
        assert torch.equal(got, ref), (args.case, i, "topk mismatch")
        h.update(got.numpy().tobytes())
    return h.hexdigest()


reference = None
records = []
budgets = c.get("budgets", [0, 512, 1024, 2048, 4096, 8192])
ends = x["ctx"].view(-1).cpu()
mask = torch.arange(L)[None, :] < ends[:, None]
ce = (ends + 7) // 8
cmask = torch.arange((L + 7) // 8)[None, :] < ce[:, None]
save(
    OUT / "IDENTITY.json",
    dict(
        case=c,
        native_sha=sha(native.__file__),
        caller_sha=sha(m.__file__),
        GPU=str(torch.cuda.get_device_properties(0)),
        python=sys.executable,
        measurement="real prepared MXFP4 Q/K, actual caller, auto, cold CUPTI graph",
        native_maps=[
            z
            for z in Path("/proc/self/maps").read_text().splitlines()
            if native.__file__ in z
        ],
    ),
)


def numeric(budget):
    global reference
    with ChunkObserver(
        torch, m, cb, SparseIndexerTopk, R, c["width"], L, c["candidate"]
    ) as obs:
        invoke()
        assert obs.upcast_calls == 0
        words = obs.buffers["producer"].cpu()
        values = words[mask]
        candidate = obs.buffers["candidate"].cpu() if c["candidate"] else None
        if reference is None:
            reference = dict(
                words=words.clone(),
                producer_bits=digest(values),
                candidate=candidate.clone() if candidate is not None else None,
            )
        assert digest(values) == reference["producer_bits"], ("producer bits", budget)
        if candidate is not None:
            assert torch.equal(candidate[cmask], reference["candidate"][cmask]), (
                "candidate bits",
                budget,
            )
        tokenhash = gate(words, ends, x["out"].cpu(), 16, K)
        candidatehash = (
            gate(candidate, ce, x["candidates"].cpu(), 32, 2048)
            if c["candidate"]
            else None
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            invoke(CUDAGraphMode.FULL)
        host_count = len(obs.records)
        obs.poison()
        x["out"].fill_(-999)
        if c["candidate"]:
            x["candidates"].fill_(-999)
        graph.replay()
        torch.accelerator.synchronize()
        assert len(obs.records) == host_count
        assert digest(obs.buffers["producer"].cpu()[mask]) == reference["producer_bits"]
        assert (
            gate(obs.buffers["producer"].cpu(), ends, x["out"].cpu(), 16, K)
            == tokenhash
        )
        if c["candidate"]:
            cc = obs.buffers["candidate"].cpu()
            assert torch.equal(cc[cmask], reference["candidate"][cmask])
            assert gate(cc, ce, x["candidates"].cpu(), 32, 2048) == candidatehash
        dynamic = False
        if args.case == "smoke":
            shorter = (ends - 1024).clamp_min(0)
            x["ctx"].copy_(shorter[:, None])
            x["schedule"].copy_(
                get_paged_mqa_logits_metadata(
                    x["ctx"], c["page"], get_num_sms(), indices=x["indices"]
                )
            )
            obs.poison()
            graph.replay()
            torch.accelerator.synchronize()
            smask = torch.arange(L)[None, :] < shorter[:, None]
            assert torch.equal(
                obs.buffers["producer"].cpu()[smask], reference["words"][smask]
            )
            gate(obs.buffers["producer"].cpu(), shorter, x["out"].cpu(), 16, K)
            x["ctx"].copy_(ends[:, None])
            x["schedule"].copy_(
                get_paged_mqa_logits_metadata(
                    x["ctx"], c["page"], get_num_sms(), indices=x["indices"]
                )
            )
            obs.poison()
            graph.replay()
            torch.accelerator.synchronize()
            assert (
                gate(obs.buffers["producer"].cpu(), ends, x["out"].cpu(), 16, K)
                == tokenhash
            )
            dynamic = True
        proof = dict(
            producer_bits=reference["producer_bits"],
            token_score_bag=tokenhash,
            candidate_score_bag=candidatehash,
            replay_host_unchanged=True,
            dynamic_shrink_grow=dynamic,
            upcasts=obs.upcast_calls,
            trace=obs.records,
        )
        save(OUT / f"proof_{budget}.json", proof)
        del graph
    del obs
    gc.collect()
    torch.accelerator.empty_cache()
    return proof


for cycle, order in enumerate([budgets, list(reversed(budgets))]):
    for budget in order:
        os.environ["VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB"] = str(budget)
        if cycle == 0:
            numeric(budget)
        for _ in range(3):
            invoke()
        torch.accelerator.synchronize()
        gc.collect()
        torch.accelerator.empty_cache()
        base = torch.accelerator.memory_allocated()
        torch.accelerator.reset_peak_memory_stats()
        invoke()
        torch.accelerator.synchronize()
        peak = torch.accelerator.max_memory_allocated() - base
        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            invoke(CUDAGraphMode.FULL)
        graph.replay()
        torch.accelerator.synchronize()
        segments = [
            s
            for s in torch.cuda.memory_snapshot()
            if tuple(s.get("segment_pool_id", s.get("owner_private_pool_id", (0, 0))))
            == tuple(pool)
        ]
        samples = [
            float(v) * 1000
            for v in bench_gpu_time_with_cupti(
                graph.replay,
                use_cuda_graph=False,
                cold_l2_cache=True,
                dry_run_iters=2,
                repeat_iters=args.repeat,
            )
        ]
        assert len(samples) == args.repeat
        gate(reference["words"], ends, x["out"].cpu(), 16, K)
        if c["candidate"]:
            gate(reference["candidate"], ce, x["candidates"].cpu(), 32, 2048)
        detail = cupti_kernel_details(graph.replay, 2, None) if cycle == 0 else None
        ranges = m._decode_chunk_ranges(R, c["width"], budget * 2**20)
        profile_rows = min(R, budget * 2**20 // (2 * c["width"])) if budget else R
        record = dict(
            case=args.case,
            cycle=cycle,
            budget_MiB=budget,
            chunks=ranges or [(0, R)],
            samples_us=samples,
            median_us=statistics.median(samples),
            extra_live_peak_bytes=peak,
            graph_reserved_bytes=sum(s["total_size"] for s in segments),
            profile_scratch_bound_bytes=m._decode_chunk_profile_bytes(
                profile_rows,
                c["width"],
                8 if c["candidate"] else 0,
                c["candidate"],
                2048,
            ),
            breakdown=detail,
        )
        records.append(record)
        save(OUT / "RESULT.json", dict(status="partial", records=records))
        print(
            "DONE",
            args.case,
            cycle,
            budget,
            record["median_us"],
            peak / 2**20,
            flush=True,
        )
        del graph
        gc.collect()
        torch.accelerator.empty_cache()
save(OUT / "RESULT.json", dict(status="complete", records=records))
