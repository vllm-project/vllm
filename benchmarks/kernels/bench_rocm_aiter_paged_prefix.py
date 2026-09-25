# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time the 1M prefill context attention on one MI355X rank.

This compares three ways to attend a 32k token extend step to a long
cached prefix. The shapes are one TP8 rank of Qwen3.8-2.4T: 8 query heads,
1 KV head, head size 256, FP8 e4m3 KV cache with 32-token kernel blocks
inside 2112-token attention blocks.

- loop: the current extend_forward. It gathers 32k context tokens at a
  time into a workspace, runs dense FMHA on each chunk and merges every
  chunk output.
- paged: extend_forward with VLLM_ROCM_AITER_PAGED_PREFIX. One aiter#4971
  paged FMHA call reads the whole prefix from the cache as 64-token pages,
  and one merge adds the causal result on the new tokens.
- one_call: one causal paged FMHA call over the prefix and the new tokens.
  The new K and V are written into the cache first. This is not wired into
  vLLM. It shows what a second step could gain.

It needs an MI355X and AITER v0.1.21.post2 or later, which includes the
aiter#4971 kernels. Pass --profile to print the kernel names, which shows
whether the ASM paged kernel ran or AITER fell back to CK.
"""

import argparse
import json
import math
import os
import time
from types import SimpleNamespace

# The environment must be set before vLLM reads it at import time.
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_MHA", "1")
os.environ.setdefault("VLLM_ROCM_FP8_DIRECT_CONTEXT_GATHER", "1")

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform
from vllm.v1.attention.backend import PrequantizedQKV
from vllm.v1.attention.backends import rocm_aiter_fa as fa

NUM_HEADS = 8
NUM_KV_HEADS = 1
HEAD_SIZE = 256
KERNEL_BLOCK = 32
BLOCKS_PER_ATTENTION_BLOCK = 66
BLOCKS_PER_PAGE = 2
SCALE = HEAD_SIZE**-0.5

# The inputs are stored in FP8 with these descales. They are not 1.0, so a
# kernel that drops or repeats a descale gives a large error against the
# FP32 reference instead of a result that happens to look right.
Q_DESCALE = 0.25
K_DESCALE = 0.5
V_DESCALE = 0.125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contexts", default="32768,131072,524288,983040")
    parser.add_argument("--q-len", type=int, default=32768)
    parser.add_argument("--cache-gib", type=float, default=10.0)
    parser.add_argument("--variants", default="loop,paged,one_call")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--ref-rows",
        type=int,
        default=64,
        help="Query rows checked against an FP32 reference. Use 0 to skip.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="Write results to this file.")
    return parser.parse_args()


def random_fp8(shape, stored_scale, device, fp8_dtype):
    """Return FP8 values that dequantize to about N(0, 1) with stored_scale."""
    return (torch.randn(shape, device=device) / stored_scale).to(fp8_dtype)


def make_cache(cache_gib, device, fp8_dtype):
    """Allocate the cache in the (blocks, heads, 32, 2 * 256) layout.

    This is the AITER FA cache layout when the shuffled KV cache layout is
    off. _split_kv_cache() splits it into K and V views of shape
    (blocks, 32, heads, 256), and so does this function. The number of
    blocks is rounded down to whole attention blocks.
    """
    block_bytes = NUM_KV_HEADS * KERNEL_BLOCK * 2 * HEAD_SIZE
    num_blocks = int(cache_gib * 2**30) // block_bytes
    num_blocks -= num_blocks % BLOCKS_PER_ATTENTION_BLOCK
    kv = torch.empty(
        (num_blocks, NUM_KV_HEADS, KERNEL_BLOCK, 2 * HEAD_SIZE),
        dtype=torch.uint8,
        device=device,
    )
    key_cache, value_cache = kv.transpose(1, 2).split(HEAD_SIZE, dim=-1)
    key_cache = key_cache.view(fp8_dtype)
    value_cache = value_cache.view(fp8_dtype)
    # Fill every block with finite values in slices to keep the temporary
    # buffers small. A wrong page index then gives a wrong but finite result,
    # and the comparisons below catch it.
    step = 8192
    for start in range(0, num_blocks, step):
        end = min(start + step, num_blocks)
        shape = (end - start, KERNEL_BLOCK, NUM_KV_HEADS, HEAD_SIZE)
        key_cache[start:end].copy_(random_fp8(shape, K_DESCALE, device, fp8_dtype))
        value_cache[start:end].copy_(random_fp8(shape, V_DESCALE, device, fp8_dtype))
    return kv, key_cache, value_cache


def make_block_table_row(num_blocks, num_tokens, generator, device):
    """Pick attention blocks for one request from the top half of the cache.

    The top half is above 4 GiB with the default 10 GiB cache, so the paged
    kernel must handle page offsets that do not fit in 32 bits. The last
    attention block of the cache is always used, which gives the largest
    possible offset. Kernel block ids follow the vLLM hybrid mapping, which
    is attention block id times 66 plus the index inside the block.
    """
    num_attention_blocks = num_blocks // BLOCKS_PER_ATTENTION_BLOCK
    needed = math.ceil(num_tokens / (BLOCKS_PER_ATTENTION_BLOCK * KERNEL_BLOCK))
    low = num_attention_blocks // 2
    if needed > num_attention_blocks - low:
        raise ValueError(
            f"{num_tokens} tokens need {needed} attention blocks, but the top "
            f"half of the cache has {num_attention_blocks - low}. "
            "Increase --cache-gib."
        )
    chosen = torch.randperm(num_attention_blocks - low, generator=generator)[:needed]
    chosen += low
    last = num_attention_blocks - 1
    if last not in chosen.tolist():
        chosen[0] = last
    kernel_ids = chosen[:, None] * BLOCKS_PER_ATTENTION_BLOCK + torch.arange(
        BLOCKS_PER_ATTENTION_BLOCK
    )
    return kernel_ids.reshape(-1).to(torch.int32).to(device)


def chunk_context_metadata(context_len, workspace, device):
    """Build the gather loop metadata for one extend request.

    This follows AiterFlashAttentionMetadataBuilder.build() with one extend
    request, so the loop variant times the same chunks as the server.
    """
    num_extends = 1
    computed_kv_lens = torch.tensor([context_len], dtype=torch.int32)
    max_context_chunk = fa._CP_TOKENS_PER_ITER_ROCM // num_extends
    num_chunks = fa.cdiv(context_len, max_context_chunk)
    chunk_starts = (
        torch.arange(num_chunks, dtype=torch.int32).unsqueeze(1).expand(-1, num_extends)
        * max_context_chunk
    )
    chunk_ends = torch.min(
        computed_kv_lens.unsqueeze(0), chunk_starts + max_context_chunk
    )
    chunk_seq_lens = (chunk_ends - chunk_starts).clamp_(min=0)
    cu_seq_lens_cpu = torch.zeros(
        [num_chunks, num_extends + 1], dtype=torch.int32, pin_memory=True
    )
    torch.cumsum(chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32)
    max_cum_tokens = cu_seq_lens_cpu[:, -1].max().item()
    range_idx = torch.arange(max_cum_tokens, dtype=torch.int32)[None, None, :]
    idx_to_batch = (range_idx == cu_seq_lens_cpu[:, 1:][:, :, None]).sum(dim=1)
    token_to_batch = torch.cumsum(idx_to_batch, dim=1)
    return fa.AiterChunkContextMetadata(
        workspace=workspace,
        cu_seq_lens_chunk=cu_seq_lens_cpu.to(device),
        chunk_starts=chunk_starts.to(device),
        token_to_batch=token_to_batch.to(device),
        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
        num_chunks=num_chunks,
        total_token_per_batch=cu_seq_lens_cpu[:, -1].tolist(),
        swa_metadata=None,
    )


def write_new_tokens(kv, block_table_row, context_len, key_new, value_new):
    """Write the new tokens' FP8 K and V into the cache after the prefix."""
    q_len = key_new.shape[0]
    positions = context_len + torch.arange(q_len, device=kv.device)
    blocks = block_table_row[positions // KERNEL_BLOCK].long()
    offsets = positions % KERNEL_BLOCK
    kv[blocks, 0, offsets, :HEAD_SIZE] = key_new[:, 0].view(torch.uint8)
    kv[blocks, 0, offsets, HEAD_SIZE:] = value_new[:, 0].view(torch.uint8)


def reference_rows(kv, block_table_row, context_len, prequantized, rows):
    """Compute exact attention for a few query rows in FP32.

    Each row attends to the whole prefix and to the new tokens up to and
    including itself. The prefix K and V come from the cache bytes, so this
    also checks that every variant read the right pages. FP8 tensors are
    indexed through their uint8 bytes because index kernels may not support
    FP8 dtypes.
    """
    fp8_dtype = prequantized.key.dtype
    positions = torch.arange(context_len, device=kv.device)
    blocks = block_table_row[positions // KERNEL_BLOCK].long()
    offsets = positions % KERNEL_BLOCK
    prefix = kv[blocks, 0, offsets]
    key_prefix = prefix[:, :HEAD_SIZE].view(fp8_dtype).float() * K_DESCALE
    value_prefix = prefix[:, HEAD_SIZE:].view(fp8_dtype).float() * V_DESCALE
    del prefix
    key_new = prequantized.key[:, 0].float() * K_DESCALE
    value_new = prequantized.value[:, 0].float() * V_DESCALE
    query_bytes = prequantized.query.view(torch.uint8)[rows]
    query = query_bytes.view(fp8_dtype).float() * Q_DESCALE
    new_positions = torch.arange(key_new.shape[0], device=kv.device)
    out = torch.empty(len(rows), NUM_HEADS, HEAD_SIZE, device=kv.device)
    # Process a few rows at a time. At a 983k prefix, one row needs about
    # 65 MB for its FP32 logits and probabilities over 8 heads.
    group = 16
    for start in range(0, len(rows), group):
        q = query[start : start + group]
        logits = torch.cat(
            [
                torch.einsum("rhd,td->rht", q, key_prefix),
                torch.einsum("rhd,td->rht", q, key_new),
            ],
            dim=-1,
        ).mul_(SCALE)
        # New token j is visible to query row i only if j <= i.
        hidden = new_positions[None, :] > rows[start : start + group, None]
        logits[:, :, context_len:].masked_fill_(hidden[:, None, :], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        out[start : start + group] = torch.einsum(
            "rht,td->rhd", probs[:, :, :context_len], value_prefix
        ) + torch.einsum("rht,td->rhd", probs[:, :, context_len:], value_new)
    return out


def errors(result, reference):
    diff = (result.float() - reference.float()).abs()
    relative = diff.norm() / reference.float().norm().clamp_min(1e-30)
    return {"max_abs": diff.max().item(), "rel_l2": relative.item()}


def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters


def attention_pflops(q_len, kv_len, ms):
    """Return PFLOPS for a dense q_len by kv_len attention with all heads."""
    flops = 4 * q_len * kv_len * HEAD_SIZE * NUM_HEADS
    return flops / (ms * 1e-3) / 1e15


def kernel_names(fn):
    """Run fn once under the profiler and return GPU kernel names and times.

    Only GPU events are kept. ROCm reports them with the CUDA device type.
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.accelerator.synchronize()
    names = {}
    for event in prof.events():
        if event.device_type != DeviceType.CUDA:
            continue
        count, ms = names.get(event.name, (0, 0.0))
        names[event.name] = (count + 1, ms + event.time_range.elapsed_us() / 1e3)
    return names


class Bench:
    """Hold the cache, the inputs and the backend shared by every context."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.q_len = args.q_len
        self.variants = args.variants.split(",")
        self.device = torch.device("cuda:0")
        torch.accelerator.set_device_index(self.device)
        self.generator = torch.Generator().manual_seed(args.seed)
        torch.manual_seed(args.seed)
        fp8_dtype = current_platform.fp8_dtype()
        device = self.device

        if rocm_aiter_ops.is_shuffle_kv_cache_enabled():
            raise SystemExit("The shuffled KV cache layout is on. The bench needs NHD.")
        self.mha_batch_prefill_func = fa._get_mha_batch_prefill_func()
        if self.mha_batch_prefill_func is None:
            raise SystemExit("aiter.ops.mha.mha_batch_prefill_func is missing.")

        self.impl = fa.AiterFlashAttentionImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            scale=SCALE,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
        )
        if not self.impl.supports_prequantized_qkv_input:
            raise SystemExit("The backend does not take prequantized QKV on this GPU.")

        started = time.time()
        self.kv, self.key_cache, self.value_cache = make_cache(
            args.cache_gib, device, fp8_dtype
        )
        print(
            f"cache: {self.kv.shape[0]} blocks of {KERNEL_BLOCK} tokens, "
            f"{self.kv.numel() / 2**30:.2f} GiB, "
            f"filled in {time.time() - started:.1f} s"
        )
        self.key_pages, self.value_pages = fa.paged_prefix_cache_views(
            self.key_cache, self.value_cache, BLOCKS_PER_PAGE
        )

        q_shape = (self.q_len, NUM_HEADS, HEAD_SIZE)
        kv_shape = (self.q_len, NUM_KV_HEADS, HEAD_SIZE)
        self.prequantized = PrequantizedQKV(
            query=random_fp8(q_shape, Q_DESCALE, device, fp8_dtype),
            key=random_fp8(kv_shape, K_DESCALE, device, fp8_dtype),
            value=random_fp8(kv_shape, V_DESCALE, device, fp8_dtype),
            query_descale=torch.full((1, NUM_KV_HEADS), Q_DESCALE, device=device),
            key_descale=torch.full((1, NUM_KV_HEADS), K_DESCALE, device=device),
            value_descale=torch.full((1, NUM_KV_HEADS), V_DESCALE, device=device),
        )
        # The layer scales are 0-dim tensors in vLLM. The new tokens use the
        # same values, so every variant sees identical inputs.
        self.k_scale = torch.tensor(K_DESCALE, dtype=torch.float32, device=device)
        self.v_scale = torch.tensor(V_DESCALE, dtype=torch.float32, device=device)
        # The workspace starts with finite values, so the dense chunk timing
        # does not depend on whether the loop variant ran first.
        chunk = fa._CP_TOKENS_PER_ITER_ROCM
        self.workspace = torch.stack(
            [
                random_fp8((chunk, NUM_KV_HEADS, HEAD_SIZE), scale, device, fp8_dtype)
                for scale in (K_DESCALE, V_DESCALE)
            ]
        )
        self.cu_seqlens_q = torch.tensor(
            [0, self.q_len], dtype=torch.int32, device=device
        )
        self.cu_seqlens_chunk = torch.tensor(
            [0, chunk], dtype=torch.int32, device=device
        )
        self.slot_mapping = torch.empty(0, dtype=torch.int64, device=device)

        self.causal_ms = time_ms(self.run_causal_new_tokens, args.warmup, args.iters)
        self.dense_chunk_ms = time_ms(self.run_dense_chunk, args.warmup, args.iters)
        self.dense_chunk_pflops = attention_pflops(
            self.q_len, chunk, self.dense_chunk_ms
        )
        print(
            f"causal FMHA on {self.q_len} new tokens: {self.causal_ms:.2f} ms. "
            f"Dense FMHA on one {chunk} token context chunk: "
            f"{self.dense_chunk_ms:.2f} ms, {self.dense_chunk_pflops:.2f} PFLOPS."
        )

    def run_causal_new_tokens(self) -> None:
        """Run the causal FMHA on the new tokens that every variant starts with.

        The arguments match the first call in extend_forward.
        """
        p = self.prequantized
        rocm_aiter_ops.flash_attn_varlen_func(
            q=p.query,
            k=p.key,
            v=p.value,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k=self.cu_seqlens_q,
            max_seqlen_q=self.q_len,
            max_seqlen_k=self.q_len,
            min_seqlen_q=self.q_len,
            dropout_p=0.0,
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            return_lse=True,
            sink_ptr=None,
            q_descale=p.query_descale,
            k_descale=p.key_descale,
            v_descale=p.value_descale,
        )

    def run_dense_chunk(self) -> None:
        """Run the dense FMHA that the loop runs on each 32k context chunk.

        The arguments match the chunk call in extend_forward.
        """
        rocm_aiter_ops.flash_attn_varlen_func(
            q=self.prequantized.query,
            k=self.workspace[0],
            v=self.workspace[1],
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k=self.cu_seqlens_chunk,
            max_seqlen_q=self.q_len,
            max_seqlen_k=fa._CP_TOKENS_PER_ITER_ROCM,
            min_seqlen_q=self.q_len,
            dropout_p=0.0,
            softmax_scale=SCALE,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            return_lse=True,
            sink_ptr=None,
            q_descale=self.prequantized.query_descale,
            k_descale=self.k_scale.expand(1, NUM_KV_HEADS),
            v_descale=self.v_scale.expand(1, NUM_KV_HEADS),
        )

    def run_context(self, context_len: int) -> dict:
        """Time and check every variant for one prefix length."""
        args = self.args
        q_len = self.q_len
        p = self.prequantized
        device = self.device
        total_len = context_len + q_len
        row = make_block_table_row(self.kv.shape[0], total_len, self.generator, device)
        block_table = row[None, :]
        chunk_meta = chunk_context_metadata(context_len, self.workspace, device)
        prefix_meta = fa.build_paged_prefix_metadata(
            row, context_len, BLOCKS_PER_PAGE, device
        )
        full_meta = fa.build_paged_prefix_metadata(
            row, total_len, BLOCKS_PER_PAGE, device
        )
        write_new_tokens(self.kv, row, context_len, p.key, p.value)
        outputs = {
            name: torch.empty(
                q_len, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=device
            )
            for name in self.variants
        }

        def run_extend(name, paged_prefix):
            attn_metadata = SimpleNamespace(
                extend_metadata=fa.AiterFlashAttentionChunkPrefillMetadata(
                    max_query_len=q_len,
                    max_seq_len=total_len,
                    query_start_loc=self.cu_seqlens_q,
                    chunk_context_metadata=chunk_meta,
                    paged_prefix=paged_prefix,
                )
            )
            self.impl.extend_forward(
                attn_metadata,
                p.query,
                p.key,
                p.value,
                self.key_cache,
                self.value_cache,
                outputs[name],
                self.cu_seqlens_q,
                q_len,
                q_len,
                total_len,
                block_table,
                self.slot_mapping,
                self.k_scale,
                self.v_scale,
                prequantized_qkv=p,
            )

        def run_paged_call(meta, kv_len, causal, out=None):
            return self.mha_batch_prefill_func(
                p.query,
                self.key_pages,
                self.value_pages,
                self.cu_seqlens_q,
                meta.kv_indptr,
                meta.kv_page_indices,
                q_len,
                kv_len,
                softmax_scale=SCALE,
                causal=causal,
                return_lse=not causal,
                out=out,
                kv_last_page_lens=meta.kv_last_page_lens,
                q_descale=p.query_descale.reshape(1),
                k_descale=self.k_scale.reshape(1),
                v_descale=self.v_scale.reshape(1),
            )

        runners = {
            "loop": lambda: run_extend("loop", None),
            "paged": lambda: run_extend("paged", prefix_meta),
            "one_call": lambda: run_paged_call(
                full_meta, total_len, True, outputs.get("one_call")
            ),
        }
        entry = {
            "context_len": context_len,
            "q_len": q_len,
            "max_page_offset_gib": round(
                row.max().item() * self.kv.stride(0) / 2**30, 2
            ),
            "ms": {},
            "causal_new_tokens_ms": self.causal_ms,
            "dense_chunk_ms": self.dense_chunk_ms,
            "dense_chunk_pflops": self.dense_chunk_pflops,
        }
        for name in self.variants:
            entry["ms"][name] = time_ms(runners[name], args.warmup, args.iters)
        # The time without the causal FMHA on the new tokens is the context
        # attention cost. That is the part the paged path replaces.
        for name in ("loop", "paged"):
            if name in self.variants:
                entry[f"{name}_context_ms"] = entry["ms"][name] - self.causal_ms
        if "paged" in self.variants:
            prefix_ms = time_ms(
                lambda: run_paged_call(prefix_meta, context_len, False),
                args.warmup,
                args.iters,
            )
            entry["paged_prefix_kernel_ms"] = prefix_ms
            entry["paged_prefix_pflops"] = attention_pflops(
                q_len, context_len, prefix_ms
            )

        if "loop" in self.variants:
            for name in self.variants:
                if name != "loop":
                    entry[f"{name}_vs_loop"] = errors(outputs[name], outputs["loop"])
        if args.ref_rows > 0:
            rows = torch.linspace(0, q_len - 1, args.ref_rows, device=device).long()
            reference = reference_rows(self.kv, row, context_len, p, rows)
            for name in self.variants:
                entry[f"{name}_vs_fp32"] = errors(outputs[name][rows], reference)
        for name in self.variants:
            entry[f"{name}_finite"] = bool(torch.isfinite(outputs[name]).all())

        if args.profile:
            for name in self.variants:
                names = kernel_names(runners[name])
                entry[f"{name}_kernels"] = {
                    key: {"count": count, "ms": round(ms, 3)}
                    for key, (count, ms) in sorted(
                        names.items(), key=lambda item: -item[1][1]
                    )
                }
            # The ASM kernels are named fmha_fwd_hd256_fp8_paged_varlen and
            # fmha_fwd_hd256_fp8_causal_paged_varlen. Without that name the
            # call fell back to the CK batch prefill kernel.
            for name in ("paged", "one_call"):
                if name in self.variants:
                    entry[f"{name}_used_asm_kernel"] = any(
                        "paged_varlen" in key for key in entry[f"{name}_kernels"]
                    )
        return entry


def main() -> None:
    args = parse_args()
    bench = Bench(args)
    results = []
    for context_len in [int(c) for c in args.contexts.split(",")]:
        entry = bench.run_context(context_len)
        results.append(entry)
        print(json.dumps(entry, indent=1))

    header = f"{'context':>9} " + " ".join(
        f"{name + ' ms':>14}" for name in bench.variants
    )
    print(header)
    for entry in results:
        cells = " ".join(f"{entry['ms'][name]:14.2f}" for name in bench.variants)
        print(f"{entry['context_len']:>9} {cells}")
    if args.json:
        payload = {
            "device": current_platform.get_device_name(bench.device.index),
            "torch": torch.__version__,
            "env": {
                key: os.environ.get(key)
                for key in (
                    "VLLM_ROCM_USE_AITER",
                    "VLLM_ROCM_USE_AITER_MHA",
                    "VLLM_ROCM_FP8_DIRECT_CONTEXT_GATHER",
                    "AITER_ASM_DIR",
                    "HIP_FORCE_DEV_KERNARG",
                )
            },
            "args": vars(args),
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
