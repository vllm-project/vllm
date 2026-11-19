# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time one PrefixLM attention call on a layer without a sliding window.

Compares, over paged KV laid out as the FlashAttention backend lays it out:

  causal  FlashAttentionImpl.forward without ranges (stock FA4 causal call over
          the same scheduled tokens, the baseline)
  rows    FlashAttentionImpl.forward on a layer that applies PrefixLM ranges:
          the ranges as causal and bidirectional varlen rows, one FA4 call with
          per-row causal flags (SM90)
  mask    FlashAttentionImpl.forward on a layer not told it applies ranges:
          FA4's PrefixLM mask_mod, which runs the whole step non-causally
  triton  Triton's PrefixLM kernel (the TRITON_FLASH_ATTN composite's route)

Each prompt is a state split into fixed-size spans and a question span, and
every path gets the ranges the model runner would give the step: those of the
spans with uncomputed tokens (`compute_bidirectional_span_ranges`). Scenarios:
"first" prefills the whole prompt; "grown" prefills a state grown by `--grow`
tokens with its complete spans cached; "next" prefills a new question with
every full KV block of the state cached; "mixed" adds 64 decodes at 4K and a
2K plain prefill to "first" (more tokens than a default step budget holds).
"row build ms" is the builder's host time for the step.
Before timing, every PrefixLM path is checked against an fp32 reference of the
span mask on small steps, and the script aborts on a mismatch.

    python benchmarks/kernels/benchmark_prefix_lm_rows.py --shapes 31B
"""

import argparse
import math
import statistics
import time
from dataclasses import replace
from types import SimpleNamespace

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.backends.prefix_lm_rows import PrefixLMRowsBuilder
from vllm.v1.attention.backends.utils import (
    compute_bidirectional_span_ranges,
    compute_mm_prefix_range_tensor,
    fill_mm_prefix_query_ranges,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

# Gemma 4 full-attention layers on one GPU: query heads, KV heads, head size,
# block size.
SHAPES = {
    "E2B": (8, 1, 512, 32),
    "E4B": (8, 2, 512, 32),
    "12B": (16, 1, 512, 64),
    "26B-A4B": (16, 2, 512, 32),
    "31B": (32, 4, 512, 32),
}
PATHS = ("causal", "rows", "mask", "triton")
DEVICE = "cuda"


def spans_of(state_len: int, span_len: int, prompt_len: int) -> list[tuple]:
    spans = [(s, min(s + span_len, state_len)) for s in range(0, state_len, span_len)]
    return spans + [(state_len, prompt_len)]


class Step:
    """Requests as (seq_len, num_computed, half-open spans) over random paged KV
    in the backend's default layout."""

    def __init__(self, reqs, num_q_heads, num_kv_heads, head_size, block_size):
        gen = torch.Generator(device=DEVICE).manual_seed(0)
        self.reqs, self.head_size, self.block_size = reqs, head_size, block_size
        self.num_q_heads, self.num_kv_heads = num_q_heads, num_kv_heads
        blocks = [math.ceil(seq_len / block_size) for seq_len, _, _ in reqs]
        num_blocks = sum(blocks) + 8
        # Token-major blocks, viewed as (num_blocks, num_kv_heads, block_size,
        # 2 * head_size) as the backend views its cache.
        kv = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            2 * head_size,
            generator=gen,
            device=DEVICE,
            dtype=torch.bfloat16,
        ).transpose(1, 2)
        self.kv_cache = kv
        # (num_blocks, block_size, num_kv_heads, head_size) views, as the backend.
        self.k, self.v = kv.transpose(1, 2).split(head_size, dim=-1)
        perm = torch.randperm(num_blocks, generator=gen, device=DEVICE).int()
        self.block_table = torch.zeros(
            len(reqs), max(blocks), dtype=torch.int32, device=DEVICE
        )
        offset = 0
        for i, n in enumerate(blocks):
            self.block_table[i, :n] = perm[offset : offset + n]
            offset += n
        query_lens = [seq_len - computed for seq_len, computed, _ in reqs]
        self.qsl_cpu = torch.tensor(
            [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
        )
        self.qsl = self.qsl_cpu.to(DEVICE)
        self.seq_cpu = torch.tensor([s for s, _, _ in reqs], dtype=torch.int32)
        self.seq = self.seq_cpu.to(DEVICE)
        self.q = torch.randn(
            int(self.qsl_cpu[-1]),
            num_q_heads,
            head_size,
            generator=gen,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        self.max_q, self.max_k = max(query_lens), max(int(s) for s in self.seq_cpu)
        self.scale = head_size**-0.5
        # Inclusive ranges, as CommonAttentionMetadata.mm_req_doc_ranges.
        self.ranges = (
            compute_bidirectional_span_ranges(
                [spans for _, _, spans in reqs],
                [computed for _, computed, _ in reqs],
                [seq_len for seq_len, _, _ in reqs],
            )
            or {}
        )
        builder = PrefixLMRowsBuilder(torch.device(DEVICE))
        build_args = (
            self.ranges,
            self.qsl_cpu,
            self.seq_cpu,
            self.seq,
            self.block_table,
            self.max_k,
        )
        self.rows = builder.build(*build_args)
        # Host time to build the rows, paid once per step and attention group.
        torch.accelerator.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            builder.build(*build_args)
        torch.accelerator.synchronize()
        self.build_ms = (time.perf_counter() - start) / 20 * 1000
        buf = torch.empty(int(self.qsl_cpu[-1]), 2, dtype=torch.int32)
        n = fill_mm_prefix_query_ranges(
            buf.numpy(), self.ranges, self.qsl_cpu, self.seq_cpu
        )
        num_tokens = int(self.qsl_cpu[-1])
        plain = FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=self.max_q,
            query_start_loc=self.qsl,
            max_seq_len=self.max_k,
            seq_lens=self.seq,
            block_table=self.block_table,
            slot_mapping=torch.empty(0, dtype=torch.int64, device=DEVICE),
        )
        mask = replace(
            plain, mm_prefix_query_range_tensor=buf[:n].to(DEVICE) if n else None
        )
        self.metadata = {
            "causal": plain,
            "mask": mask,
            "rows": replace(mask, prefix_lm_rows=self.rows),
        }
        with set_current_vllm_config(VllmConfig()):
            impl_args = dict(
                num_heads=num_q_heads,
                head_size=head_size,
                scale=self.scale,
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
            )
            self.impls = {
                "causal": FlashAttentionImpl(**impl_args),
                "mask": FlashAttentionImpl(**impl_args),
                "rows": FlashAttentionImpl(**impl_args, prefix_lm_ranges=True),
            }
        one = torch.ones((), device=DEVICE)
        self.layer = SimpleNamespace(
            _q_scale=one,
            _k_scale=one,
            _v_scale=one,
            _q_scale_float=1.0,
            _k_scale_float=1.0,
            _v_scale_float=1.0,
        )
        self.new_kv = torch.empty(
            num_tokens, num_kv_heads, head_size, device=DEVICE, dtype=torch.bfloat16
        )
        self.triton_ranges = compute_mm_prefix_range_tensor(
            self.ranges, len(reqs), torch.device(DEVICE)
        )
        self.ones = torch.ones(1, 1, device=DEVICE).expand(len(reqs), num_kv_heads)
        self.out = torch.empty_like(self.q)

    def run(self, path: str, out: torch.Tensor) -> None:
        if path in self.impls:
            if path == "rows":
                assert self.rows is not None, "this step prefills no range"
            self.impls[path].forward(
                self.layer,
                self.q,
                self.new_kv,
                self.new_kv,
                self.kv_cache,
                self.metadata[path],
                out,
            )
        elif path == "triton":
            unified_attention(
                q=self.q,
                k=self.k,
                v=self.v,
                out=out,
                cu_seqlens_q=self.qsl,
                max_seqlen_q=self.max_q,
                seqused_k=self.seq,
                max_seqlen_k=self.max_k,
                softmax_scale=self.scale,
                causal=True,
                window_size=(-1, -1),
                block_table=self.block_table,
                softcap=0,
                q_descale=None,
                k_descale=self.ones,
                v_descale=self.ones,
                mm_prefix_range=self.triton_ranges,
            )

    def reference(self) -> torch.Tensor:
        """fp32 attention under the span mask, request by request."""
        out = torch.empty(self.q.shape, dtype=torch.float32, device=DEVICE)
        rep = self.num_q_heads // self.num_kv_heads
        for i, (seq_len, computed, spans) in enumerate(self.reqs):
            blocks = self.block_table[i, : math.ceil(seq_len / self.block_size)].long()
            k = self.k[blocks].reshape(-1, self.num_kv_heads, self.head_size)[:seq_len]
            v = self.v[blocks].reshape(-1, self.num_kv_heads, self.head_size)[:seq_len]
            k, v = (t.float().repeat_interleave(rep, 1) for t in (k, v))
            pos = torch.arange(computed, seq_len, device=DEVICE)
            horizon = pos + 1
            for start, end in spans:
                inside = (pos >= start) & (pos < end)
                horizon = torch.where(inside, torch.full_like(pos, end), horizon)
            mask = torch.arange(seq_len, device=DEVICE)[None, :] < horizon[:, None]
            q = self.q[int(self.qsl_cpu[i]) : int(self.qsl_cpu[i + 1])].float()
            scores = torch.einsum("qhd,khd->hqk", q, k) * self.scale
            probs = scores.masked_fill(~mask[None], float("-inf")).softmax(-1)
            out[int(self.qsl_cpu[i]) : int(self.qsl_cpu[i + 1])] = torch.einsum(
                "hqk,khd->qhd", probs, v
            )
        return out


def scenarios(prompt_len, span_len, question_len, grow, block_size):
    state = prompt_len - question_len
    spans = spans_of(state, span_len, prompt_len)
    previous = state - grow  # the state one step earlier
    out = {"first": [(prompt_len, 0, spans)]}
    if previous >= span_len:
        out["grown"] = [(prompt_len, (previous // span_len) * span_len, spans)]
    out["next"] = [(prompt_len, (state // block_size) * block_size, spans)]
    out["mixed"] = out["first"] + [(4096, 4095, [])] * 64 + [(2048, 0, [])]
    return out


def check(shape) -> None:
    """Every PrefixLM path against the fp32 reference on small steps; abort if
    one is off."""
    _, _, _, block_size = SHAPES[shape]
    cases = {
        "spans_and_question": [(1500, 0, spans_of(1200, 256, 1500))],
        "cache_hit_inside_a_span": [(1500, 3 * block_size, spans_of(1200, 256, 1500))],
        "gaps_between_spans": [(1400, 0, [(0, 300), (450, 700), (900, 1000)])],
        "spans_plain_and_decodes": [
            (1300, 0, spans_of(1000, 256, 1300)),
            (900, 0, []),
            (2000, 1999, []),
            (1600, 640, spans_of(1300, 256, 1600)),
        ],
        "one_token_and_unaligned_spans": [(700, 0, [(0, 1), (1, 2), (5, 133)])],
        # A few queries over many cached keys, as in a next question.
        "next_question_long_context": [
            (8192, 7872, spans_of(7892, 1024, 8192)),
            (6000, 5999, []),
        ],
        "more_than_256_rows": [(800, 0, [(s, s + 2) for s in range(0, 600, 2)])],
    }
    for name, reqs in cases.items():
        step = Step(reqs, *SHAPES[shape])
        reference = step.reference()
        floor = (reference.bfloat16().float() - reference).abs().max().item()
        step.run("causal", torch.zeros_like(step.q))  # stock kernel compiled first
        for path in ("rows", "mask", "triton"):
            out = torch.zeros_like(step.q)
            step.run(path, out)
            torch.accelerator.synchronize()
            err = (out.float() - reference).abs().max().item()
            print(
                f"check {shape} {name} {path}: max error {err:.4f} (bf16 {floor:.4f})"
            )
            if err > max(4 * floor, 1e-2):
                raise SystemExit(f"{path} does not match the reference on {name}")


def timed(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    times = []
    for _ in range(iters):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.accelerator.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=SHAPES)
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[8192])
    parser.add_argument("--span-len", type=int, default=1024)
    parser.add_argument("--question-len", type=int, default=300)
    parser.add_argument("--grow", type=int, default=400)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    if not current_platform.is_device_capability_family(90):
        raise SystemExit("the rows run in one call on SM90 only")
    if not args.skip_check:
        for shape in args.shapes:
            check(shape)

    print(
        "| shape | prompt | scenario | scheduled tokens | rows | causal ms "
        "| rows / causal | mask / causal | triton / causal | row build ms |"
    )
    print("|" + " --- |" * 10)
    for shape in args.shapes:
        for prompt_len in args.prompt_lens:
            cases = scenarios(
                prompt_len,
                args.span_len,
                args.question_len,
                args.grow,
                SHAPES[shape][3],
            )
            for name, reqs in cases.items():
                step = Step(reqs, *SHAPES[shape])
                rows = step.rows
                assert rows is not None and step.impls["rows"].runs_prefix_lm_rows
                ms = {
                    p: timed(lambda p=p, step=step: step.run(p, step.out), args.iters)
                    for p in PATHS
                }
                ratios = " | ".join(
                    f"{ms[p] / ms['causal']:.2f}" for p in ("rows", "mask", "triton")
                )
                print(
                    f"| {shape} | {prompt_len} | {name} | {int(step.qsl_cpu[-1])} | "
                    f"{rows.num_rows} | {ms['causal']:.3f} | {ratios} | "
                    f"{step.build_ms:.3f} |"
                )


if __name__ == "__main__":
    main()
