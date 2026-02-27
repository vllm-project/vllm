# SPDX-License-Identifier: Apache-2.0

"""
Benchmark vLLM v1 attention backends:
    FlashAttention, Triton, and FlashInfer.
"""

import argparse
import math
import re
import time

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable


def parse_batch_arg(batch_args):
    """
    Parse manual batch pairs ['q,kv', ...] into list[(q,kv)].
    """
    pairs = []
    for s in batch_args:
        try:
            q_str, kv_str = s.split(",")
            q, kv = int(q_str), int(kv_str)
            if kv < q:
                raise ValueError(f"kv_len ({kv}) >= q_len ({q}) required")
            pairs.append((q, kv))
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid batch pair '{s}': {e}") from e
    return pairs


def parse_batch_spec(spec: str):
    """
    Grammar per segment (underscore separated):
      (<count>?) q<q_len>(k?) (s<kv_len>(k?))? : prefill/extend
      (<count>?) s<kv_len>(k?)            : decode
    'k' suffix multiplies by 1024.
    Examples:
      q2k -> [(2048,2048)]
      q2  -> [(2,2)]
      8s1k-> [(1,1024)]*8
      2q1k_32s1k -> [(1024,1024)]*2 + [(1,1024)]*32
    """
    pairs = []
    for seg in spec.split("_"):
        m = re.match(r"^(?:(\d+))?q(\d+)(k?)(?:s(\d+)(k?))?$", seg)
        if m:
            cnt = int(m.group(1)) if m.group(1) else 1
            q_len = int(m.group(2))
            qlen = q_len * 1024 if m.group(3) == "k" else q_len
            if m.group(4):
                kv_len = int(m.group(4))
                klen = kv_len * 1024 if m.group(5) == "k" else kv_len
            else:
                klen = qlen
            pairs.extend([(qlen, klen)] * cnt)
            continue
        m = re.match(r"^(?:(\d+))?s(\d+)(k?)$", seg)
        if m:
            cnt = int(m.group(1)) if m.group(1) else 1
            kv_len = int(m.group(2))
            klen = kv_len * 1024 if m.group(3) == "k" else kv_len
            pairs.extend([(1, klen)] * cnt)
            continue
        raise argparse.ArgumentTypeError(f"Invalid batch spec '{seg}'")
    return pairs


def format_batch_spec(pairs):
    """Pretty-print list[(q,kv)] into human-readable segments."""
    from collections import Counter

    kinds = {"prefill": [], "extend": [], "specdecode": [], "decode": [], "unknown": []}
    for q, kv in pairs:
        if q > 1 and kv == q:
            kinds["prefill"].append((q, kv))
        elif q > 1 and kv > q:
            kinds["extend"].append((q, kv))
        elif q > 1 and q <= 16:
            kinds["specdecode"].append((q, kv))
        elif q == 1 and kv > 1:
            kinds["decode"].append((q, kv))
        else:
            kinds["unknown"].append((q, kv))
    parts = []
    for kind in ["prefill", "extend", "specdecode", "decode", "unknown"]:
        lst = kinds[kind]
        if not lst:
            continue
        cnt_total = len(lst)
        ctr = Counter(lst)
        inner = []
        for (q, kv), cnt in ctr.items():
            if kind == "prefill":
                size = f"{q // 1024}k" if q % 1024 == 0 else str(q)
                inner.append(f"{cnt}x{size}")
            elif kind == "decode":
                size = f"{kv // 1024}k" if kv % 1024 == 0 else str(kv)
                inner.append(f"{cnt}x{size}")
            else:
                qstr = f"{q // 1024}k" if q % 1024 == 0 else str(q)
                kstr = f"{kv // 1024}k" if kv % 1024 == 0 else str(kv)
                inner.append(f"{cnt}xq{qstr}s{kstr}")
        parts.append(f"{cnt_total} {kind} ({', '.join(inner)})")
    return ", ".join(parts)


# Mock stubs for vLLM's metadata builder
class MockLayer:
    def __init__(self, device):
        # Scale tensors and corresponding float values
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale = torch.tensor(1.0, device=device)
        # Scalar floats for FlashInfer wrappers
        self._k_scale_float = float(self._k_scale.item())
        self._v_scale_float = float(self._v_scale.item())
        self._q_scale_float = float(self._q_scale.item())


class MockModelConfig:
    """Mock model config holding Q/KV head counts and head dimension."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self._n_q = num_q_heads
        self._n_kv = num_kv_heads
        self._d = head_dim

    def get_num_attention_heads(self, _) -> int:
        return self._n_q

    def get_num_kv_heads(self, _) -> int:
        return self._n_kv

    def get_head_size(self) -> int:
        return self._d


class MockParallelConfig:
    pass


class MockCompilationConfig:
    def __init__(self):
        self.full_cuda_graph = False
        self.static_forward_context = {}


class MockVLLMConfig:
    def __init__(self):
        self.compilation_config = MockCompilationConfig()


class MockRunner:
    def __init__(
        self,
        seq_np,
        qs_np,
        device,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype,
    ):
        """
        Initialize with sequence lengths, device,
        head configuration, and data type.
        """
        # Model and parallel config
        self.model_config = MockModelConfig(
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
        self.parallel_config = MockParallelConfig()
        # vLLM config stub
        self.vllm_config = MockVLLMConfig()
        # Sequence metadata
        self.seq_lens_np = seq_np
        self.query_start_loc_np = qs_np
        self.device = device
        # Attention parameters
        self.attention_chunk_size = None
        # Record number of query heads and data type for FlashInfer
        self.num_query_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention backends")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["flash", "triton", "flashinfer"],
        default=["flash"],
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--num-layers", type=int, default=10, help="Number of transformer layers"
    )
    parser.add_argument(
        "--batch-specs",
        nargs="+",
        default=["q2k", "8s1k", "2q1k_32s1k", "2q1ks2k_32s1k", "32q4s1k", "4s32k"],
        help="Batch-spec strings",
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--num-q-heads", type=int, default=32, help="Query heads")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="KV heads")
    parser.add_argument(
        "--block-size", type=int, default=16, help="KV cache block size"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--repeats", type=int, default=1, help="Benchmark repetitions")
    args = parser.parse_args()

    console = Console()
    backends = args.backends
    multi = len(backends) > 1
    table = Table(title="Attention Benchmark")
    # BatchSpec labels should not wrap
    table.add_column("BatchSpec", no_wrap=True)
    for be in backends:
        # Time column
        col_time = f"{be} Time (s)"
        table.add_column(
            col_time, justify="right", no_wrap=True, min_width=len(col_time)
        )
        if multi:
            # Percentage column
            col_pct = f"{be} % of Fastest"
            table.add_column(
                col_pct, justify="right", no_wrap=True, min_width=len(col_pct)
            )

    for spec in tqdm(args.batch_specs, desc="Specs"):
        # Parse and (for FlashInfer) reorder
        pairs = parse_batch_spec(spec)
        decs = [(q, kv) for q, kv in pairs if q == 1 and kv > 1]
        prefs = [p for p in pairs if p not in decs]
        pairs = decs + prefs
        num_decs, num_prefs = len(decs), len(prefs)
        # Total number of new query tokens: sum of q-lengths for decoding and prefill
        num_dec_toks = sum(q for q, kv in decs)
        num_pref_toks = sum(q for q, kv in prefs)

        # Precompute sequence and block info
        q_lens = [q for q, _ in pairs]
        kv_lens = [kv for _, kv in pairs]
        total_q = sum(q_lens)
        q_off = [0] + [sum(q_lens[: i + 1]) for i in range(len(q_lens))]
        num_bl = [(kv + args.block_size - 1) // args.block_size for kv in kv_lens]
        max_bl = max(num_bl)

        device = torch.device(args.device)
        torch.cuda.set_device(device)
        dh, nq, nk = args.head_dim, args.num_q_heads, args.num_kv_heads
        scale = 1 / math.sqrt(dh)

        seq_np = np.array(kv_lens, dtype=np.int32)
        qs_np = np.array(q_off, dtype=np.int32)

        case_times = []
        for be in tqdm(args.backends, desc=f"Backends for {spec}", leave=False):
            if be == "flash":
                from vllm.v1.attention.backends.flash_attn import (
                    FlashAttentionBackend as BE,
                )

                dt = torch.float16
            elif be == "triton":
                from vllm.v1.attention.backends.triton_attn import (
                    TritonAttentionBackend as BE,
                )

                dt = torch.float32
            elif be == "flashinfer":
                from vllm.v1.attention.backends.flashinfer import (
                    FlashInferBackend as BE,
                )

                dt = torch.float16
            else:
                continue

            # Instantiate attention # impl
            impl = BE.get_impl_cls()(
                num_heads=nq,
                head_size=dh,
                scale=scale,
                num_kv_heads=nk,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
            )
            ml = MockLayer(device)

            # One-liner per-layer inputs
            q_list = [
                torch.randn(total_q, nq, dh, device=device, dtype=dt)
                for _ in range(args.num_layers)
            ]
            k_list = [
                torch.randn(total_q, nk, dh, device=device, dtype=dt)
                for _ in range(args.num_layers)
            ]
            v_list = [
                torch.randn(total_q, nk, dh, device=device, dtype=dt)
                for _ in range(args.num_layers)
            ]
            # KV cache shape differs for flashinfer vs other backends
            if be == "flashinfer":
                cache_list = [
                    torch.zeros(
                        max_bl, 2, args.block_size, nk, dh, device=device, dtype=dt
                    )
                    for _ in range(args.num_layers)
                ]
            else:
                cache_list = [
                    torch.zeros(
                        2, max_bl, args.block_size, nk, dh, device=device, dtype=dt
                    )
                    for _ in range(args.num_layers)
                ]

            # Build builder & metadata
            # Instantiate runner with data type for FlashInfer backend
            runner = MockRunner(seq_np, qs_np, device, nq, nk, dh, dt)
            bt = BlockTable(len(pairs), max_bl, total_q, False, device)
            for i, nb in enumerate(num_bl):
                bt.add_row(list(range(nb)), i)
            bt.commit(len(pairs))

            builder = BE.get_builder_cls()(
                runner=runner,
                kv_cache_spec=AttentionSpec(
                    block_size=args.block_size,
                    num_kv_heads=nk,
                    head_size=dh,
                    dtype=dt,
                    use_mla=False,
                ),
                block_table=bt,
            )

            # FlashInfer internal hack: set decode/prefill counts and hyperparameters
            if be == "flashinfer":
                builder._num_decodes = num_decs
                builder._num_prefills = num_prefs
                builder._num_decode_tokens = num_dec_toks
                builder._num_prefill_tokens = num_pref_toks
                # Initialize global hyperparameters for planning
                from vllm.v1.attention.backends.flashinfer import PerLayerParameters

                builder.global_hyperparameters = PerLayerParameters(
                    window_left=impl.sliding_window[0]
                    if hasattr(impl, "sliding_window")
                    else -1,
                    logits_soft_cap=impl.logits_soft_cap
                    if hasattr(impl, "logits_soft_cap")
                    else None,
                    sm_scale=impl.scale,
                )

            common_meta = CommonAttentionMetadata(
                query_start_loc=torch.tensor(q_off, dtype=torch.int32, device=device),
                seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
            )
            meta = builder.build(
                num_reqs=len(pairs),
                num_actual_tokens=total_q,
                max_query_len=max(q_lens),
                common_prefix_len=0,
                common_attn_metadata=common_meta,
            )

            # Warmup
            out = torch.empty(total_q, nq, dh, device=device, dtype=dt)
            for i in range(args.num_layers):
                impl.forward(
                    ml, q_list[i], k_list[i], v_list[i], cache_list[i], meta, output=out
                )
            torch.cuda.synchronize()

            # Timing
            times = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.time()
                for i in range(args.num_layers):
                    impl.forward(
                        ml,
                        q_list[i],
                        k_list[i],
                        v_list[i],
                        cache_list[i],
                        meta,
                        output=out,
                    )
                torch.cuda.synchronize()
                times.append((time.time() - start) / args.num_layers)
            case_times.append((be, sum(times) / len(times)))

        # Pivot timings into a single row per batch spec
        times = {be: t for be, t in case_times}
        best = min(times.values()) if times else 0.0
        row = [format_batch_spec(pairs)]
        for be in backends:
            t = times.get(be, 0.0)
            ts = f"{t:.6f}"
            row.append(ts)
            if multi:
                pct = t / best * 100 if best > 0 else 0.0
                ps = f"{pct:.1f}%"
                if t == best:
                    ps = f"[bold green]{ps}[/]"
                row.append(ps)
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()
