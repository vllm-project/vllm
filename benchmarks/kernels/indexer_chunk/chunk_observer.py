# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pre-consumer snapshots indexed by the actual PR2 helper's global row window."""

import functools
import inspect


class ChunkObserver:
    def __init__(
        self, torch, module, candidates, selector_class, rows, width, length, candidate
    ):
        self.torch = torch
        self.module = module
        self.candidates = candidates
        self.selector_class = selector_class
        self.rows = rows
        self.width = width
        self.length = length
        self.window = (0, rows)
        self.records = []
        self.stride = None
        self.upcast_calls = self.candidate_calls = 0
        self.converted_rows = torch.zeros(rows, dtype=torch.bool)
        self.buffers = {
            "producer": torch.empty((rows, length), dtype=torch.int16, device="cuda"),
            "conversion": torch.empty((rows, length), dtype=torch.int32, device="cuda"),
        }
        if candidate:
            self.buffers["candidate"] = torch.empty(
                (rows, (length + 7) // 8), dtype=torch.int32, device="cuda"
            )
        self.originals = []

    def replace(self, owner, name, replacement):
        self.originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def __enter__(self):
        torch = self.torch
        m = self.module
        original_rows = getattr(m, "_varlen_decode_rows", None)
        signature = (
            inspect.signature(original_rows) if original_rows is not None else None
        )

        @functools.wraps(original_rows)
        def row_window(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            old = self.window
            start, end = bound["start"], bound["end"]
            assert 0 <= start < end <= self.rows
            self.window = (start, end)
            self.records.append(dict(operation="window_begin", start=start, end=end))
            try:
                result = original_rows(*args, **kwargs)
                self.records.append(
                    dict(operation="window_success", start=start, end=end)
                )
                return result
            finally:
                self.window = old

        original_producer = m.fp8_fp4_paged_mqa_logits

        @functools.wraps(original_producer)
        def producer(*args, **kwargs):
            value = original_producer(*args, **kwargs)
            start, end = self.window
            assert tuple(value.shape) == (end - start, self.width)
            assert value.dtype == torch.bfloat16
            if self.stride is not None:
                assert self.stride == value.stride(0)
            self.stride = value.stride(0)
            assert value.shape[1] >= self.length and value.stride(1) == 1
            self.buffers["producer"][start:end].copy_(
                value[:, : self.length].view(torch.int16)
            )
            self.records.append(
                dict(
                    operation="producer",
                    start=start,
                    end=end,
                    shape=list(value.shape),
                    stride=list(value.stride()),
                    q_shape=list(args[0][0].shape),
                    cache_shape=list(args[1].shape),
                    cache_stride=list(args[1].stride()),
                    seq_shape=list(args[3].shape),
                    block_table_shape=list(args[4].shape),
                    block_table_stride=list(args[4].stride()),
                    schedule_shape=list(args[5].shape),
                )
            )
            return value

        original_schedule = getattr(m, "get_paged_mqa_logits_metadata", None)

        @functools.wraps(original_schedule)
        def schedule(*args, **kwargs):
            value = original_schedule(*args, **kwargs)
            start, end = self.window
            self.records.append(
                dict(
                    operation="schedule",
                    start=start,
                    end=end,
                    seq_shape=list(args[0].shape),
                    page=args[1],
                    num_sms=args[2],
                    shape=list(value.shape),
                    stride=list(value.stride()),
                )
            )
            return value

        original_upcast = m.as_fp32_logits

        @functools.wraps(original_upcast)
        def upcast(*args, **kwargs):
            value = original_upcast(*args, **kwargs)
            start, end = self.window
            assert value.shape[0] == end - start and value.dtype == torch.float32
            self.buffers["conversion"][start:end].copy_(
                value[:, : self.length].view(torch.int32)
            )
            self.converted_rows[start:end] = True
            self.upcast_calls += 1
            self.records.append(dict(operation="upcast", start=start, end=end))
            return value

        original_ds = self.candidates.deep_select_topk

        @functools.wraps(original_ds)
        def deep_select(value, k, *args, **kwargs):
            assert "candidate" in self.buffers and k == 2048
            start, end = self.window
            assert value.shape[0] == end - start and value.dtype == torch.float32
            target = self.buffers["candidate"][start:end]
            target.copy_(value[:, : target.shape[1]].view(torch.int32))
            self.candidate_calls += 1
            self.records.append(
                dict(
                    operation="candidate_ds",
                    start=start,
                    end=end,
                    dtype=str(value.dtype),
                    stride=list(value.stride()),
                )
            )
            return original_ds(value, k, *args, **kwargs)

        original_resolve = self.selector_class.resolve_backend

        @functools.wraps(original_resolve)
        def resolve(provider, logits, topk, num_rows):
            result = original_resolve(provider, logits, topk, num_rows)
            start, end = self.window
            assert num_rows == end - start
            self.records.append(
                dict(
                    operation="selector_resolve",
                    start=start,
                    end=end,
                    configured=provider._backend,
                    resolved=result,
                    dtype=str(logits.dtype),
                    shape=list(logits.shape),
                    stride=list(logits.stride()),
                )
            )
            return result

        if original_rows is not None:
            self.replace(m, "_varlen_decode_rows", row_window)
        self.replace(m, "fp8_fp4_paged_mqa_logits", producer)
        self.replace(m, "as_fp32_logits", upcast)
        if original_schedule is not None:
            self.replace(m, "get_paged_mqa_logits_metadata", schedule)
        self.replace(self.candidates, "deep_select_topk", deep_select)
        if not self.candidates.has_deep_select():
            observer = self
            original_scores = self.candidates._block_scores_kernel

            class ScoreKernelObserver:
                def __getitem__(self, grid):
                    launch = original_scores[grid]

                    def snapshot(*args, **kwargs):
                        result = launch(*args, **kwargs)
                        scores = args[3]
                        start, end = observer.window
                        target = observer.buffers["candidate"][start:end]
                        target.copy_(scores[:, : target.shape[1]].view(torch.int32))
                        observer.candidate_calls += 1
                        observer.records.append(
                            dict(operation="candidate_fallback", start=start, end=end)
                        )
                        return result

                    return snapshot

            self.replace(self.candidates, "_block_scores_kernel", ScoreKernelObserver())
        self.replace(self.selector_class, "resolve_backend", resolve)
        return self

    def poison(self):
        for value in self.buffers.values():
            value.bitwise_not_()

    def __exit__(self, *exc):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()
