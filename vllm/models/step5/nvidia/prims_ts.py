# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16/FP8 PrimTS attention plans for Step5's cache and selector ABI."""

import math

import torch

from .flashinfer_sparse import Step5FlashInferSparse
from .ops.attention_gate import headwise_sigmoid_gate

MIN_DECODE_BATCH_SIZE = 32
_SPARSE_BLOCK_SIZE = 8


def uses_prims_ts_attention(metadata, num_speculative_tokens, kv_cache_dtype):
    if metadata is None or metadata.num_actual_tokens == 0:
        return False
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return True
    # Full CUDA graphs select attention using their padded request capacity.
    requests, query_len = metadata.num_actual_reqs, metadata.max_query_len
    decode = (
        metadata.num_actual_tokens == requests * query_len
        and query_len <= num_speculative_tokens + 1
    )
    if not decode:
        return True
    return (
        num_speculative_tokens == 3
        and query_len == 4
        and requests >= MIN_DECODE_BATCH_SIZE
    )


def get_kv_cache_and_scales(query, cache, layer):
    """View the cache and return descales for Q quantized by the producer.

    QK uses q_scale * k_scale and PV uses v_scale inside attention. Cache
    views and cached CPU scales require no GPU kernel or data transfer.
    """
    if cache.dtype == torch.uint8:
        cache = cache.view(torch.float8_e4m3fn)
    if query.dtype != cache.dtype or cache.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise NotImplementedError(
            "Step5 PrimTS requires matching BF16/E4M3 QKV; "
            "FP8 Q must be quantized by the Q/KV producer"
        )
    if cache.dtype == torch.bfloat16:
        return cache, 1.0, 1.0
    return cache, layer._q_scale_float * layer._k_scale_float, layer._v_scale_float


def _require_qkv(query, cache):
    """Check Step5's shared K/V channels; FlashInfer receives zero-copy slices."""
    if query.dtype != cache.dtype or query.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise NotImplementedError("Step5 PrimTS requires homogeneous BF16 or E4M3 QKV")
    if cache.shape[-1] != 2 * query.shape[-1]:
        raise ValueError("Step5 cache must concatenate K/V with 2 * head_dim channels")


def _sparse_wrapper(query, cache, table, offsets, group, topk, model_len, split_kv):
    from flashinfer.attention.prims_ts import (
        QTokenKvBlockSparsePagedTSWrapper,
        get_q_token_kv_block_sparse_workspace_size,
    )

    _require_qkv(query, cache)
    head_dim = query.shape[-1]
    model_len = min(model_len, table.shape[1] * cache.shape[2])
    size = get_q_token_kv_block_sparse_workspace_size(
        query,
        cache[..., :head_dim],
        table,
        block_topk=topk,
        max_seq_len_kv=model_len,
        qo_indptr=offsets,
        group_size=group,
        kv_block_size=_SPARSE_BLOCK_SIZE,
        split_kv=split_kv,
        share_pattern_across_kv_heads=False,
        count_current_block_in_topk=False,
        o_data_type=torch.bfloat16,
    )
    workspace = torch.empty(size, dtype=torch.uint8, device=query.device)
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        offsets.numel() - 1 if offsets is not None else query.shape[0] * query.shape[1],
        query.shape[-2],
        cache.shape[1],
        head_dim,
        _SPARSE_BLOCK_SIZE,
        cache.shape[2],
        topk,
        model_len,
        group_size=group,
        device=query.device,
        workspace_buffer=workspace,
        use_packed_q=offsets is not None,
        split_kv=split_kv,
        share_pattern_across_kv_heads=False,
        count_current_block_in_topk=False,
        q_data_type=query.dtype,
        o_data_type=torch.bfloat16,
    )
    return wrapper


class Step5PrimsTSDecode:
    """Reuse a fixed-Q plan over vLLM's existing request capacity."""

    def __init__(
        self,
        query,
        cache,
        table,
        *,
        topk,
        query_len,
        model_len,
        num_sms,
    ):
        from flashinfer.attention.prims_ts import (
            suggest_q_token_kv_block_sparse_group_size,
        )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm Step5 PrimTS decode before graph capture")
        self.head_dim = query.shape[-1]
        self.groups = cache.shape[1]
        tokens = query.shape[0] // self.groups
        self.tokens = tokens
        if tokens % query_len:
            raise ValueError("Fixed decode must contain complete requests")
        self.heads = query.shape[1] * self.groups
        self.query_len = query_len
        self.topk = topk
        # Current contributes up to one full block outside the history budget.
        max_selected_kv_tokens = (topk + 1) * _SPARSE_BLOCK_SIZE
        self.group = suggest_q_token_kv_block_sparse_group_size(
            batch_size=self.tokens // query_len,
            seq_len_q=query_len,
            selected_seq_len_kv=max_selected_kv_tokens,
            num_qo_heads=self.heads,
            num_kv_heads=self.groups,
            multi_processor_count=num_sms,
            head_dim=self.head_dim,
            q_data_type=query.dtype,
            split_kv=True,
            kv_block_size=_SPARSE_BLOCK_SIZE,
        )
        self.group = math.gcd(self.group, query_len)
        self.shape = (
            tokens // query_len,
            query_len // self.group,
            self.group,
            self.heads,
            self.head_dim,
        )
        self.ids = torch.empty(
            (self.tokens, self.groups, topk), dtype=torch.int32, device=query.device
        )
        self.wrapper = _sparse_wrapper(
            query.view(self.shape),
            cache,
            table,
            None,
            self.group,
            topk,
            model_len,
            True,
        )

    def run(
        self,
        query,
        cache,
        table,
        seq_lens,
        active,
        out,
        scale,
        gate,
        v_scale=1.0,
    ):
        tokens = query.shape[0] // self.groups
        if tokens != self.tokens:
            raise ValueError("Decode shape must match its vLLM request capacity")
        self.wrapper.run(
            query.view(self.shape),
            (cache[..., : self.head_dim], cache[..., self.head_dim :]),
            table,
            self.ids,
            seq_lens=seq_lens,
            sm_scale=scale,
            v_scale=v_scale,
            out=out.view(self.shape),
        )
        output = out.view(tokens, self.heads, self.head_dim)
        if gate is not None:
            # TODO: fuse the gate into the final epilogue, preserving BF16 rounding.
            headwise_sigmoid_gate(
                output.flatten(1),
                gate,
                active=active.view(tokens, self.groups)[:, 0],
                out=output.flatten(1),
            )


class Step5PrimsTSPrefill:
    """Collect all provider selections, then run one multi-head sparse launch."""

    def __init__(
        self,
        shape,
        dtype,
        device,
        positions,
        region_count,
        scale,
        request_offsets,
        groups,
        model_len,
        num_sms,
        *,
        max_topk=None,
    ):
        from flashinfer.attention.prims_ts import (
            make_q_token_kv_block_sparse_qo_indptr,
            suggest_q_token_kv_block_sparse_group_size,
        )

        tokens, heads, dim = shape
        if dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise NotImplementedError("Step5 PrimTS prefill requires BF16/E4M3 queries")
        self.head_dim = dim
        lengths = [
            int(b - a) for a, b in zip(request_offsets[:-1], request_offsets[1:])
        ]
        self.groups, self.tokens = groups, tokens
        self.topk = region_count - 1 if max_topk is None else max_topk
        self.scale, self.model_len = scale, model_len
        # This is a per-query capacity, not the original context length.
        max_selected_kv_tokens = (self.topk + 1) * _SPARSE_BLOCK_SIZE
        self.group = suggest_q_token_kv_block_sparse_group_size(
            batch_size=len(lengths),
            seq_len_q=max(lengths),
            selected_seq_len_kv=max_selected_kv_tokens,
            num_qo_heads=heads * groups,
            num_kv_heads=groups,
            multi_processor_count=num_sms,
            head_dim=dim,
            q_data_type=dtype,
            split_kv=False,
            kv_block_size=_SPARSE_BLOCK_SIZE,
        )
        cpu_offsets = make_q_token_kv_block_sparse_qo_indptr(
            torch.as_tensor(request_offsets, dtype=torch.int32, device="cpu"),
            tokens,
            group_size=self.group,
            device="cpu",
        )
        self.offsets = cpu_offsets.pin_memory().to(device, non_blocking=True)
        self.ids = torch.empty(
            tokens, groups, self.topk, dtype=torch.int32, device=device
        )
        self.query_buffer = None
        self.wrapper = None

    def page_metadata(self, table, owners, page_stride=1, start=0, end=None):
        # PrimTS builds its own union; leave the selector's ordinary IDs intact.
        return None

    def fixup_positions(self, positions):
        self.positions = positions

    def bind(
        self,
        query,
        cache,
        table,
        output,
        *,
        owners,
        positions,
        qk_scale=1.0,
        v_scale=1.0,
    ):
        _require_qkv(query, cache)
        self.owners, self.positions = owners, positions
        self.qk_scale, self.v_scale = qk_scale, v_scale
        self.query_buffer = query
        self.cache, self.table, self.output = cache, table, output
        self.current_group = 0
        if self.wrapper is None:
            self.wrapper = _sparse_wrapper(
                query,
                cache,
                table,
                self.offsets,
                self.group,
                self.topk,
                self.model_len,
                False,
            )

    def selection_output(self, start, end):
        return self.ids[start:end, self.current_group]

    def finish(self, query, cache, out):
        if self.wrapper is None:
            raise RuntimeError("Bind PrimTS prefill buffers before finishing")
        self.current_group += 1
        if self.current_group == self.groups:
            self.wrapper.run(
                self.query_buffer,
                (self.cache[..., : self.head_dim], self.cache[..., self.head_dim :]),
                self.table,
                self.ids,
                self.owners,
                self.positions,
                qo_indptr=self.offsets,
                sm_scale=self.scale * self.qk_scale,
                v_scale=self.v_scale,
                out=self.output,
            )


class Step5PrimsTSSWA:
    """Reuse dense PrimTS plans and apply Step5's existing separate gate."""

    def __init__(self, model_len, window):
        self.model_len, self.window = model_len, window
        self.plans = {}

    def run(self, query, cache, metadata, output, gate, scale, v_scale=1.0):
        from flashinfer.attention.prims_ts import (
            BatchDecodePagedTSWrapper,
            BatchPrefillPagedTSWrapper,
        )

        _require_qkv(query, cache)
        head_dim = query.shape[-1]
        n = metadata.num_actual_tokens
        output[n:].zero_()
        if not n:
            return output
        requests, qlen = metadata.seq_lens.shape[0], metadata.max_query_len
        decode = n == requests * qlen and qlen <= 4
        request_capacity = requests
        q_capacity = qlen if decode else 1 << (qlen - 1).bit_length()
        table = metadata.block_table[:requests]
        lengths = metadata.seq_lens
        model_len = min(self.model_len, table.shape[1] * cache.shape[2])
        key = (
            request_capacity,
            q_capacity,
            cache.data_ptr(),
            cache.stride(),
            table.shape[1],
            decode,
            query.shape[1:],
            query.dtype,
            scale,
            v_scale,
        )
        state = self.plans.get(key)
        if state is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm Step5 PrimTS SWA before graph capture")
            if decode:
                wrapper = BatchDecodePagedTSWrapper()
                wrapper.plan(
                    device=query.device,
                    batch_size=request_capacity,
                    num_qo_heads=query.shape[1],
                    num_kv_heads=cache.shape[1],
                    head_dim=head_dim,
                    page_size=cache.shape[2],
                    max_kv_len=model_len,
                    max_q_len=qlen,
                    q_data_type=query.dtype,
                    o_data_type=torch.bfloat16,
                    mask_type="causal",
                    # Step5's total window includes the current token.
                    window_left=self.window - 1,
                    split_kv=False,
                )
            else:
                wrapper = BatchPrefillPagedTSWrapper()
                wrapper.plan(
                    device=query.device,
                    batch_size=requests,
                    max_q_len=q_capacity,
                    max_kv_len=model_len,
                    num_qo_heads=query.shape[1],
                    num_kv_heads=cache.shape[1],
                    head_dim=head_dim,
                    q_dtype=query.dtype,
                    k_dtype=cache.dtype,
                    out_dtype=torch.bfloat16,
                    page_size=cache.shape[2],
                    mask_type="causal",
                    # Step5's total window includes the current token.
                    window_left=self.window - 1,
                    sm_scale=scale,
                    output_scale=v_scale,
                )
            state = wrapper
            self.plans[key] = state
        wrapper = state
        if decode:
            actual_shape = (
                (requests, qlen, query.shape[1], head_dim)
                if qlen > 1
                else (requests, query.shape[1], head_dim)
            )
            fixed_query = query[:n].view(actual_shape)
            fixed_output = output[:n].view(actual_shape)
            wrapper.run(
                fixed_query,
                (cache[..., :head_dim], cache[..., head_dim:]),
                lengths,
                table,
                out=fixed_output,
                bmm1_scale=scale,
                bmm2_scale=v_scale,
                validate=False,
            )
        else:
            wrapper.run(
                query[:n],
                cache[..., :head_dim],
                cache[..., head_dim:],
                metadata.query_start_loc[: requests + 1],
                table,
                lengths,
                out=output[:n],
                validate=False,
            )
        if gate is not None:
            headwise_sigmoid_gate(
                output[:n].flatten(1),
                gate[:n],
                slot_mapping=metadata.slot_mapping[:n],
                out=output[:n].flatten(1),
            )
        return output


class Step5PrimsTSSparse(Step5FlashInferSparse):
    """Keep the selector and summaries while replacing main attention with PrimTS."""

    native_decode_attention = Step5FlashInferSparse.select_and_run_decode_attention

    def __init__(self, device, max_model_len, topk):
        super().__init__(device)
        self.max_model_len = max_model_len
        self.topk = topk
        self._decode_plans = {}

    def prepare_prefill_attention(
        self,
        query_shape,
        dtype,
        device,
        block_size,
        positions,
        region_count,
        scale,
        *,
        contiguous_query=True,
        request_offsets=None,
        num_kv_groups=1,
    ):
        assert request_offsets is not None
        # Four bounded capacities avoid recompiling for every live history
        # width while keeping short prompts from paying the full 512-block plan.
        live_topk = region_count - 1
        plan_topk = min(self.topk, max(64, 1 << (live_topk - 1).bit_length()))
        return Step5PrimsTSPrefill(
            query_shape,
            dtype,
            device,
            positions,
            region_count,
            scale,
            request_offsets,
            num_kv_groups,
            self.max_model_len,
            self._num_sms,
            max_topk=plan_topk,
        )

    def select_and_run_decode_attention(
        self,
        query,
        kv_cache,
        scores,
        current,
        topk,
        block_table,
        positions,
        active,
        out,
        scale,
        *,
        seq_lens=None,
        topk_lengths=None,
        clear_inactive=True,
        k_scale=None,
        v_scale=None,
        query_len=1,
        gate=None,
    ):
        if seq_lens is None:
            raise ValueError("PrimTS decode requires per-request KV lengths")
        tokens = query.shape[0] // kv_cache.shape[1]
        key = (
            tokens,
            query.shape[1:],
            query.stride(),
            query.dtype,
            kv_cache.data_ptr(),
            kv_cache.shape,
            kv_cache.stride(),
            block_table.shape[1],
            topk,
            query_len,
        )
        plan = self._decode_plans.get(key)
        if plan is None:
            plan = Step5PrimsTSDecode(
                query,
                kv_cache,
                block_table,
                topk=topk,
                query_len=query_len,
                model_len=self.max_model_len,
                num_sms=self._num_sms,
            )
            self._decode_plans[key] = plan
        self.select_regions(
            scores,
            current,
            topk,
            topk_lengths=topk_lengths,
            sort=False,
            out=plan.ids.view(-1, topk),
            history_only=True,
        )
        plan.run(
            query,
            kv_cache,
            block_table,
            seq_lens,
            active,
            out,
            scale * (1.0 if k_scale is None else k_scale),
            gate,
            v_scale=1.0 if v_scale is None else v_scale,
        )
