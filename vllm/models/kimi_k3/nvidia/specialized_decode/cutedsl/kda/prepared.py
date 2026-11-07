# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared KDA core. One packed projection supplies Q/K/V, the output gate, F_A and
beta; per-batch policies select the gate schedule and the projection row pitch."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, layout, stream, to_cute
from ..common.policy import batch_table
from . import fused_ws, specialized
from .fused import KdaFused
from .fused_ws import KdaFusedWS
from .kernel import KdaSplit
from .specialized import KdaWS
from .stream import KdaStream
from .wide import KdaWide

# KdaSplit schedule "early.lanes.occupancy.cluster.evict" per batch by heads (TP8 12,
# TP16 6), from GB300 sweeps: early predecessor-independent loads (MMA gate modes only;
# 2 also stages the state rows in shared memory, 3 also the F_B operands), lanes per
# state row, minimum CTAs per SM, CTAs per head (0: defaults) and L2 evict-first state
# traffic (staged only).
SCHEDULE = {12: "16-32:3.16.3.2.1,64:3.16.4.2.1", 6: "16:1.0.0,32-64:3.16.3.2"}
# Verify (q > 1) schedules by request count B, from GB300
# sweeps over q 4-8: no staging; eight-CTA clusters, then fewer as B grows (TP8:
# from B10 two-CTA clusters at two CTAs per SM, one wave where four-CTA clusters
# at three per SM take two: q5 B10-12 15.9-16.3 us instead of 18.9-20.7).
SPEC_SCHEDULE = {
    12: "1:1.8.2.8,2-4:1.8.3.8,5-9:1.8.3.4,10-12:1.8.2.2,13-16:1.8.3.2",
    6: "1:1.8.2.8,2:1.8.0.8,3-12:1.8.3.8,13-16:1.8.3.4",
}


# (request, head) items from which KdaWide (a whole head per CTA, two CTAs per SM)
# beats KdaSplit's cluster-split heads, per step kind, from GB300 sweeps
# (FP32 state and conv, q5; us per layer split / wide):
#   record (RecoverSSM verify)  TP8 R16 15.7 / 17.5, R32 34.6 / 27.5, R128 115 / 83;
#                               TP1 R2 18.4 / 19.6, R4 34.3 / 29.2, R128 841 / 557
#   verify (plain, q state writes)  TP8 R64 73.8 / 76.8, R128 137.7 / 137.2;
#                               TP1 R16 137.9 / 139.8, R64 519 / 506, R128 1028 / 994
#   decode (q = 1)              TP8 R64 23.1 / 27.9, R256 88.3 / 85.8;
#                               TP1 R16 46.8 / 48.5, R64 170.7 / 162.5
WIDE_ITEMS = {"decode": 3072, "verify": 6144, "record": 384}
# (request, head) items from which the persistent KdaStream, then the
# warp-specialized KdaWS take the step, from GB300 sweeps (FP32 state
# and conv; us per layer split / wide / stream / ws):
#   record q5  TP8 R32 34.7 / 27.7 / 30.9 / 30.1, R64 61.4 / 48.2 / 45.9 / 51.7,
#              R128 115 / 82.0 / 84.2 / 68.3; TP1 R4 34.4 / 29.3 / 31.1 / 30.1,
#              R8 59.6 / 49.6 / 46.4 / 50.6, R16 111 / 84.6 / 84.0 / 68.7,
#              R128 840 / 556 / 533 / 401 (q6: TP1 R128 955 / 570 / 604 / 451)
#   decode     TP8 R64 23.1 / 27.9 / 28.1 / 24.4, R256 87.7 / 86.0 / 90.4 / 75.2;
#              TP1 R64 169 / 163 / 167 / 145, R256 665 / 615 / 666 / 564
#   verify     (q state writes) KdaWS and KdaStream lose: TP1 R128 1028 / 995 /
#              1198 / 1211.
NEVER = 1 << 30
STREAM_ITEMS = {"decode": NEVER, "verify": NEVER, "record": 768}
#   review (record q5, stream / ws): TP8 R64 45.9 / 48.2, R72 46.3 / 42.9, R80 57.6 /
#              47.3, R112 72.1 / 56.7; TP1 R8 46.1 / 47.2, R9 46.9 / 42.4, R12 60.1 /
#              51.1, R14 72.3 / 57.1: KdaWS from 864 items.
WS_ITEMS = {"decode": 3072, "verify": NEVER, "record": 864}
# Heads per rank from which RecoverSSM verify steps below WS_ITEMS take KdaStream at
# every batch (DEP's TP1 attention, 96 heads).
STREAM_HEADS = 96
# RecoverSSM verify steps of WS_SMALL to WS_SMALL_END (request, head) items take
# KdaWS too (two or three items per CTA: little steady state to sustain): TP8 R16
# split / wide / stream / ws 16.6 / 17.7 / 17.9 / 16.0, R24 27.4 / 18.2 / 18.9 /
# 16.8, R26 28.0 / 25.7 / - / 22.5, R28 32.0 / 26.5 / - / 22.1,
# R31 34.1 / 27.3 / - / 22.5, R32 34.6 / 27.6 / 30.9 / 27.9; TP1 R2 stream / ws
# 17.2 / 16.2, R3 18.6 / 17.2, R1 11.4 / 15.6. The window ends where KdaWide starts.
WS_SMALL, WS_SMALL_END = 192, 384
# Deferred RecoverSSM (``fused``): a RecoverSSM verify of at least FUSED_ITEMS
# (request, head) items runs KdaFused, the previous step's commit replayed in the
# same pass; smaller steps fold the pending replays first (the deferred commit, every
# layer at the step's first KDA layer) and run the cores below. CTAs per item by
# item count: enough clusters to fill the GPU at small batches (the rows of the
# chunked verify are independent), one CTA per item from FUSED_SPLIT[-1][0] items.
# GB300, FP32 state and conv, q5, AL 4 replayed, us per layer core + FP32 commit
# (69 layers, per layer) / KdaFused / KdaFusedWS: TP8 R1 5.8 / 10.7 / -, R8 16.2 /
# 17.2 / -, R12 18.4 / 18.1 / -, R16 22.9 / 21.3 / 18.7, R32 39.6 / 38.3 / 37.9, R64
# 81.2 / 57.1 / 70.4, R96 97 / 75.4 / 64.1, R128 118 / 97.5 / 80.4; TP1 R1 15.0 /
# 19.1 / -, R2 21.6 / 22.1 / -, R4 43.9 / 39.6 / 38.0, R8 76 / 58.8 / 71.3, R16 124 /
# 96.8 / 80.2, R64 398 / 341 / 253, R128 747 / 667 / 491.
FUSED_ITEMS = {12: 192}
FUSED_DEFAULT_ITEMS = 384
FUSED_SPLIT = ((48, 8), (96, 4), (192, 2), (1 << 30, 1))


# Deferred RecoverSSM steps of at least FUSED_WS_ITEMS items run the persistent
# warp-specialized KdaFusedWS (where its shared memory fits), else KdaFused. Its CTAs
# walk the items strided (CTA c: c, c + ctas, ...); with contiguous ranges a multiple
# of the head count (TP8 R64, TP1 R8: 6 items per CTA) every CTA read the same head's
# F_B rows and the same state page offset at once, and KdaFusedWS lost to KdaFused at
# 640-1023 items. GB300 (152 SMs), FP32 state and conv, q5, 4 replayed, us per layer
# (L2 flushed) KdaFused / contiguous / strided KdaFusedWS: TP8 R16 20.5 / 19.7 / 19.4,
# R32 26.1 / 29.1 / 26.0, R48 34.6 / 38.9 / 34.8, R56 52.6 / 44.3 / 44.6, R62 57.0 /
# 44.6 / 45.6, R64 56.9 / 50.7 / 50.4, R72 58.6 / 50.5 / 50.6, R80 67.2 / 56.4 / 56.1,
# R96 - / 63.3 / 62.4, R128 - / 80.1 / 80.8; TP1 R4 - / 31.1 / 26.4, R8 58.7 / 50.0 /
# 48.7, R16 - / 80.8 / 76.2, R64 - / 254.4 / 247.4, R128 - / 491.2 / 468.9.
FUSED_WS_ITEMS = 192


def fused_split(items):
    return next(split for limit, split in FUSED_SPLIT if items < limit)


def replays(heads, requests, fused):
    """Whether a deferred RecoverSSM verify step runs KdaFused (else: flush + core)."""
    if isinstance(FUSED_ITEMS, int):  # a forced threshold (tests, benchmarks)
        return fused and heads * requests >= FUSED_ITEMS
    items = FUSED_ITEMS.get(heads, FUSED_DEFAULT_ITEMS)
    return fused and heads * requests >= items


def kind_of(tokens, record):
    return "record" if record else "verify" if tokens > 1 else "decode"


def wide(heads, requests, tokens, record):
    return heads * requests >= WIDE_ITEMS[kind_of(tokens, record)]


def stream_layout(specs, tokens) -> tuple[bool, int] | None:
    """(conv dim-first, window) when KdaStream can cp.async this step's operands
    (16-byte aligned rows and slots), else None."""
    (_, x_stride, _), conv = specs[0], specs[3]
    shape, stride, dtype = conv
    size = torch.empty((), dtype=dtype).element_size()
    window = shape[-1]
    if stride[-1] == 1 and stride[-2] == window:
        dim_first = True
    elif stride[-2] == 1:
        dim_first = False
    else:
        return None
    aligned = (stride[0] * size) % 16 == 0 and (x_stride[0] * 2) % 16 == 0
    if not dim_first:
        aligned = aligned and (stride[-1] * size) % 16 == 0
    return (dim_first, window) if aligned and window >= tokens + 2 else None


def stream_ctas(items, device):
    """Persistent CTAs (two per SM) with an equal item count each."""
    slots = 2 * torch.cuda.get_device_properties(device).multi_processor_count
    per = -(-items // slots)
    return -(-items // per)


def core(specs, heads, gate_mode, has_bias, device, tokens, record, fused=False):
    """The KDA core for a step shape: KdaWS, KdaStream, KdaWide or KdaSplit by the
    per-kind (request, head) item thresholds above (DEP's 96-head RecoverSSM
    verify takes KdaStream below KdaWS: TP1 R1 split / wide / stream 14.3 / 13.0 /
    12.2, R2 19.2 / 19.6 / 17.8, R6 47.3 / 37.6 / 33.7 us)."""
    requests = specs[0][0][0] // tokens
    items, kind = heads * requests, kind_of(tokens, record)
    if replays(heads, requests, fused):
        streamed = stream_layout(specs, tokens)
        # q = 2 (k = 1): KdaFusedWS faults (open issue), KdaFused serves it.
        if streamed is not None and items >= FUSED_WS_ITEMS and tokens >= 3:
            conv_size = torch.empty((), dtype=specs[3][2]).element_size()
            rq = specs[15][0][2]  # tokens a record holds (corrections)
            size = fused_ws.shared_bytes(tokens, rq, *streamed, conv_size)
            if size <= fused_ws.SMEM_LIMIT:
                sms = torch.cuda.get_device_properties(device).multi_processor_count
                per = max(2, -(-items // sms))
                ctas = -(-items // per)
                mode = gate_mode % 2
                return KdaFusedWS(heads, tokens, mode, has_bias, *streamed, ctas)
        return KdaFused(heads, tokens, gate_mode % 2, has_bias, fused_split(items))
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    streamed = stream_layout(specs, tokens)
    mode = gate_mode % 2
    small = kind == "record" and WS_SMALL <= items < WS_SMALL_END
    if streamed is not None and (items >= WS_ITEMS[kind] or small):
        conv_size = torch.empty((), dtype=specs[3][2]).element_size()
        if (
            specialized.shared_bytes(tokens, *streamed, conv_size)
            <= specialized.SMEM_LIMIT
        ):
            per = max(2, -(-items // sms))  # items per CTA (one CTA per SM)
            ctas = -(-items // per)
            return KdaWS(heads, tokens, record, mode, has_bias, *streamed, ctas)
    stream = items >= STREAM_ITEMS[kind] or (kind == "record" and heads >= STREAM_HEADS)
    if streamed is not None and stream:
        ctas = stream_ctas(items, device)
        return KdaStream(heads, tokens, record, mode, has_bias, *streamed, ctas)
    if wide(heads, requests, tokens, record):
        return KdaWide(heads, tokens, record, mode, has_bias, 2 * sms)
    table = SCHEDULE if tokens == 1 else SPEC_SCHEDULE
    schedule = batch_table(table.get(heads, ""))
    schedule = schedule.get(requests, (0, 0, 0)) if gate_mode != 2 else (0,) * 3
    if schedule[0] >= 2 and specs[9][1][-2:] != (128, 1):  # staging needs whole rows
        schedule = (1, *schedule[1:4])  # no evict hint without staging
    return KdaSplit(heads, requests, gate_mode, has_bias, schedule, tokens, record)


@cache
def executable(
    specs, heads, gate_mode, has_bias, device, tokens=1, record=False, fused=False
):
    args = tuple(fake(shape, stride, dtype) for shape, stride, dtype in specs)
    scalars = (cutlass.Float32(v) for v in (-5, 128**-0.5, 1e-6))
    kernel = core(specs, heads, gate_mode, has_bias, device, tokens, record, fused)
    with torch.accelerator.device_index(device):
        return cute.compile(
            kernel, *args[:13], *scalars, *args[13:], make_fake_stream()
        )


def deferred_supported(backend, state, lower_bound):
    """Whether the KDA metadata of the specialized decode defers a layer's RecoverSSM
    commit: the layer's RecoverSSM kernels are Triton (the stock FP32-state path) and
    the specialized FP32 commit can fold its checkpoints (FP32 [slot, H, 128, 128] on
    SM10x, a bounded gate). ``recoverssm.context.KDADeferredCommitContext`` takes
    the same decision (its commit backend ``fp32_deferred``)."""
    from .commit import supported

    return backend == "triton" and supported(state, lower_bound)


def decode_conv_weight(weight):
    """The [3, width, D] (Q, K, V) decode layout of the [3 * D, 1, width] conv
    weight."""
    width = weight.shape[-1]
    return weight.detach().view(3, -1, width).transpose(1, 2).contiguous()


class PreparedKda:
    def __init__(self, attention, policies, conv_dim_first):
        bias, dt = attention.conv1d.bias, attention.dt_bias
        self.heads = attention.A_log.numel()
        self.dim, self.has_bias = self.heads * 128, bias is not None
        conv_w = decode_conv_weight(attention.conv1d.weight)
        bias, fb = dt if bias is None else bias, attention.f_b_proj.weight
        norm = attention.o_norm.weight.detach().float()  # BF16 slows the fused kernel
        self.weights = (conv_w, bias, fb, attention.A_log, dt, norm)
        self.device, self.policies = dt.device, policies
        scalars = (attention.gate_lower_bound, 128**-0.5, attention.o_norm.eps)
        self.scalars = tuple(map(cutlass.Float32, scalars))
        self.conv_dim_first, self.kernels = conv_dim_first, {}
        # RecoverSSM: verify steps leave the state checkpoint and write the records.
        records = getattr(attention, "recoverssm_records", None)
        self.records: tuple = (records.correction, records.key_gate) if records else ()
        self.backend = getattr(attention, "recoverssm_backend", None)
        self.lower_bound = attention.gate_lower_bound
        self.fused = False  # set by prepare_caches (deferred_supported)
        self.flushes = set()  # deferred steps that fold the replays first (keys)

    @torch.no_grad()
    def process_weights_after_loading(self, attention):
        """Refill the derived conv weight and FP32 norm after a weight reload."""
        conv_w, *_, norm = self.weights
        conv_w.copy_(decode_conv_weight(attention.conv1d.weight))
        norm.copy_(attention.o_norm.weight)

    def _operands(self, caches, projected, slots, out, accepted=None, deferred=()):
        """Kernel tensors in launch order: 13 before the scalars, then F_B (and the
        verify step's accepted counts and RecoverSSM records; fused: the record rows
        and pending counts)."""
        conv, state = caches[:2]
        conv = conv if self.conv_dim_first else conv.transpose(-1, -2)
        w, bias, fb, a_log, dt, norm = self.weights
        d, h = self.dim, self.heads
        p = lambda start, width: projected[:, start : start + width]
        ops = (p(0, 3 * d), w, bias, conv, p(4 * d, 128), p(4 * d + 128, h), a_log, dt)
        tail = (fb,) if accepted is None else (fb, accepted, *self.records, *deferred)
        return ops + (slots, state, out, p(3 * d, d), norm) + tail

    def prepare_caches(self, caches):
        """Policy keys: a batch, or (requests, tokens) for verify steps."""
        self.flushes = set()
        self.fused = bool(self.records) and deferred_supported(
            self.backend, caches[1], self.lower_bound
        )
        for key, (pitch, mode) in self.policies.items():
            requests, tokens = key if isinstance(key, tuple) else (key, 1)
            batch, meta = requests * tokens, dict(device="meta")
            projected = torch.empty((batch, pitch), dtype=torch.bfloat16, **meta)
            out = torch.empty(batch, self.dim, dtype=torch.bfloat16, **meta)
            accepted = torch.empty(requests, dtype=torch.int32, **meta)
            record = tokens > 1 and bool(self.records)
            # RecoverSSM verify steps have one state slot (the checkpoint).
            shape = (requests, 1 if record else tokens) if tokens > 1 else (batch,)
            slots = torch.empty(shape, dtype=torch.int32, **meta)
            accepted = accepted if tokens > 1 else None
            fused = record and self.fused
            deferred: tuple = ()
            if fused and not replays(self.heads, requests, True):
                self.flushes.add(key)
            if fused:
                rows = torch.empty(requests, dtype=torch.int32, **meta)
                count = self.records[0].shape[0]  # one pending count per record slot
                deferred = (rows, torch.empty(count, dtype=torch.int32, **meta))
            operands = self._operands(caches, projected, slots, out, accepted, deferred)
            self.kernels[key] = executable(
                tuple(map(layout, operands)),
                *(self.heads, mode, self.has_bias, self.device.index, tokens),
                record,
                fused,
            )
        return self

    def run(self, projected, out, *, caches, slots, accepted=None, q=1, deferred=None):
        """``accepted`` [>= B] (verify steps of ``q`` tokens): ``slots`` is [>= B, q]
        contiguous, or [>= B, 1] (the checkpoints) under RecoverSSM. ``deferred``
        (the KDA metadata's ``recoverssm_deferred``): the record rows [>= B] and
        pending counts of a deferred RecoverSSM step."""
        batch = projected.shape[0]
        key, rows = batch, batch
        extra: tuple = ()
        if accepted is not None:
            rows = batch // q
            key, accepted = (rows, q), accepted[:rows]
            if slots.stride() != (slots.shape[1], 1):  # e.g. a block-table slice
                slots = slots.contiguous()
            if self.records and self.fused:
                assert deferred is not None, "a deferred KDA core needs its metadata"
                extra = (deferred.record_rows[:rows], deferred.pending)
                if key in self.flushes:
                    deferred.flush()  # this core verifies from committed checkpoints
        tensors = self._operands(caches, projected, slots[:rows], out, accepted, extra)
        views = tuple(to_cute(t, 16) for t in tensors)
        self.kernels[key](*views[:13], *self.scalars, *views[13:], stream(self.device))
        return out
