# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The all-reduce protocol of the folded all-reduces (kernels/wob_ar.py,
kernels/comb_ar.py) across the 4 TP ranks: reduce-scatter + all-gather, pushed
over xGMI, with an MXFP8 wire.

- The payload is 4 quarters; rank q reduces quarter q.
- Hop 1: push quarter d of my partial to rank d's landing slot R (sc1 stores)
  as MXFP8 (e4m3 codes, one E8M0 per 32 elements, vLLM's ceil rule), vmcnt(0),
  flag. Poll the 3 peers' flags, sum my own exact quarter and the 3 dequantised
  ones in fp32 in rank order.
- Hop 2: quantise the sum once, push codes + scales to every peer's ``land2``
  slot R, flag; every rank (the owner too) writes bf16(dequant) of each quarter
  into its own ``out``, so all ranks hold the same values.
- Seq numbers live in device memory: a CTA (or wave) reads its ctr, uses ctr + 1
  as this launch's flag value and writes it back, so graph replays need no host
  input. Every rank runs the same launches, so the counters agree.
- No double buffering: a peer writes my landing chunk for launch s + 1 only
  after it saw my flags for launch s + 1, i.e. after I read the landing of
  launch s.
- Stores to peers sc1, polls sc1; plain stores to a peer stay in our L2.

Landing slot layout: codes at 8·granule (16 B granules), scales at SCALE_OFF +
granule / 4.
"""

H, MAX_M = 5120, 256
QMAX = MAX_M * H * 2 // 4  # bytes of a quarter (bf16) at m = 256
SCALE_OFF = QMAX // 2  # scale bytes of a landing slot (codes take ≤ QMAX / 2)
# ~minutes (2^30 ≈ 144 s of L2-hit polls): a late peer (eager mode) must not turn
# into stale data
POLLS = 1 << 30
