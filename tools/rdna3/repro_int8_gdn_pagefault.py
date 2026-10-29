#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic reproducer for the RDNA3 INT8-KV + hybrid-GDN GPU page fault.

Symptom (prod and reproduced):
    ``Memory access fault by GPU node-N ... Page not present`` ->
    ``Worker proc VllmWorker-* died unexpectedly`` -> EngineCore ``cancelled``.
    Pure decode, ``kv_cache_dtype=int8_per_token_head``, hybrid GDN model
    (Qwen3.6), ``--enable-prefix-caching``, RDNA3 (gfx1100), TP2/TP4.

ROOT-CAUSE NEIGHBOURHOOD:
    The per-request GDN state-slot index ``non_spec_state_indices_tensor =
    block_table_tensor[:, 0]`` (gdn_attn.py) is consumed WITHOUT an upper-bound
    check by the decode fast-path writes ``causal_conv1d_update(...,
    validate_data=False)`` and ``fused_recurrent_gated_delta_rule_packed_decode``
    (qwen_gdn_linear_attn.py). For a heterogeneous (mixed context-length) decode
    batch a bad/edge-case index escapes into a streaming OOB write of the conv /
    SSM state buffer -> the RW:0x1 / TCP consecutive-page page fault.

WHY THIS RECIPE AND NOT A SIMPLER ONE:
    The trigger is a decode batch of *mixed* context lengths, NOT one huge
    context and NOT uniform identical requests. A uniform single huge prefix
    resumed identically (even 50+ times, even at 80k tokens) does NOT crash:
    the block-table layout is identical every step, so a valid layout stays
    valid. Reproduction REQUIRES several requests at *different* resumed
    context lengths co-decoding in the same batch. This script sweeps many
    prefix lengths round-robin so the scheduler co-batches heterogeneous
    decodes; it faults within ~20-25 successful requests.

CONFIRMED (test box vllm-vllm1-1, TP2, build 1037a9b5a, unpatched baseline):
    crash dump ``num_computed_tokens=[7862, 15515, 23459, 31547]``,
    ``num_scheduled_tokens=1`` each, ``num_common_prefix_blocks=[0,0,0,5]``,
    ``kv_cache_usage=0.078`` (NOT an OOM), same ``step_with_batch_queue ->
    RuntimeError: cancelled`` traceback as the production crash.

USAGE (run inside the serving container; the API must be the vLLM port, not a
reverse proxy):
    python repro_int8_gdn_pagefault.py [duration_s] [concurrency]
    # env overrides:
    #   REPRO_URL     (default http://localhost:8000/v1/chat/completions)
    #   REPRO_MODEL   (default INCCODER)
    #   REPRO_PROMPT  (default /tmp/haystack.txt) -- any large text file used
    #                 as the shared, cache-hit prefix material
Watch ``/health`` (from inside the container) and the engine log: a drop to a
non-200 / connection-refused plus ``Memory access fault by GPU`` in the log
confirms the fault. The engine dies; the server must be restarted afterwards.
"""
import json
import os
import random
import sys
import threading
import time
import urllib.request

URL = os.environ.get("REPRO_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("REPRO_MODEL", "INCCODER")
PROMPT_FILE = os.environ.get("REPRO_PROMPT", "/tmp/haystack.txt")

DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 4

# Mixed context lengths (tokens ~= chars / 4). Heterogeneity is the trigger.
LENS = [32000, 64000, 96000, 128000, 160000, 200000, 256000, 304000, 352000, 400000]

BIG = open(PROMPT_FILE, encoding="utf-8", errors="ignore").read()
while len(BIG) < max(LENS):
    BIG += BIG

res = {"ok": 0, "err": 0}
lock = threading.Lock()
stop = [False]


def fire(pc: int, i: int) -> None:
    prefix = BIG[:pc]
    # Identical prefix per length -> prefix-cache hit -> resume path.
    # Variable tail -> distinct trailing blocks each time.
    tail = ("paso %d. " % i) + ("analiza. " * random.randint(1, 40))
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "It %d ctx%d. %s siguiente." % (i, pc // 4, tail)},
        ],
        "max_tokens": 4,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.3,
        "repetition_penalty": 1.1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        req = urllib.request.Request(
            URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=400).read()
        with lock:
            res["ok"] += 1
    except Exception as e:  # noqa: BLE001
        with lock:
            res["err"] += 1
        print("[ERR ctx%d it%d] %s:%s" % (pc // 4, i, type(e).__name__, str(e)[:70]),
              flush=True)


def health() -> None:
    base = URL.rsplit("/v1/", 1)[0]
    while not stop[0]:
        try:
            urllib.request.urlopen(base + "/health", timeout=6).read()
        except Exception as e:  # noqa: BLE001
            print("!!! SERVER DEAD %s" % type(e).__name__, flush=True)
            return
        time.sleep(1)


def main() -> None:
    print("VARIED dur=%ds conc=%d lens=%s" % (DUR, CONC, [x // 4 for x in LENS]),
          flush=True)
    threading.Thread(target=health, daemon=True).start()
    t_end = time.time() + DUR
    i = 0
    sem = threading.Semaphore(CONC)
    live = []

    def worker(pc, idx):
        with sem:
            fire(pc, idx)

    while time.time() < t_end and not stop[0]:
        pc = LENS[i % len(LENS)]
        t = threading.Thread(target=worker, args=(pc, i))
        t.start()
        live = [x for x in live if x.is_alive()] + [t]
        i += 1
        if i % 25 == 0:
            print("fired=%d ok=%d err=%d t=%ds"
                  % (i, res["ok"], res["err"], int(time.time() - (t_end - DUR))),
                  flush=True)
        sem.acquire()
        sem.release()
        time.sleep(0.05)
    for t in live:
        t.join()
    stop[0] = True
    print("DONE fired=%d ok=%d err=%d" % (i, res["ok"], res["err"]), flush=True)


if __name__ == "__main__":
    main()
