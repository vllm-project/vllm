# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that a decoder keeps serving while one of its p2p sources faults.

Two prefillers act as p2p sources and one decoder as the consumer. A probe
seeds a fresh prefix on each source and then asks the decoder to pull it
from that source with a ``remote_kv_source`` block, twice a second per source.
A probe "pulled" when the decoder reports cached prompt tokens for it: the
prefix is unique, so they can only have come over the p2p tier.

FAULT_CASE picks what happens to source 0 after a healthy baseline:
    stall   SIGSTOP its processes for STALL_S, then SIGCONT
    kill    SIGKILL them, then restart it on the same ports
    remove  SIGTERM them and leave it down

The check asserts time bounds and recovery, not accuracy:
    - every decoder request returns 200
    - requests naming the faulted source return within FAULTED_MAX_S, and
      those sent once the fault has been detected within FAULTED_P50_S at the
      median: the decoder prefills locally instead of waiting on the source
    - requests naming the healthy source keep pulling with normal latency,
      so the fault does not freeze the decoder's p2p control plane
    - after SIGCONT or the restart, requests naming source 0 pull again

Driven by run_accuracy_test.sh via env vars:
    TEST_MODEL, FAULT_CASE, DECODE_URL, FAULT_SOURCES (JSON list of
    {http, host, port, sid, cmd} for source 0 and 1), FAULT_SID_FILE (where
    a restarted source's session id is appended for the driver's cleanup).
"""

import json
import os
import random
import statistics
import subprocess
import threading
import time
import uuid

import requests

MODEL_NAME = os.environ.get("TEST_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
FAULT_CASE = os.environ.get("FAULT_CASE", "stall")
DECODE_URL = os.environ.get("DECODE_URL", "http://localhost:8200")
SOURCES = json.loads(os.environ.get("FAULT_SOURCES", "[]"))
SID_FILE = os.environ.get("FAULT_SID_FILE", "")

PROBE_INTERVAL_S = 0.5
PROMPT_WORDS = 600
# A source that is stopped never answers the seed; send the pull regardless.
SEED_TIMEOUT_S = 3.0
# Well past VLLM_P2P_HANDSHAKE_TIMEOUT_S (30 s), so a request that waits on
# the faulted source shows up as a slow 200 rather than a client timeout.
PULL_TIMEOUT_S = 60.0
# Give the source time to store the seeded blocks before the pull.
SEED_TO_PULL_S = 0.3

BASELINE_S = 10.0
STALL_S = 30.0
REMOVE_OBSERVE_S = 30.0
RECOVERY_S = 40.0
RESTART_READY_TIMEOUT_S = 900.0

# A request that is in flight when the fault hits ends on the tier's 8 s
# lookup timeout or the 10 s control heartbeat, whichever fires first.
FAULTED_MAX_S = 15.0
# Once the heartbeat has caught the fault, a request naming the faulted
# source must not wait on it at all.
FAULT_DETECTED_AFTER_S = 12.0
FAULTED_P50_S = 2.0
HEALTHY_MAX_S = 5.0
HEALTHY_MIN_PULLED = 0.9
BASELINE_MIN_PULLED = 0.5
# Measured from RECOVERY_GRACE_S after SIGCONT or the restart.
RECOVERY_GRACE_S = 15.0
RECOVERY_MIN_PULLED = 0.5

WORDS = [
    "apple",
    "river",
    "cloud",
    "stone",
    "forest",
    "lamp",
    "ocean",
    "garden",
    "silver",
    "mountain",
    "candle",
    "harbor",
    "meadow",
    "lantern",
    "valley",
    "copper",
    "willow",
    "thunder",
    "desert",
    "violet",
]


def _prompt(salt: str) -> str:
    rng = random.Random(salt)
    return salt + " " + " ".join(rng.choice(WORDS) for _ in range(PROMPT_WORDS))


def _completion(base: str, prompt: str, timeout: float, **extra) -> dict:
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{base}/v1/completions",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                **extra,
            },
            timeout=timeout,
        )
        status = resp.status_code
        body = resp.json() if status == 200 else {}
    except requests.RequestException as exc:
        status, body = -1, {"error": repr(exc)[:200]}
    return {"status": status, "lat": time.monotonic() - t0, "body": body}


class Probe:
    """Seeds and pulls a fresh prefix per source every PROBE_INTERVAL_S."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._loop = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._loop.start()

    def stop(self) -> None:
        self._stop.set()
        self._loop.join()
        for th in self._threads:
            th.join(PULL_TIMEOUT_S + SEED_TIMEOUT_S + 5)

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            for idx in range(len(SOURCES)):
                th = threading.Thread(target=self._one, args=(idx,), daemon=True)
                th.start()
                self._threads.append(th)
            next_tick += PROBE_INTERVAL_S
            self._stop.wait(max(0.0, next_tick - time.monotonic()))

    def _one(self, idx: int) -> None:
        src = SOURCES[idx]
        prompt = _prompt(f"{idx}-{uuid.uuid4().hex}")
        seed = _completion(src["http"], prompt, SEED_TIMEOUT_S)
        time.sleep(SEED_TO_PULL_S)
        t_pull = time.monotonic()
        pull = _completion(
            DECODE_URL,
            prompt,
            PULL_TIMEOUT_S,
            kv_transfer_params={
                "remote_kv_source": {
                    "kv_request_id": uuid.uuid4().hex,
                    "remote_host": src["host"],
                    "remote_port": src["port"],
                }
            },
        )
        details = (pull["body"].get("usage") or {}).get("prompt_tokens_details")
        cached = (details or {}).get("cached_tokens") or 0
        with self._lock:
            self.records.append(
                {
                    "src": idx,
                    "t_pull": t_pull,
                    "seed_status": seed["status"],
                    "status": pull["status"],
                    "lat": pull["lat"],
                    "pulled": cached > 0,
                }
            )


def _signal_source(sig: str) -> None:
    subprocess.run(["pkill", f"-{sig}", "-s", str(SOURCES[0]["sid"])], check=False)


def _wait_ready(src: dict) -> None:
    deadline = time.monotonic() + RESTART_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{src['http']}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise AssertionError(f"restarted source {src['http']} never became ready")


def _restart_source() -> None:
    src = SOURCES[0]
    # Wait for the killed session to exit so the restart can bind its ports.
    for _ in range(60):
        if subprocess.run(["pgrep", "-s", str(src["sid"])]).returncode != 0:
            break
        time.sleep(1)
    proc = subprocess.Popen(["bash", "-c", src["cmd"]], start_new_session=True)
    if SID_FILE:
        with open(SID_FILE, "a") as f:
            f.write(f"{proc.pid}\n")
    _wait_ready(src)


def _window(records, src, start, end=float("inf")):
    return [r for r in records if r["src"] == src and start <= r["t_pull"] < end]


def _pulled_fraction(window) -> float:
    return sum(r["pulled"] for r in window) / len(window) if window else 0.0


def _describe(name, window) -> str:
    if not window:
        return f"{name}: no probes"
    lats = [r["lat"] for r in window]
    return (
        f"{name}: n={len(window)} pulled={sum(r['pulled'] for r in window)} "
        f"p50={statistics.median(lats):.2f}s max={max(lats):.2f}s"
    )


def test_decoder_survives_source_fault():
    assert FAULT_CASE in ("stall", "kill", "remove"), FAULT_CASE
    assert len(SOURCES) == 2, "needs a faulted source and a healthy source"

    probe = Probe()
    t_start = time.monotonic()
    probe.start()
    time.sleep(BASELINE_S)

    t_fault = time.monotonic()
    t_recover = None
    if FAULT_CASE == "stall":
        _signal_source("STOP")
        time.sleep(STALL_S)
        _signal_source("CONT")
        t_recover = time.monotonic()
    elif FAULT_CASE == "kill":
        _signal_source("KILL")
        _restart_source()
        t_recover = time.monotonic()
    else:
        _signal_source("TERM")
        time.sleep(REMOVE_OBSERVE_S)
    if t_recover is not None:
        time.sleep(RECOVERY_S)
    probe.stop()

    records = probe.records
    fault_end = t_recover if t_recover is not None else float("inf")
    windows = {
        "baseline faulted": _window(records, 0, t_start, t_fault),
        "baseline healthy": _window(records, 1, t_start, t_fault),
        "fault faulted": _window(records, 0, t_fault, fault_end),
        "fault healthy": _window(records, 1, t_fault, fault_end),
    }
    if t_recover is not None:
        windows["recovery faulted"] = _window(records, 0, t_recover + RECOVERY_GRACE_S)
    print(f"\nFAULT_CASE={FAULT_CASE}")
    for name, window in windows.items():
        print("  " + _describe(name, window))

    errors = [r for r in records if r["status"] != 200]
    assert not errors, f"{len(errors)} decoder requests failed, e.g. {errors[0]}"

    for name in ("baseline faulted", "baseline healthy"):
        assert _pulled_fraction(windows[name]) >= BASELINE_MIN_PULLED, (
            f"{name}: the probe does not pull over p2p even before the fault"
        )

    faulted = windows["fault faulted"]
    assert faulted, "no probes named the faulted source during the fault"
    worst = max(r["lat"] for r in faulted)
    assert worst <= FAULTED_MAX_S, (
        f"a request naming the faulted source took {worst:.1f}s "
        f"(bound {FAULTED_MAX_S}s)"
    )
    detected = [
        r["lat"] for r in faulted if r["t_pull"] >= t_fault + FAULT_DETECTED_AFTER_S
    ]
    assert detected, "the fault window is too short to measure detection"
    p50 = statistics.median(detected)
    assert p50 <= FAULTED_P50_S, (
        f"once the fault was detected, requests naming the faulted source "
        f"still took p50 {p50:.1f}s (bound {FAULTED_P50_S}s)"
    )

    healthy = windows["fault healthy"]
    assert healthy, "no probes named the healthy source during the fault"
    worst = max(r["lat"] for r in healthy)
    assert worst <= HEALTHY_MAX_S, (
        f"a pull from the healthy source took {worst:.1f}s during the fault "
        f"(bound {HEALTHY_MAX_S}s): the fault stalled the decoder"
    )
    assert _pulled_fraction(healthy) >= HEALTHY_MIN_PULLED, (
        "pulls from the healthy source stopped hitting during the fault"
    )

    if t_recover is not None:
        recovered = windows["recovery faulted"]
        assert _pulled_fraction(recovered) >= RECOVERY_MIN_PULLED, (
            "pulls from the faulted source did not resume after it came back"
        )
