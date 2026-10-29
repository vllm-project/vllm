#!/usr/bin/env python3
"""
Liveness guard for the po-2023 embeddings + STT chain.

WHY THIS EXISTS
---------------
Three RAG outages in three days (2026-10-27, 2026-10-29 09:21Z, 2026-10-29 16:30Z)
plus a fourth on 2026-10-29 evening. Each one took the whole fleet's retrieval
down (7 Open-WebUI tenants, EPF students included) and none of them was caught
by an existing instrument:

  * the container healthcheck only looks at the container;
  * the fleet guard (#220) covers "engine DOWN", not "relay dead while engine up";
  * a TCP port that accepts proves nothing. On 2026-10-27 `:8004` accepted the
    connection and returned no HTTP at all.

The 2026-10-29 episode is the textbook case: the chain went 502 (facade alive,
backend dead) -> 500 (container back, internal error) -> 401 (someone sending a
fingerprint instead of a key). A port probe sees nothing through all three.

WHAT IS CHECKED
---------------
1. Docker engine reachable (discriminates "engine dead" from "service dead").
2. Embeddings through the FLEET path :8004 -> a REAL POST /v1/embeddings with
   the key; requires HTTP 200, `model == qwen3-4b-awq-embedding` and
   `len(data[0].embedding) == 2560` (the stable contract, vllm issue #71).
3. Embeddings direct on the engine :8005 -> same request, to localise the fault
   (sidecar dead vs engine dead). Does not raise on its own.
4. STT (whisper-webui) -> `/health` must be 200 AND `/v1/models` must answer any
   of {200, 401, 403}. Those three statuses prove the FastAPI app is alive and
   routing; the episode's 502/500 would fail this, and the keyless 401 is
   precisely "alive and enforcing auth".

DELIBERATE NON-GOAL: no real whisper transcription.
`whisper-webui` is pinned to GPU 1 (`DeviceIDs: ["1"]`), which belongs to another
lane. Firing a transcription would load a model on that GPU. The HTTP-level
check catches the class that actually bit us (502/500) without touching GPU 1.
An operator who accepts the GPU cost can pass --deep-whisper.

ESCALATION LADDER (least invasive first)
----------------------------------------
  attempt 1 fails -> retry after RETRY_DELAY (absorbs a transient / a warm-up)
  still failing   -> `docker restart` of the implicated container only,
                     sidecar before engine, then wait for healthy and re-probe
  still failing   -> loud FAIL line + exit 1 (alert; nothing else is touched)

Safety rails, because an operator window is not an outage:
  * Engine unreachable AND the Docker Desktop process is not running -> someone
    stopped it deliberately: SKIP, no action.
  * Engine unreachable AND Docker Desktop is running -> this is the "bricked
    Docker" case covered by the user mandate of 2026-10-24: one
    `docker desktop restart`, loudly logged.
  * Per-container restart cap (default 2/hour, state file) so a persistent fault
    cannot turn into a restart storm.

Usage
-----
    python embedding_liveness_guard.py              # one pass (cron / Task Scheduler)
    python embedding_liveness_guard.py --dry-run    # probe + report, never restart
    python embedding_liveness_guard.py --deep-whisper
    python embedding_liveness_guard.py --loop 300   # self-paced loop, seconds

Exit codes: 0 = healthy or skipped, 1 = degraded after escalation, 2 = engine down.

Secrets: the key is read from the .env and never printed, logged or echoed.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_ENV = r"D:\Production\Embeddings\compose\.env"
DEFAULT_LOG = r"D:\Production\Embeddings\logs\liveness-guard.jsonl"
DEFAULT_STATE = r"D:\Production\Embeddings\logs\liveness-guard.state.json"

EMB_MODEL = "qwen3-4b-awq-embedding"
EMB_DIMS = 2560
SIDECAR_URL = os.environ.get("EMB_GUARD_SIDECAR_URL", "http://127.0.0.1:8004/v1/embeddings")
ENGINE_URL = os.environ.get("EMB_GUARD_ENGINE_URL", "http://127.0.0.1:8005/v1/embeddings")
SIDECAR_CONTAINER = "embedding-proxy-logger"
ENGINE_CONTAINER = "qwen3_4b_awq_embedding_server"
WHISPER_HEALTH = os.environ.get("EMB_GUARD_WHISPER_HEALTH", "http://127.0.0.1:8190/health")
WHISPER_MODELS = os.environ.get("EMB_GUARD_WHISPER_MODELS", "http://127.0.0.1:8190/v1/models")
WHISPER_CONTAINER = "whisper-webui"
WHISPER_ALIVE = {200, 401, 403}

RETRY_DELAY = 3.0
HEALTH_TIMEOUT = 90
RESTART_CAP_PER_HOUR = 2
BOOT_RAIL = 300  # seconds; a container younger than this is booting, not broken


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_key(path):
    """Read VLLM_API_KEY from the .env. Value is never returned in any log."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("VLLM_API_KEY") and "=" in s:
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def http_json(url, body=None, key=None, timeout=30):
    """Return (status, parsed_json_or_None, error_text_or_None)."""
    headers = {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw), None
            except ValueError:
                return resp.status, None, "non-JSON body"
    except urllib.error.HTTPError as exc:
        return exc.code, None, "HTTP %s" % exc.code
    except Exception as exc:  # connection refused, reset, timeout, DNS...
        return None, None, "%s: %s" % (type(exc).__name__, exc)


def probe_embeddings(url, key, tag):
    """One real POST. Returns a dict; 'ok' is the only thing callers branch on."""
    nonce = "%s|%s|%d" % (tag, os.getpid(), time.time_ns())
    status, payload, err = http_json(
        url, body={"model": EMB_MODEL, "input": "sonde de vivacite %s" % nonce}, key=key
    )
    rec = {"check": "embeddings", "path": tag, "status": status, "ok": False}
    if status != 200:
        rec["error"] = err or "status %s" % status
        return rec
    try:
        vec = payload["data"][0]["embedding"]
        model = payload.get("model")
    except (KeyError, IndexError, TypeError):
        rec["error"] = "malformed response"
        return rec
    rec["model"] = model
    rec["dims"] = len(vec)
    if model != EMB_MODEL:
        rec["error"] = "model mismatch"
    elif len(vec) != EMB_DIMS:
        rec["error"] = "dims mismatch (contract is %d)" % EMB_DIMS
    else:
        rec["ok"] = True
    return rec


def probe_whisper(deep, key=None):
    status, _, err = http_json(WHISPER_HEALTH, timeout=15)
    models_status, _, models_err = http_json(WHISPER_MODELS, key=key, timeout=15)
    rec = {
        "check": "whisper",
        "health_status": status,
        "models_status": models_status,
        "ok": status == 200 and models_status in WHISPER_ALIVE,
    }
    if not rec["ok"]:
        rec["error"] = "; ".join(x for x in (err, models_err) if x) or "status pair"
    rec["deep"] = False
    if deep:
        # Explicitly opt-in only: loads a model on GPU 1, which is another lane's.
        rec["deep"] = True
        rec["deep_note"] = "transcription skipped: no audio fixture configured"
    return rec


def engine_status():
    try:
        out = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip(), None
        return None, (out.stderr or "").strip()[:200] or "rc=%s" % out.returncode
    except Exception as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def docker_desktop_running():
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Docker Desktop.exe", "/NH"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        return "Docker Desktop.exe" in (out.stdout or "")
    except Exception:
        return True  # unknown -> do not claim a deliberate stop


def container_state(name):
    """(status, health). health is 'none' when the container defines no healthcheck
    (the sidecar has none), which is NOT an error - a bare template on
    .State.Health.Status makes docker inspect fail and reads as 'absent'."""
    try:
        out = subprocess.run(
            ["docker", "inspect", name, "--format",
             "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if out.returncode != 0:
            return None, "absent"
        parts = out.stdout.strip().split("|")
        return parts[0], (parts[1] if len(parts) > 1 else "none")
    except Exception as exc:
        return None, "%s" % type(exc).__name__


def age_seconds(name):
    """Seconds since the container started, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["docker", "inspect", name, "--format", "{{.State.StartedAt}}"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if out.returncode != 0:
            return None
        raw = out.stdout.strip()
        # Go emits nanosecond precision; python only parses microseconds.
        if "." in raw:
            head, _, tail = raw.partition(".")
            tz = "Z" if tail.endswith("Z") else ""
            raw = head + "." + tail.rstrip("Z")[:6] + tz
        started = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - started).total_seconds()
    except Exception:
        return None


def wait_healthy(name, timeout=HEALTH_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, health = container_state(name)
        if status == "running" and health in ("healthy", "none", ""):
            return True
        time.sleep(3)
    return False


class RestartBudget:
    """Cap restarts per container so a persistent fault cannot become a storm."""

    def __init__(self, path, cap=RESTART_CAP_PER_HOUR):
        self.path = path
        self.cap = cap
        self.data = {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.data = json.load(fh)
        except (OSError, ValueError):
            self.data = {}

    def allow(self, name):
        hist = [t for t in self.data.get(name, []) if time.time() - t < 3600]
        self.data[name] = hist
        return len(hist) < self.cap

    def record(self, name):
        self.data.setdefault(name, []).append(time.time())

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh)
        except OSError:
            pass


def restart_container(name, budget, dry_run):
    if dry_run:
        return "would restart %s (dry-run)" % name
    # Boot rail: same lesson as the vLLM watchdog v5 - a cold boot is not a
    # fault, and restarting into it turns an operator window into a loop.
    # Docker Desktop relaunches the stack on daemon start; the embedder needs
    # ~1-2 min to answer. Never touch a container younger than BOOT_RAIL.
    age = age_seconds(name)
    if age is not None and age < BOOT_RAIL:
        return "SKIP restart of %s: started %ds ago (< %ds boot rail)" % (name, age, BOOT_RAIL)
    status, _ = container_state(name)
    if status in ("created", "restarting"):
        return "SKIP restart of %s: state=%s (boot in progress)" % (name, status)
    if not budget.allow(name):
        return "restart cap reached for %s this hour" % name
    budget.record(name)
    budget.save()
    try:
        out = subprocess.run(
            ["docker", "restart", name], capture_output=True, text=True, timeout=180
        )
        if out.returncode != 0:
            return "docker restart %s failed: %s" % (name, (out.stderr or "").strip()[:160])
    except Exception as exc:
        return "docker restart %s raised %s" % (name, type(exc).__name__)
    ok = wait_healthy(name)
    return "%s restarted (%s)" % (name, "healthy" if ok else "NOT healthy yet")


def log_record(path, record):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def one_pass(args):
    key = read_key(args.env)
    result = {"ts": now(), "actions": [], "checks": [], "verdict": "UNKNOWN"}

    if not key:
        result["verdict"] = "NO_KEY"
        result["checks"].append({"check": "config", "ok": False, "error": "VLLM_API_KEY not found in %s" % args.env})
        return result

    version, engine_err = engine_status()
    if not version:
        if not docker_desktop_running():
            result["verdict"] = "SKIP_ENGINE_STOPPED"
            result["checks"].append(
                {"check": "engine", "ok": False, "skipped": True,
                 "note": "Docker Desktop not running -> deliberate stop, no action"}
            )
            return result
        result["actions"].append("engine mute with Docker Desktop running -> mandate 2026-10-24: docker desktop restart")
        if not args.dry_run:
            try:
                subprocess.run(["docker", "desktop", "restart"], capture_output=True, timeout=300)
            except Exception as exc:
                result["actions"].append("docker desktop restart raised %s" % type(exc).__name__)
            for _ in range(40):
                time.sleep(5)
                version, engine_err = engine_status()
                if version:
                    break
        if not version:
            result["verdict"] = "ENGINE_DOWN"
            result["checks"].append({"check": "engine", "ok": False, "error": engine_err})
            return result
    result["checks"].append({"check": "engine", "ok": True, "version": version})
    result["engine_version"] = version

    # --- embeddings, fleet path then direct engine -------------------------
    fleet = probe_embeddings(SIDECAR_URL, key, "sidecar")
    if not fleet["ok"]:
        time.sleep(RETRY_DELAY)
        fleet = probe_embeddings(SIDECAR_URL, key, "sidecar")

    direct = None
    if not fleet["ok"]:
        direct = probe_embeddings(ENGINE_URL, key, "engine")

    if not fleet["ok"]:
        # Localise before acting: never restart the engine for a sidecar fault.
        if fleet.get("status") in (401, 403):
            # 2026-10-29 lesson: a fingerprint sent as a bearer renders a
            # misleading 401. A restart fixes nothing here - alert only.
            result["actions"].append(
                "AUTH class (401/403): restart is futile, no action taken - check the key in %s" % args.env
            )
        else:
            if direct is not None and direct["ok"]:
                targets = [SIDECAR_CONTAINER]          # engine fine -> relay dead
            else:
                targets = [SIDECAR_CONTAINER, ENGINE_CONTAINER]
            for name in targets:
                result["actions"].append(restart_container(name, args.budget, args.dry_run))
            if not args.dry_run:
                time.sleep(5)
                fleet = probe_embeddings(SIDECAR_URL, key, "sidecar")

    fleet["ok_after_escalation"] = fleet["ok"]
    result["checks"].append(fleet)
    if direct is not None:
        result["checks"].append(direct)

    # --- whisper, HTTP level only (GPU 1 is another lane) ------------------
    whisper = probe_whisper(args.deep_whisper, key)
    if not whisper["ok"]:
        time.sleep(RETRY_DELAY)
        whisper = probe_whisper(args.deep_whisper, key)
    if not whisper["ok"]:
        # Only act if the app answers nothing at all; a 502/500 means the
        # container needs a restart, a dead port means the same.
        result["actions"].append(restart_container(WHISPER_CONTAINER, args.budget, args.dry_run))
        if not args.dry_run:
            time.sleep(5)
            whisper = probe_whisper(args.deep_whisper, key)
    result["checks"].append(whisper)

    failures = [c for c in result["checks"] if c.get("check") in ("embeddings", "whisper") and not c.get("ok")]
    result["verdict"] = "OK" if not failures else "DEGRADED"
    result["failures"] = [c.get("check") + ":" + (c.get("error") or "?") for c in failures]
    return result


def main():
    ap = argparse.ArgumentParser(description="Liveness guard for the po-2023 embeddings + STT chain")
    ap.add_argument("--env", default=DEFAULT_ENV, help="path to the embeddings .env")
    ap.add_argument("--log", default=DEFAULT_LOG, help="JSONL log path")
    ap.add_argument("--state", default=DEFAULT_STATE, help="restart-budget state file")
    ap.add_argument("--dry-run", action="store_true", help="probe and report, never restart")
    ap.add_argument("--deep-whisper", action="store_true", help="reserved: real transcription (loads GPU 1)")
    ap.add_argument("--loop", type=int, default=0, help="repeat every N seconds (0 = single pass)")
    args = ap.parse_args()
    args.budget = RestartBudget(args.state)

    while True:
        res = one_pass(args)
        log_record(args.log, res)

        stamp = res["ts"]
        if res["verdict"] in ("OK", "SKIP_ENGINE_STOPPED"):
            print("[%s] LIVENESS %s" % (stamp, res["verdict"]))
        else:
            # Greppable, loud, one line - this is what an operator watches for.
            print("[%s] LIVENESS FAIL %s :: %s" % (stamp, res["verdict"], "; ".join(res.get("failures") or [res["verdict"]])))
        for c in res["checks"]:
            if c["check"] == "embeddings":
                print("  embeddings %-8s status=%s model=%s dims=%s ok=%s %s" % (
                    c.get("path"), c.get("status"), c.get("model"), c.get("dims"), c.get("ok"), c.get("error", "")))
            elif c["check"] == "whisper":
                print("  whisper    health=%s models=%s ok=%s %s" % (
                    c.get("health_status"), c.get("models_status"), c.get("ok"), c.get("error", "")))
            else:
                print("  %-10s ok=%s %s" % (c["check"], c.get("ok"), c.get("error") or c.get("version") or c.get("note") or ""))
        for a in res["actions"]:
            print("  ACTION %s" % a)

        if not args.loop:
            return 0 if res["verdict"] in ("OK", "SKIP_ENGINE_STOPPED") else (2 if res["verdict"] == "ENGINE_DOWN" else 1)
        time.sleep(args.loop)


if __name__ == "__main__":
    sys.exit(main())
