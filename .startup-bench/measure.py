"""Frozen measurement harness for vLLM cold-start optimization.

See CLAUDE.local.md § "Startup optimization experiments" for the contract this
harness implements. Do NOT modify without explicit user approval — this is the
ground-truth metric. If an optimization requires changing the measurement,
something has gone wrong.

Invocation:
    .venv/bin/python .startup-bench/measure.py --tag <tag> --config qwen-0.5b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.path.expanduser("~"))
TORCHINDUCTOR_CACHE = Path(f"/tmp/torchinductor_simon_{os.getuid()}")

COLD_CACHE_PATHS = [
    TORCHINDUCTOR_CACHE,
    HOME / ".cache" / "vllm",
    HOME / ".cache" / "triton",
    HOME / ".cache" / "vllm-fastboot",
]

CONFIGS = {
    "qwen-0.5b": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "extra_args": [],
    },
    "qwen-7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "extra_args": [],
    },
    "qwen-32b": {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "extra_args": [],
    },
}

SAMPLE_TIMEOUT_S = 300.0
EXPERIMENT_TIMEOUT_S = 1800.0
HEALTH_POLL_S = 0.1
CHAT_TIMEOUT_S = 60.0
NOISY_THRESHOLD = 0.15
NUM_SAMPLES = 3


@dataclass
class SampleResult:
    kind: str  # "cold" | "warm"
    t_ready_s: float | None
    peak_vram_mib: float
    correctness_ok: bool
    error: str | None = None
    server_log_path: str | None = None


@dataclass
class ExperimentResult:
    tag: str
    config: str
    gpu_id: int
    samples: list[SampleResult] = field(default_factory=list)
    status: str = "ok"
    reason: str = ""


def log(msg: str) -> None:
    print(f"[measure] {msg}", flush=True)


def pick_free_gpu() -> int:
    """Pick the first GPU with 0 MiB used. Raises if none free."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    for i, line in enumerate(out.strip().splitlines()):
        if int(line.strip()) == 0:
            return i
    raise RuntimeError("no GPU with 0 MiB used — try again later")


def pids_in_group(pgid: int) -> set[int]:
    """Return all PIDs in the given process group. pgrep exits non-zero if
    none found, which we treat as empty."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-g", str(pgid)], text=True, timeout=2
        )
        return {int(p) for p in out.split() if p.strip()}
    except subprocess.CalledProcessError:
        return set()
    except Exception:
        return set()


def query_group_vram_mib(gpu_id: int, pgid: int) -> float:
    """Sum memory.used over all compute processes on `gpu_id` whose pid is in
    the subprocess's process group. Resilient to neighbor pollution on shared
    boxes (unlike querying the whole-GPU memory.used)."""
    try:
        pids = pids_in_group(pgid)
        if not pids:
            return 0.0
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        total = 0.0
        for line in out.strip().splitlines():
            if "," not in line:
                continue
            pid_s, mem_s = line.split(",", 1)
            try:
                pid = int(pid_s.strip())
                mem = float(mem_s.strip())
            except ValueError:
                continue
            if pid in pids:
                total += mem
        return total
    except Exception:
        return 0.0


class VramSampler(threading.Thread):
    """Background thread that polls nvidia-smi every 200 ms and tracks max,
    filtered to the spawned server's process group (to ignore noisy neighbors
    on a shared GPU)."""

    def __init__(self, gpu_id: int, pgid: int):
        super().__init__(daemon=True)
        self.gpu_id = gpu_id
        self.pgid = pgid
        self.peak_mib = 0.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak_mib = max(
                self.peak_mib, query_group_vram_mib(self.gpu_id, self.pgid)
            )
            self._stop_event.wait(0.2)

    def stop(self) -> float:
        self._stop_event.set()
        self.join(timeout=2)
        return self.peak_mib


def wipe_cold_caches() -> None:
    """Remove all caches that should be empty on a true cold start.

    Does NOT touch ~/.cache/huggingface — model weight download is out of scope.
    """
    for p in COLD_CACHE_PATHS:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def drop_model_page_cache(model: str) -> None:
    """Hint the kernel to drop OS page cache for this model's weight shards
    (.safetensors / .bin) so the next read goes to disk, not RAM. Uses
    posix_fadvise(POSIX_FADV_DONTNEED) which requires no root privileges.

    Without this, 'cold' samples on a box that has previously loaded the
    weights read at RAM speed (4+ GB/s), hiding all disk-I/O effects.

    Best-effort: silently skips files that can't be resolved or opened.
    """
    import glob

    candidate_dir: str | None = None
    if os.path.isdir(model):
        candidate_dir = model
    else:
        try:
            from huggingface_hub import snapshot_download

            candidate_dir = snapshot_download(
                repo_id=model,
                allow_patterns=["*.safetensors", "*.bin"],
                local_files_only=True,
            )
        except Exception:
            return
    if not candidate_dir or not os.path.isdir(candidate_dir):
        return

    shards = sorted(
        glob.glob(os.path.join(candidate_dir, "*.safetensors"))
        + glob.glob(os.path.join(candidate_dir, "*.bin"))
    )
    for shard in shards:
        try:
            fd = os.open(shard, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except Exception:
            pass


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_free_port() -> int:
    """Pick an ephemeral port by binding to 0; kernel picks a free one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port_free(port: int, timeout: float = 30.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if port_is_free(port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"port {port} did not free within {timeout}s")


def poll_health(port: int, deadline: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(HEALTH_POLL_S)
    raise TimeoutError("/health never returned 200")


def send_chat_completion(port: int, model: str) -> tuple[bool, str]:
    """Returns (ok, content_or_error)."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT_S) as r:
            if r.status != 200:
                return False, f"http {r.status}"
            data = json.loads(r.read())
    except Exception as e:
        return False, f"chat request failed: {e!r}"
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        return False, f"malformed response: {e!r}"
    if not content or not content.strip():
        return False, "empty content"
    return True, content


def tail(path: str, n: int = 50) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            data = f.read().decode(errors="replace")
        return "\n".join(data.splitlines()[-n:])
    except Exception as e:
        return f"<could not read {path}: {e!r}>"


def run_single_sample(
    kind: str,
    gpu_id: int,
    model: str,
    port: int,
    extra_args: list[str],
    log_dir: Path,
    sample_idx: int,
) -> SampleResult:
    """Run one server launch + one chat completion. Returns SampleResult."""
    server_log = log_dir / f"sample_{sample_idx:02d}_{kind}.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)

    wait_port_free(port)

    if kind == "cold":
        wipe_cold_caches()
        drop_model_page_cache(model)
    TORCHINDUCTOR_CACHE.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(TORCHINDUCTOR_CACHE)
    # `vllm` is a console-script; Python sets sys.path[0] = script_dir,
    # not cwd. The editable-precompiled install on this box drops an
    # __init__-less `vllm/` dir in site-packages which PathFinder then
    # returns as a namespace package before _EditableFinder gets a turn.
    # Forcing PYTHONPATH=REPO_ROOT puts the source tree ahead of
    # site-packages so `import vllm` resolves to the real __init__.py.
    # (A pip-install user on their own box doesn't need this; it's a
    # workaround for the local editable-precompiled install only.)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + (os.pathsep + existing_pp if existing_pp else "")
    )
    # Spawn with cwd=REPO_ROOT so PathFinder resolves `import vllm` to the
    # source tree's __init__.py. On this box the editable-precompiled install
    # drops a partial `vllm/` dir in site-packages without an __init__.py; when
    # cwd is outside the repo, PathFinder (meta_path[4]) returns a namespace
    # package from site-packages and the setuptools _EditableFinder
    # (meta_path[5]) never runs, so `from vllm import SamplingParams` fails.
    #
    # [apr-17] Switched from `python -m vllm.entrypoints.openai.api_server`
    # to `vllm serve <model> --port N`. The CLI path is what users actually
    # invoke via `pip install vllm && vllm serve ...`; measuring it includes
    # an extra ~1-2 s of CLI-layer startup that the `-m` path skipped.
    vllm_cli = str(Path(sys.executable).parent / "vllm")
    cmd = [
        vllm_cli,
        "serve",
        model,
        "--port",
        str(port),
        *extra_args,
    ]

    log_fp = open(server_log, "wb")
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # so we can kill the process group
    )
    # With start_new_session=True, proc.pid is the new session+pgid leader.
    sampler = VramSampler(gpu_id, pgid=proc.pid)
    sampler.start()

    error: str | None = None
    t_ready_s: float | None = None
    correctness_ok = False
    try:
        deadline = t0 + SAMPLE_TIMEOUT_S
        try:
            poll_health(port, deadline)
        except TimeoutError as e:
            error = str(e)
        else:
            ok, detail = send_chat_completion(port, model)
            t_ready_s = time.monotonic() - t0
            correctness_ok = ok
            if not ok:
                error = detail
        # if subprocess died early, capture that
        if proc.poll() is not None and t_ready_s is None:
            error = f"server exited early (rc={proc.returncode})"
    finally:
        # Kill the server subprocess group to catch any worker children.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        log_fp.close()
        peak_mib = sampler.stop()

    return SampleResult(
        kind=kind,
        t_ready_s=t_ready_s,
        peak_vram_mib=peak_mib,
        correctness_ok=correctness_ok,
        error=error,
        server_log_path=str(server_log),
    )


def run_experiment(tag: str, config_name: str) -> ExperimentResult:
    cfg = CONFIGS[config_name]
    model, extra = cfg["model"], cfg["extra_args"]
    port = pick_free_port()

    gpu_id = pick_free_gpu()
    log(f"using GPU {gpu_id}, port {port}, model {model}")

    log_dir = REPO_ROOT / ".startup-bench" / "logs" / tag / "samples"
    log_dir.mkdir(parents=True, exist_ok=True)

    result = ExperimentResult(tag=tag, config=config_name, gpu_id=gpu_id)

    exp_start = time.monotonic()
    # Interleaved: cold0, warm0, cold1, warm1, cold2, warm2
    sequence = []
    for i in range(NUM_SAMPLES):
        sequence.append(("cold", i))
        sequence.append(("warm", i))

    for kind, i in sequence:
        if time.monotonic() - exp_start > EXPERIMENT_TIMEOUT_S:
            log(f"experiment timeout hit after {len(result.samples)} samples")
            result.status = "timeout"
            result.reason = "experiment budget exceeded"
            return result
        sample_idx = len(result.samples)
        log(f"sample {sample_idx} ({kind}, pair {i}) starting...")
        sr = run_single_sample(kind, gpu_id, model, port, extra, log_dir, sample_idx)
        if sr.t_ready_s is not None:
            log(
                f"  -> t_ready={sr.t_ready_s:.2f}s  "
                f"peak_vram={sr.peak_vram_mib:.0f}MiB  ok={sr.correctness_ok}"
            )
        else:
            log(f"  -> FAILED: {sr.error}")
            log(f"  last 30 lines of server log ({sr.server_log_path}):")
            for line in tail(sr.server_log_path, 30).splitlines():
                log(f"    {line}")
        result.samples.append(sr)

        if sr.error and sr.t_ready_s is None:
            result.status = "crash"
            result.reason = sr.error
            return result

    return result


def summarize(result: ExperimentResult) -> dict:
    cold = [s.t_ready_s for s in result.samples if s.kind == "cold" and s.t_ready_s]
    warm = [s.t_ready_s for s in result.samples if s.kind == "warm" and s.t_ready_s]
    all_correct = all(s.correctness_ok for s in result.samples if s.t_ready_s)
    peak_mib = max((s.peak_vram_mib for s in result.samples), default=0.0)

    def _stats(xs):
        if not xs:
            return None, None
        med = statistics.median(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return med, sd

    t_cold_med, t_cold_sd = _stats(cold)
    t_warm_med, t_warm_sd = _stats(warm)

    status = result.status
    reason = result.reason
    if status == "ok":
        if not all_correct:
            status = "crash"
            reason = "correctness failed on at least one sample"
        elif t_cold_med and t_cold_sd and (t_cold_sd / t_cold_med) > NOISY_THRESHOLD:
            status = "noisy"
            reason = f"cold stdev/median = {t_cold_sd / t_cold_med:.2%}"

    return {
        "tag": result.tag,
        "config": result.config,
        "gpu_id": result.gpu_id,
        "correctness": "pass" if all_correct else "fail",
        "samples_cold": [round(x, 3) for x in cold],
        "samples_warm": [round(x, 3) for x in warm],
        "t_cold_median_s": round(t_cold_med, 3) if t_cold_med else None,
        "t_cold_stdev_pct": (
            round(100 * t_cold_sd / t_cold_med, 2) if t_cold_med and t_cold_sd else None
        ),
        "t_warm_median_s": round(t_warm_med, 3) if t_warm_med else None,
        "t_warm_stdev_pct": (
            round(100 * t_warm_sd / t_warm_med, 2) if t_warm_med and t_warm_sd else None
        ),
        "peak_vram_gb": round(peak_mib / 1024, 2),
        "status": status,
        "reason": reason,
    }


def print_summary(s: dict) -> None:
    print("---")
    for k in [
        "tag",
        "config",
        "gpu_id",
        "correctness",
        "samples_cold",
        "samples_warm",
        "t_cold_median_s",
        "t_cold_stdev_pct",
        "t_warm_median_s",
        "t_warm_stdev_pct",
        "peak_vram_gb",
        "status",
        "reason",
    ]:
        print(f"{k}: {s[k]}")
    print("---")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    args = ap.parse_args()

    log(f"starting experiment tag={args.tag} config={args.config}")
    log(f"samples per kind: {NUM_SAMPLES}   sample timeout: {SAMPLE_TIMEOUT_S}s")

    try:
        result = run_experiment(args.tag, args.config)
    except Exception as e:
        log(f"experiment aborted: {e!r}")
        summary = {
            "tag": args.tag,
            "config": args.config,
            "gpu_id": -1,
            "correctness": "fail",
            "samples_cold": [],
            "samples_warm": [],
            "t_cold_median_s": None,
            "t_cold_stdev_pct": None,
            "t_warm_median_s": None,
            "t_warm_stdev_pct": None,
            "peak_vram_gb": 0.0,
            "status": "crash",
            "reason": repr(e),
        }
        print_summary(summary)
        return 1

    summary = summarize(result)
    print_summary(summary)
    return 0 if summary["status"] in ("ok", "noisy") else 1


if __name__ == "__main__":
    sys.exit(main())
