# SPDX-License-Identifier: Apache-2.0
"""Run the baseline sweep on a Modal GPU.

    modal run inco/modal/modal_baseline.py::prefetch             # weights, on CPU
    modal run inco/modal/modal_baseline.py --concurrencies 1,32  # quick check
    modal run inco/modal/modal_baseline.py                       # full sweep
    modal volume get inco-results baseline ./inco/results        # pull artifacts

Server and client share one container: aiperf then measures the engine rather
than the network, and per-second billing stops the moment the sweep finishes.
Results and the HF cache live on Modal volumes so a re-run skips the download.

The defaults target one 80GB H100, where Qwen3-30B-A3B's bf16 weights (~61GB)
cap the KV cache near 60 resident requests at ISL+OSL=1280. For the full
256-wide sweep use INCO_MODAL_GPU=H200 and pass --max-num-seqs 256.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

import modal

GPU = os.environ.get("INCO_MODAL_GPU", "H100")  # H200 fits a 256-wide sweep
TIMEOUT_S = int(os.environ.get("INCO_MODAL_TIMEOUT", 3 * 60 * 60))

# Modal re-imports this module inside the container, where there is no git
# checkout. Anything that inspects the local repo has to be local-only, or the
# container crash-loops on import and the call never starts.
LOCAL = modal.is_local()
REMOTE_REPO = "/workspace/vllm"
REPO_ROOT = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if LOCAL
    else REMOTE_REPO
)

CUDA_VARIANT = os.environ.get("INCO_CUDA_VARIANT", "cu130")

app = modal.App("inco-vllm-baseline")

HF_SECRETS = (
    [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]
    if LOCAL and os.environ.get("HF_TOKEN")
    else []
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", REPO_ROOT, *args], text=True).strip()


def _fork_build_env() -> dict[str, str]:
    """Resolve what setup.py normally reads out of `.git`.

    `.git` is deliberately not uploaded: it is 350MB, and including it would
    invalidate the image layer on every commit. Without it setup.py cannot
    derive a version (setuptools-scm) or find a matching prebuilt wheel, so
    both are pinned here.

    The wheel is pinned to this checkout's merge-base with `origin/main`, not
    to HEAD: nightly wheels only exist for upstream commits, so once you start
    committing your own changes HEAD will not have one. The merge-base wheel
    supplies kernels for the last upstream commit you branched from, and your
    Python changes are layered on top by the editable install.
    """
    base = _git("merge-base", "HEAD", "origin/main")
    meta = f"https://wheels.vllm.ai/{base}/{CUDA_VARIANT}/vllm/metadata.json"
    try:
        with urllib.request.urlopen(meta, timeout=30) as resp:
            version = json.loads(resp.read())[0]["version"]
    except Exception as exc:
        raise RuntimeError(
            f"no {CUDA_VARIANT} nightly wheel for upstream base commit {base}\n"
            f"  ({meta}: {exc})\n"
            "Fetch upstream main so merge-base resolves to a commit with a "
            "published wheel, or set INCO_CUDA_VARIANT."
        ) from exc

    print(f"[modal] precompiled {CUDA_VARIANT} wheel {version} (base {base[:9]})")
    return {
        "VLLM_PRECOMPILED_WHEEL_COMMIT": base,
        "VLLM_MAIN_CUDA_VERSION": CUDA_VARIANT.removeprefix("cu")[:2] + ".0",
        # setuptools-scm cannot see .git; report the version of the kernels
        # actually being linked against.
        "SETUPTOOLS_SCM_PRETEND_VERSION": version,
    }


# Already baked into the image by the time the container imports this.
FORK_BUILD_ENV = _fork_build_env() if LOCAL else {}

hf_cache = modal.Volume.from_name("inco-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("inco-results", create_if_missing=True)
vllm_cache = modal.Volume.from_name("inco-vllm-cache", create_if_missing=True)

# Install THIS fork rather than a published wheel. The harness reads config
# fields (cache_config.kv_cache_size_tokens) and passes server flags
# (--async-scheduling, --default-chat-template-kwargs) that were verified
# against this tree's source; HEAD here is months ahead of any vLLM release,
# so a PyPI wheel would not have them. It is also the point of the exercise:
# the baseline must be the engine you are going to modify.
#
# VLLM_USE_PRECOMPILED downloads prebuilt kernels and links this tree's Python
# source on top, so no CUDA compile happens during the image build.
# A CUDA *devel* base is required, not just a runtime one: FlashInfer builds
# its sampling kernels with nvcc on first use (during engine init, well after
# the precompiled vLLM wheel is installed), and debian_slim has no toolkit --
# engine startup dies with "Could not find nvcc". It also gives us the toolkit
# needed later for nsys and any hand-written kernels.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "curl")
    .pip_install("uv")
    .env(FORK_BUILD_ENV)
    # The engine and the harness are deliberately layered separately. The copy
    # below excludes `inco/`, so editing the benchmark harness does not
    # invalidate the (slow) vLLM install layer -- only touching actual engine
    # source does. `inco/` is mounted at container start instead.
    .add_local_file(
        os.path.join(REPO_ROOT, "inco", "requirements.txt"),
        remote_path="/tmp/inco-requirements.txt",
        copy=True,
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_REPO,
        copy=True,  # baked into the layer so the install below can see it
        # Anything transient must be excluded: Modal aborts the build if a
        # file changes mid-upload, and a stray linter run is enough to do it.
        ignore=[
            ".git",
            ".venv*",
            "inco/results",
            "**/__pycache__",
            "**/*.pyc",
            "**/*.so",
            ".ruff_cache",
            ".pytest_cache",
            ".mypy_cache",
            ".coverage*",
            "build",
            "dist",
            "*.egg-info",
            "**/*.log",
            "inco",
        ],
    )
    .run_commands(
        f"cd {REMOTE_REPO} && VLLM_USE_PRECOMPILED=1 uv pip install --system -e .",
        "uv pip install --system -r /tmp/inco-requirements.txt",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "VLLM_SERVER_DEV_MODE": "1",
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            # Persist FlashInfer's JIT output on the volume so the nvcc build
            # is paid once rather than on every run.
            "FLASHINFER_WORKSPACE_BASE": "/cache/vllm/flashinfer",
            "FLASHINFER_CACHE_DIR": "/cache/vllm/flashinfer/jit",
        }
    )
    # Runtime mount (no copy): harness edits take effect without a rebuild.
    .add_local_dir(
        os.path.join(REPO_ROOT, "inco"),
        remote_path=f"{REMOTE_REPO}/inco",
        ignore=["results", "**/__pycache__", "**/*.pyc", ".pytest_cache"],
    )
)


def _await_server(server, log_path: str, timeout_s: float = 1800.0) -> None:
    """Wait for /health, but abort the moment the server process dies.

    Without the liveness check a crashed server would be indistinguishable
    from a slow one: the client would poll for its full timeout, burning GPU
    time and reporting nothing useful. vLLM's own startup log lives in a file
    inside the container, so its tail is echoed to stdout where Modal captures
    it -- that log is the only place a bad flag or an OOM actually shows up.
    """
    import time
    import urllib.error
    import urllib.request

    def tail(lines: int = 60) -> str:
        try:
            with open(log_path) as handle:
                return "".join(handle.readlines()[-lines:])
        except OSError:
            return "(no server log)"

    def last_progress() -> str:
        """Most recent informative line, so the phase is visible while waiting."""
        for line in reversed(tail(40).splitlines()):
            stripped = line.strip()
            if stripped and "INFO" in stripped:
                return stripped.split("] ", 1)[-1][:110]
        return "starting up"

    started = time.monotonic()
    deadline = started + timeout_s
    while True:
        if (code := server.poll()) is not None:
            raise RuntimeError(
                f"vllm serve exited with code {code} before becoming healthy\n"
                f"--- server.log tail ---\n{tail()}"
            )
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=5):
                elapsed = time.monotonic() - started
                print(f"[modal] server healthy after {elapsed:.0f}s", flush=True)
                # Surface the engine's own memory accounting; this is where
                # the real KV cache size is reported.
                for line in tail(400).splitlines():
                    if any(
                        key in line
                        for key in (
                            "KV cache",
                            "GPU KV cache",
                            "Maximum concurrency",
                            "memory profiling",
                            "Graph capturing",
                        )
                    ):
                        print(f"[vllm] {line.strip()}", flush=True)
                return
        except Exception:  # noqa: BLE001 - not up yet
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"server not healthy after {timeout_s:.0f}s\n"
                    f"--- server.log tail ---\n{tail()}"
                ) from None
            print(
                f"[modal] loading ({time.monotonic() - started:.0f}s) "
                f"| {last_progress()}",
                flush=True,
            )
            time.sleep(5)


@app.function(
    image=image,
    timeout=TIMEOUT_S,
    volumes={"/cache/hf": hf_cache},
    secrets=HF_SECRETS,
)
def prefetch(model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507") -> str:
    """Pull the weights on a CPU container so the GPU run starts warm.

    Qwen3-30B-A3B is ~61GB. Downloading it inside the GPU function would bill
    ten minutes of H100 for pure network transfer, and would also mean a slow
    first run cannot be told apart from a slow engine.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(model)
    hf_cache.commit()
    print(f"[modal] {model} cached at {path}", flush=True)
    return path


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={"/cache/hf": hf_cache, "/cache/vllm": vllm_cache, "/results": results},
    secrets=HF_SECRETS,
)
def sweep(
    label: str = "baseline",
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    isl: int = 1024,
    osl: int = 256,
    concurrencies: str = "1,2,4,8,16,32,64",
    max_num_seqs: int = 64,
    max_num_batched_tokens: int = 8192,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    kv_cache_gib: float = 12.0,
    extra_serve_args: str = "",
    extra_sweep_args: str = "",
    moe_shape_log: bool = False,
) -> str:
    """Launch the server, run the sweep, return the markdown summary."""
    import shlex

    if moe_shape_log:
        os.environ["INCO_MOE_SHAPE_LOG"] = "1"

    serve_cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        "8000",
        "--served-model-name",
        model,
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--async-scheduling",
        "--enable-prefix-caching",
        # Pinned, not derived from gpu-memory-utilization: Modal's "H100" can
        # be an 80GB HBM3 or a 94GB NVL, which at util 0.90 leaves ~12.5GiB vs
        # ~25GiB of KV and doubles servable concurrency between runs.
        *(
            ["--kv-cache-memory-bytes", str(int(kv_cache_gib * 1024**3))]
            if kv_cache_gib > 0
            else []
        ),
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
        *shlex.split(extra_serve_args),
    ]
    print("[modal] launching:", shlex.join(serve_cmd), flush=True)

    log_dir = f"/results/{label}"
    os.makedirs(log_dir, exist_ok=True)
    server_log = open(f"{log_dir}/server.log", "w")  # noqa: SIM115
    server = subprocess.Popen(serve_cmd, stdout=server_log, stderr=subprocess.STDOUT)

    try:
        _await_server(server, f"{log_dir}/server.log")
        sweep_cmd = [
            sys.executable,
            "-m",
            "bench.sweep",
            "--label",
            label,
            "--model",
            model,
            "--isl",
            str(isl),
            "--osl",
            str(osl),
            "--num-gpus",
            "1",
            "--artifact-root",
            "/results",
            "--concurrency",
            *concurrencies.replace(",", " ").split(),
            *shlex.split(extra_sweep_args),
        ]
        print("[modal] sweeping:", shlex.join(sweep_cmd), flush=True)
        # Modal ships Python-level stdout, not an inherited fd, so a child
        # writing straight to fd 1 is invisible. Pipe it and re-emit each line
        # through print() -- otherwise a 20-minute sweep looks like a hang.
        child = subprocess.Popen(
            sweep_cmd,
            cwd=f"{REMOTE_REPO}/inco",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        for line in child.stdout:
            print(line.rstrip(), flush=True)
        returncode = child.wait()
        if returncode != 0:
            print(f"[modal] sweep exited {returncode}", file=sys.stderr, flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()
        results.commit()
        # torch.compile and CUDA graph artifacts: persisting them saves several
        # minutes of GPU time on every subsequent run of the same config.
        vllm_cache.commit()

    summary_path = f"/results/{label}/summary.md"
    return (
        open(summary_path).read()
        if os.path.exists(summary_path)
        else f"no summary at {summary_path}; see {log_dir}/server.log"
    )


@app.local_entrypoint()
def main(
    label: str = "baseline",
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    isl: int = 1024,
    osl: int = 256,
    concurrencies: str = "1,2,4,8,16,32,64",
    max_num_seqs: int = 64,
    kv_cache_gib: float = 12.0,
    extra_serve_args: str = "",
    extra_sweep_args: str = "",
    moe_shape_log: bool = False,
) -> None:
    print(
        sweep.remote(
            label=label,
            model=model,
            isl=isl,
            osl=osl,
            concurrencies=concurrencies,
            max_num_seqs=max_num_seqs,
            kv_cache_gib=kv_cache_gib,
            extra_serve_args=extra_serve_args,
            extra_sweep_args=extra_sweep_args,
            moe_shape_log=moe_shape_log,
        )
    )
    print(
        "\nPull artifacts with:\n"
        f"  modal volume get inco-results {label} ./inco/results --force"
    )
