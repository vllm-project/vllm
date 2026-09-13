# SPDX-License-Identifier: Apache-2.0
"""Run REAP expert pruning on a Modal GPU.

    modal run inco/modal/modal_reap.py --compression-ratio 0.5
    modal volume get inco-reap pruned ./inco/results/reap    # pull the checkpoint

Deliberately a separate app and image from `modal_baseline.py`: REAP's
`scripts/build.sh` installs its *own* vLLM, so sharing an environment would
overwrite the engine under test. The HF cache volume is shared, so the 61GB of
Qwen3 weights are not downloaded twice.

Calibration sample count is `batches_per_category x batch_size`. REAP's shell
wrapper hardcodes 128 x 8 = 1024; this exposes both so they can be set, and
calls `reap.layerwise_prune` directly to skip the eval stage the wrapper
appends (lm_eval + evalplus + LiveCodeBench, which take hours).
"""

from __future__ import annotations

import os
import subprocess
import sys

import modal

GPU = os.environ.get("INCO_REAP_GPU", "H100")
TIMEOUT_S = int(os.environ.get("INCO_REAP_TIMEOUT", 6 * 60 * 60))
INCO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_REAP = os.path.join(INCO_DIR, "reap")
REMOTE_REAP = "/workspace/reap"

TOTAL_BLOCKS = 48  # Qwen3-30B-A3B decoder layers; only used for the ETA

app = modal.App("inco-reap")

hf_cache = modal.Volume.from_name("inco-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("inco-reap", create_if_missing=True)

HF_SECRETS = (
    [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]
    if modal.is_local() and os.environ.get("HF_TOKEN")
    else []
)

# CUDA 12.8, not 13: REAP pins torch==2.7.1 and vllm==0.10.0, which are cu12x.
# (The benchmark image uses CUDA 13 because this fork defaults to
# VLLM_MAIN_CUDA_VERSION=13.0 -- another reason the two environments must not
# be shared.)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "curl")
    .pip_install("uv")
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "TOKENIZERS_PARALLELISM": "false",
            # third-party/evalplus derives its version from git metadata, which
            # is not uploaded. It is a hard dependency of reap even though no
            # eval is run here, so give setuptools-scm a version directly.
            # Must satisfy reap's `evalplus[vllm]>=0.3.1`.
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EVALPLUS": "0.3.1",
        }
    )
    .add_local_dir(
        LOCAL_REAP,
        remote_path=REMOTE_REAP,
        copy=True,
        # third-party/ cannot be excluded wholesale: pyproject declares
        # evalplus, livecodebench, crfm-helm and evalscope as *core* deps with
        # [tool.uv.sources] pointing at those paths, so the install fails
        # without them even though no eval is run. The two submodules not
        # referenced there are dropped -- creative-writing-bench alone is
        # 172MB of the 470MB checkout.
        ignore=[
            "**/.git",
            ".venv*",
            "artifacts",
            "third-party/creative-writing-bench",
            "third-party/llm-compressor",
            "**/__pycache__",
            "**/*.pyc",
            "fig",
        ],
    )
    .run_commands(
        # --no-deps, then an explicit dependency set.
        #
        # reap's pyproject declares four eval harnesses as *core* deps
        # (evalplus, livecodebench, crfm-helm, evalscope). Only `evalplus` and
        # `lm_eval` are imported on the pruning path -- reap.layerwise_prune
        # imports reap.eval unconditionally, and that module imports both at
        # module level. `crfm-helm` and `evalscope` are never reached, but
        # resolving them drags in a long tail of source builds (helm pins
        # zstandard==0.18.0, which has no cp312 wheel and needs clang).
        #
        # The set below is every third-party module imported anywhere under
        # src/reap/, nothing more.
        f"cd {REMOTE_REAP} && VLLM_USE_PRECOMPILED=1 "
        "uv pip install --system --no-deps -e .",
        f"cd {REMOTE_REAP} && uv pip install --system "
        "'torch==2.7.1' 'transformers==4.55.0' 'vllm==0.10.0' "
        "'datasets>=3.6.0,<4.0.0' 'accelerate>=1.7.0' 'lm-eval[vllm]>=0.4.9.1' "
        "python-dotenv pyyaml numpy scipy scikit-learn matplotlib seaborn "
        "tqdm uvloop requests "
        "-e third-party/evalplus",
    )
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={"/cache/hf": hf_cache, "/artifacts": artifacts},
    secrets=HF_SECRETS,
)
def prune(
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    compression_ratio: float = 0.5,
    prune_method: str = "reap",
    dataset: str = "theblackcat102/evol-codealpaca-v1",
    batches_per_category: int = 128,
    batch_size: int = 8,
    model_max_length: int = 2048,
    seed: int = 42,
    distance_measure: str = "cosine",
    extra_args: str = "",
) -> str:
    """Calibrate and prune, writing the checkpoint to the artifacts volume."""
    import shlex

    samples = batches_per_category * batch_size
    print(
        f"[reap] {model} | {prune_method} | ratio={compression_ratio} | "
        f"{samples} calibration samples ({batches_per_category} x {batch_size}) "
        f"| max_len={model_max_length}",
        flush=True,
    )

    cmd = [
        sys.executable,
        "-m",
        "reap.layerwise_prune",
        "--model-name", model,
        "--dataset-name", dataset,
        "--compression-ratio", str(compression_ratio),
        "--prune-method", prune_method,
        "--batches_per_category", str(batches_per_category),
        "--batch_size", str(batch_size),
        "--model_max_length", str(model_max_length),
        "--distance_measure", distance_measure,
        "--seed", str(seed),
        "--output_file_name", f"observations_{samples}_{distance_measure}-seed_{seed}.pt",
        "--low_cpu_mem_usage", "True",
        "--profile", "false",
        "--do-eval", "false",
        *shlex.split(extra_args),
    ]
    print("[reap] " + shlex.join(cmd), flush=True)

    # Artifacts land under the repo's `artifacts/` by default; symlink it onto
    # the volume so the checkpoint survives the container.
    os.makedirs("/artifacts/pruned", exist_ok=True)
    repo_artifacts = f"{REMOTE_REAP}/artifacts"
    if not os.path.islink(repo_artifacts):
        os.makedirs("/artifacts/pruned", exist_ok=True)
        if os.path.exists(repo_artifacts):
            subprocess.run(["cp", "-rn", repo_artifacts + "/.", "/artifacts/pruned/"])
            subprocess.run(["rm", "-rf", repo_artifacts])
        os.symlink("/artifacts/pruned", repo_artifacts)

    child = subprocess.Popen(
        cmd,
        cwd=REMOTE_REAP,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # REAP logs with the default `INFO:name:message` format -- no timestamps,
    # and no progress bar over the calibration block loop. At 128 batches per
    # block that loop is the bulk of the run, so prefix elapsed time and
    # project a finish time from the per-block rate.
    import re
    import time

    started = time.monotonic()
    block_re = re.compile(r"Completed block (\d+):")
    first_block_at: float | None = None
    for line in child.stdout:
        line = line.rstrip()
        elapsed = time.monotonic() - started
        suffix = ""
        if match := block_re.search(line):
            done = int(match.group(1)) + 1
            if first_block_at is None:
                first_block_at = elapsed
            span = elapsed - (first_block_at or elapsed)
            if done > 1 and span > 0:
                per_block = span / (done - 1)
                remaining = per_block * (TOTAL_BLOCKS - done)
                suffix = (
                    f"  [{done}/{TOTAL_BLOCKS} blocks, {per_block:.1f}s each, "
                    f"~{remaining / 60:.0f} min left in calibration]"
                )
        print(f"[{elapsed / 60:6.1f}m] {line}{suffix}", flush=True)
    rc = child.wait()
    print(f"[{(time.monotonic() - started) / 60:6.1f}m] finished rc={rc}", flush=True)

    artifacts.commit()
    hf_cache.commit()

    found = subprocess.run(
        ["find", "/artifacts/pruned", "-name", "config.json", "-maxdepth", "6"],
        capture_output=True, text=True,
    ).stdout.strip()
    return (
        f"exit={rc}\ncheckpoints found:\n{found or '  (none)'}\n"
        "pull with: modal volume get inco-reap pruned ./inco/results/reap"
    )


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    compression_ratio: float = 0.5,
    prune_method: str = "reap",
    dataset: str = "theblackcat102/evol-codealpaca-v1",
    batches_per_category: int = 128,
    batch_size: int = 8,
    model_max_length: int = 2048,
    seed: int = 42,
    extra_args: str = "",
) -> None:
    print(
        prune.remote(
            model=model,
            compression_ratio=compression_ratio,
            prune_method=prune_method,
            dataset=dataset,
            batches_per_category=batches_per_category,
            batch_size=batch_size,
            model_max_length=model_max_length,
            seed=seed,
            extra_args=extra_args,
        )
    )
