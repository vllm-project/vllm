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


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={"/cache/hf": hf_cache, "/artifacts": artifacts},
    secrets=HF_SECRETS,
)
def evaluate(
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    tasks: str = "openbookqa",
    num_fewshot: int = 0,
    limit: int = 0,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    # Qwen3 non-thinking defaults from the model card. Only affect *generative*
    # tasks (humaneval, gsm8k); multiple-choice tasks such as openbookqa,
    # arc_challenge and winogrande are scored by loglikelihood over the
    # candidate answers and never sample.
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
) -> str:
    """Run lm-eval tasks against a model, in-process via vLLM."""
    import json

    # Two independent gates gate code execution, and satisfying one is not
    # enough. `confirm_run_unsafe_code=True` below is lm-eval's; this is
    # HuggingFace `evaluate`'s, checked inside the `code_eval` metric
    # (`if os.getenv("HF_ALLOW_CODE_EVAL", 0) != "1": raise ValueError`).
    # humaneval's utils.py invokes that metric at *import* time, so without
    # this the whole run dies while building the task dict -- taking the
    # multiple-choice tasks down with it.
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    from lm_eval import evaluator
    from lm_eval.tasks import get_task_dict
    from lm_eval.utils import make_table

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    print(f"[eval] {model} | tasks={task_list} | {num_fewshot}-shot", flush=True)

    # Preflight the task configs before the model loads. simple_evaluate
    # constructs the LM *first* and only then calls get_task_dict, so a bad
    # task name or an ungated metric otherwise costs a full ~5 min weight load
    # before it reports. Resolving the dict here fails in seconds instead.
    get_task_dict(task_list)
    print(f"[eval] {len(task_list)} task config(s) resolved", flush=True)

    model_args = {
        "pretrained": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
        "dtype": "bfloat16",
    }
    # do_sample=True is load-bearing, not decoration. lm-eval merges this
    # string into each task's own `generation_kwargs` rather than replacing
    # it, and `humaneval.yaml` ships `do_sample: false`. The vLLM backend's
    # modify_gen_kwargs() then does
    #     if do_sample is False or "temperature" not in kwargs:
    #         kwargs["temperature"] = 0.0
    # so without this the four sampling flags below are silently discarded and
    # the run is greedy.
    gen_kwargs = ",".join(
        [
            "do_sample=True",
            f"temperature={temperature}",
            f"top_p={top_p}",
            f"top_k={top_k}",
            f"min_p={min_p}",
        ]
    )
    results = evaluator.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=task_list,
        num_fewshot=num_fewshot,
        limit=limit or None,
        batch_size="auto",
        gen_kwargs=gen_kwargs,
        apply_chat_template=False,
        # humaneval declares `unsafe_code: true` -- it scores by exec()ing the
        # model's completion against the reference tests. lm-eval refuses to
        # run such a task without explicit opt-in. Safe here because the whole
        # run is inside a disposable Modal container.
        confirm_run_unsafe_code=True,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
    )

    table = make_table(results)
    print(table, flush=True)

    label = model.rstrip("/").split("/")[-1]
    out = f"/artifacts/evals/{label}"
    os.makedirs(out, exist_ok=True)
    name = "-".join(task_list)[:60]
    with open(f"{out}/{name}.json", "w") as handle:
        json.dump(results.get("results", results), handle, indent=2, default=str)
    with open(f"{out}/{name}.txt", "w") as handle:
        handle.write(f"{model}\ntasks={task_list} {num_fewshot}-shot\n\n{table}\n")
    artifacts.commit()
    return f"{model}\n{table}\n\nsaved to {out}/{name}.{{json,txt}}"


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
