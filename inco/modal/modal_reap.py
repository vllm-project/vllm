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


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={"/cache/hf": hf_cache, "/artifacts": artifacts},
    secrets=HF_SECRETS,
)
def evalplus_eval(
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    dataset: str = "humaneval",
    greedy: bool = True,
    temperature: float = 0.0,
    max_new_tokens: int = 1280,
    tp: int = 1,
) -> str:
    """Run EvalPlus on one dataset, reporting both base and plus pass@1.

    Preferred over `evaluate()` for code benchmarks. EvalPlus applies the
    tokenizer's chat template, sanitizes generations with tree-sitter before
    execution, and scores against the expanded plus test suites -- none of
    which lm-eval's `humaneval`/`mbpp` tasks do. lm-eval's `mbpp_plus` in
    particular reads `test_list` rather than the expanded `test`, so it is not
    MBPP+ at all.

    One dataset per call, deliberately: each call builds its own `LLM`, and two
    in a single container would OOM because the first is never released.

    Args:
        model: HF repo id, or a checkpoint path on the artifacts volume.
        dataset: Either ``humaneval`` or ``mbpp``.
        greedy: Greedy decoding, as the EvalPlus leaderboard uses. Forces
            ``temperature=0``, ``bs=1``, ``n_samples=1`` inside `run_codegen`.
        temperature: Only consulted when ``greedy`` is False.
        max_new_tokens: Must stay under `VllmDecoder`'s hardcoded
            ``max_model_len=2048``. See the patch note below.
        tp: Tensor parallel size.

    Returns:
        A one-line-per-metric summary of base and plus pass@1.
    """
    import json

    # Patch before evalplus.provider.base is imported: it binds MAX_NEW_TOKENS
    # as a default argument at import time, so a later assignment is ignored.
    # The vendored fork ships 16384 while VllmDecoder hardcodes
    # max_model_len=2048, which makes every vllm-backend request invalid --
    # why reap's own eval.py only ever uses the hf and openai backends.
    import evalplus.config

    evalplus.config.MAX_NEW_TOKENS = max_new_tokens

    from evalplus.evaluate import evaluate as evalplus_evaluator

    label = model.rstrip("/").split("/")[-1]
    out = f"/artifacts/evalplus/{label}"
    os.makedirs(out, exist_ok=True)

    mode = "greedy" if greedy else f"t{temperature}"
    output_file = f"{out}/{dataset}-{mode}.json"
    print(f"[evalplus] {model} | {dataset} | {mode}", flush=True)

    evalplus_evaluator(
        dataset=dataset,
        output_file=output_file,
        model=model,
        # Stamped with the decoding mode so `resume` cannot silently serve
        # generations produced under different sampling settings.
        root=f"{out}/codegen-{mode}",
        backend="vllm",
        greedy=greedy,
        temperature=0.0 if greedy else temperature,
        tp=tp,
        trust_remote_code=True,
    )

    with open(output_file) as handle:
        scores = json.load(handle).get("pass_at_k", {})
    base = scores.get("base", {}).get("pass@1")
    plus = scores.get("plus", {}).get("pass@1")

    table = "\n".join(
        [
            f"{model}",
            f"{dataset} ({mode})",
            f"  {dataset} pass@1      = {base}",
            f"  {dataset}+ pass@1     = {plus}",
        ]
    )
    print(table, flush=True)
    with open(f"{out}/{dataset}-{mode}.txt", "w") as handle:
        handle.write(table + "\n")
    artifacts.commit()
    return f"{table}\n\nsaved to {output_file}"


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    volumes={"/cache/hf": hf_cache, "/artifacts": artifacts},
    secrets=HF_SECRETS,
)
def expert_activation(
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    num_seqs: int = 256,
    seq_len: int = 16,
    batch_sizes: str = "1,2,4,8,16,32,64,128,256",
    repeats: int = 64,
    seed: int = 42,
) -> str:
    """Count distinct experts activated per MoE layer vs decode batch size.

    Shows why pruning buys throughput only at scale: one token routes to
    `top_k` experts, so a batch of B decoding sequences touches at most
    `B * top_k` of them. Past saturation every expert is read every forward
    pass, and halving the expert count halves MoE weight traffic.

    A single forward pass over `num_seqs` sequences yields every layer's
    router logits; decode at batch B is emulated by subsampling B sequences'
    final-position decisions, averaged over `repeats` draws. The whole curve
    therefore costs one forward pass, not one per batch size.

    Args:
        model: HF repo id, or a checkpoint path on the artifacts volume.
        num_seqs: Sequences in the forward pass; caps the largest batch size.
        seq_len: Tokens per sequence. Only the final position is counted.
        batch_sizes: Comma-separated decode batch sizes to report.
        repeats: Random subsamples averaged per batch size.
        seed: Seed for prompt selection and subsampling.

    Returns:
        A markdown table of mean distinct experts per layer per batch size.
    """
    import json

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    sizes = sorted(int(b) for b in batch_sizes.split(",") if b.strip())
    if max(sizes) > num_seqs:
        raise ValueError(f"batch size {max(sizes)} exceeds num_seqs={num_seqs}")

    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    n_exp, top_k = cfg.num_experts, cfg.num_experts_per_tok
    print(
        f"[experts] {model} | E={n_exp} top_k={top_k} "
        f"layers={cfg.num_hidden_layers}",
        flush=True,
    )

    # Only prompts at least `seq_len` tokens long, so every position is a real
    # token and no padding mask is needed.
    tok = AutoTokenizer.from_pretrained(model)
    rows = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
    kept: list[list[int]] = []
    for row in rows:
        ids = tok(row["instruction"], add_special_tokens=False)["input_ids"]
        if len(ids) >= seq_len:
            kept.append(ids[:seq_len])
        if len(kept) == num_seqs:
            break
    if len(kept) < num_seqs:
        raise ValueError(f"only {len(kept)} prompts reached {seq_len} tokens")

    # transformers 4.55 predates the `dtype` alias; it wants `torch_dtype`.
    lm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    with torch.no_grad():
        out = lm(
            input_ids=torch.tensor(kept, device="cuda"),
            output_router_logits=True,
        )

    # topk over logits == topk over the router softmax (monotonic), so
    # selection is unaffected by normalisation.
    picks = torch.stack(
        [
            logits.view(num_seqs, seq_len, n_exp)[:, -1, :]
            .topk(top_k, dim=-1)
            .indices
            for logits in out.router_logits
        ]
    )  # (layers, num_seqs, top_k)
    n_layers = picks.shape[0]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    curve: dict[int, float] = {}
    for size in sizes:
        means = []
        for _ in range(repeats):
            idx = torch.randperm(num_seqs, generator=gen)[:size].to(picks.device)
            sel = picks[:, idx, :].reshape(n_layers, -1)
            seen = torch.zeros(
                n_layers, n_exp, dtype=torch.bool, device=sel.device
            )
            seen.scatter_(1, sel, True)
            means.append(seen.sum(1).float().mean().item())
        curve[size] = sum(means) / len(means)

    lines = [
        f"{model}",
        f"E={n_exp} top_k={top_k} layers={n_layers} "
        f"({num_seqs} seqs x {seq_len} tok, {repeats} draws)",
        "",
        "| decode batch | tokens | experts/layer | % of E | vs full |",
        "|---|---|---|---|---|",
    ]
    for size, mean in curve.items():
        lines.append(
            f"| {size} | {size} | {mean:.1f} | {mean / n_exp * 100:.1f}% | "
            f"{mean / n_exp:.3f} |"
        )
    table = "\n".join(lines)
    print(table, flush=True)

    label = model.rstrip("/").split("/")[-1]
    out_dir = "/artifacts/experts"
    os.makedirs(out_dir, exist_ok=True)
    # The picks are the only thing the forward pass is needed for; saving them
    # lets any other batch grid be recomputed locally without a GPU.
    np.save(f"{out_dir}/{label}.picks.npy", picks.cpu().numpy().astype("int16"))
    with open(f"{out_dir}/{label}.json", "w") as handle:
        json.dump(
            {
                "model": model,
                "num_experts": n_exp,
                "top_k": top_k,
                "layers": n_layers,
                "num_seqs": num_seqs,
                "seq_len": seq_len,
                "repeats": repeats,
                "experts_per_layer": curve,
            },
            handle,
            indent=2,
        )
    with open(f"{out_dir}/{label}.txt", "w") as handle:
        handle.write(table + "\n")
    artifacts.commit()
    return f"{table}\n\nsaved to {out_dir}/{label}.{{json,txt,picks.npy}}"


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
