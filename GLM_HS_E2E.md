# GLM-5.3-Flash hidden-state extraction: machine handoff

## Code and status

- Draft PR: <https://github.com/vllm-project/vllm/pull/59037>
- Branch: `tomasruizt/vllm:hidden-states-blhnc`, implementation commit `450237c49`.
- Includes packed-layout prerequisite #57169 (`8d3e39144`) and aux-capture prerequisite #56983 (applied as `b7d338f71`). Use this branch, not upstream main alone.
- Passed on H200: 136 cache/connector unit tests, four GPU integration cases, and pre-commit checks.
- Integration coverage: Qwen3.5 with dummy weights under LBNHC/BLHNC; reduced GLM with dummy weights, real MLA/indexer/tail/KDA/mHC kernels, and both model runners.
- **Full trained GLM extraction has not been run.** The full-checkpoint procedure below is the next experiment, not a validated recipe.

## Machine and storage

Start with **4× H200, TP4, FP8**; 8× H200 is an alternative with more memory headroom. Fit is estimated, not verified by loading the full model.
Prefer `RedHatAI/GLM-5.3-Flash` (FP8, approximately 306 GiB / 328 GB of weights); `zai-org/GLM-5.3-Flash` is the official FP8 alternative.
The RedHatAI BF16 checkpoint is approximately 599 GiB and would require a larger GPU allocation; it is unnecessary for the initial test.

Check for an existing shared checkpoint before downloading. On the previous machine, `/home` had 1.1 TiB free **shared by all users**, with quotas disabled; that was not a personal allocation. Choose an appropriate model-storage volume on the new machine, and budget for download cache and saved hidden states as well as weights. Do not use a small `/tmp` volume for the checkpoint.

## Setup on a fresh machine

```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/tomasruizt/vllm-scripts.git
cat vllm-scripts/rh-setup/AGENTS.md
ln -s vllm-scripts/rh-setup/AGENTS.md AGENTS.md
mkdir -p ~/.bashrc.d
cp -i vllm-scripts/rh-setup/.bashrc-d/* ~/.bashrc.d/

git clone --filter=blob:none https://github.com/tomasruizt/vllm.git
git -C vllm worktree add -b hidden-states-blhnc ../vllm-hidden-states-blhnc origin/hidden-states-blhnc
cd vllm-hidden-states-blhnc
cat AGENTS.md
uv python install 3.12
uv venv --python 3.12 ~/.venv
source ~/.venv/bin/activate
source ~/.bashrc.d/vllm-install.bash
vllm-install
uv pip install -r requirements/lint.txt -r requirements/test/cuda.in ninja
pre-commit install
```

Preserve an existing environment or symlink instead of overwriting it. Follow `rh-setup/README.md` to load the shell snippets from `.bashrc` if needed, and configure the two Codex approval settings in `AGENTS.md` if using Codex.
The install helper uses the latest CUDA 13.0 nightly wheel, so record the installed versions; a later wheel may differ from the original validation environment. Ensure `nvcc` and `ninja` are on `PATH` for FlashInfer JIT compilation.

## Reproduce the existing tests first

Run from the worktree with `~/.venv` activated. Reserve GPUs through `canhazgpu`; do not select unreserved devices manually.

```bash
canhazgpu status --json
canhazgpu run --gpus 1 --timeout 20m -- ~/.venv/bin/python -m pytest \
  tests/v1/core/test_kv_cache_utils.py \
  tests/v1/kv_connector/unit/test_hidden_states_connector.py -q

canhazgpu run --gpus 1 --timeout 25m -- ~/.venv/bin/python -m pytest \
  tests/v1/kv_connector/extract_hidden_states_integration/test_extraction.py \
  -k 'qwen35 or glm5next' -q
```

Expected at `450237c49`: **136 passed** and **4 passed**, respectively. Reduced GLM cases require SM90-or-newer CUDA; inspect skips on other hardware. The GLM test parameter sets `VLLM_USE_V2_MODEL_RUNNER` to both 0 and 1.

## Download the full checkpoint

Replace these storage paths with appropriate locations on the new machine. Authenticate with Hugging Face locally if needed; do not put tokens in this file or scripts.

```bash
export HF_HUB_CACHE=/path/to/shared/model-cache
export HF_XET_CACHE=/path/to/download-cache
export HS_OUTPUT_DIR=/path/to/extraction-results
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HS_OUTPUT_DIR"
df -h "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HS_OUTPUT_DIR"
export MODEL_DIR="$(hf download RedHatAI/GLM-5.3-Flash)"
printf '%s\n' "$MODEL_DIR"
```

Keep the returned snapshot path: it identifies the downloaded revision. Reuse an existing snapshot by setting `MODEL_DIR` directly. Download before reserving GPUs.

## Full-checkpoint experiment

Save this as `full_glm_hs.py` in the worktree. It loads all trained weights with no dimension overrides, extracts layers 5/22/43, and compares mixed-length concurrent requests with serial references using one engine. It is a text-only, prompt-state extraction check.

```python
import os
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    load_hidden_states,
)


def main():
    layer_ids = [5, 22, 43]
    output_dir = Path(os.environ["HS_OUTPUT_DIR"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=os.environ["MODEL_DIR"],
        tensor_parallel_size=int(os.getenv("TP_SIZE", "4")),
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(output_dir)},
        },
    )
    tokenizer = llm.get_tokenizer()
    texts = [
        "Explain why the sky appears blue.",
        "A gardener records rainfall and soil moisture each morning. " * 80,
        "A compiler translates source code and optimizes the resulting program. " * 160,
    ]
    prompts = [{"prompt_token_ids": tokenizer.encode(text)} for text in texts]
    assert all(0 < len(p["prompt_token_ids"]) < 4096 for p in prompts)
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    params = SamplingParams(temperature=0, max_tokens=1)

    references = []
    for prompt in prompts:
        (output,) = llm.generate([prompt], params)
        references.append(read_checked(output, len(layer_ids), hidden_size))

    for repeat in range(2):
        outputs = llm.generate(prompts, params)
        assert len(outputs) == len(references)
        for index, (output, reference) in enumerate(zip(outputs, references)):
            actual = read_checked(output, len(layer_ids), hidden_size)
            delta = (actual.float() - reference.float()).abs()
            print(f"batch={repeat} request={index} shape={tuple(actual.shape)} "
                  f"max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g}",
                  flush=True)
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-2)
    print("PASS: full-checkpoint serial/concurrent hidden-state extraction")


def read_checked(output, num_layers, hidden_size):
    path = output.kv_transfer_params["hidden_states_path"]
    tensors = load_hidden_states(path)
    states = tensors["hidden_states"]
    assert torch.equal(tensors["token_ids"], torch.tensor(output.prompt_token_ids))
    assert states.shape == (len(output.prompt_token_ids), num_layers, hidden_size)
    assert torch.isfinite(states).all()
    assert torch.count_nonzero(states).item() > 0
    return states


if __name__ == "__main__":
    main()
```

Run the original model runner first, then V2 in a separate launch after the first passes. The checkpoint's FP8 configuration remains active; `dtype="bfloat16"` does not request conversion of all weights to BF16.

```bash
export VLLM_KV_CACHE_LAYOUT=BLHNC
export VLLM_USE_V2_MODEL_RUNNER=0
export TP_SIZE=4
canhazgpu run --gpus 4 --timeout 2h -- ~/.venv/bin/python full_glm_hs.py \
  > "$HS_OUTPUT_DIR/tp4-runner0.log" 2>&1

export VLLM_USE_V2_MODEL_RUNNER=1
canhazgpu run --gpus 4 --timeout 2h -- ~/.venv/bin/python full_glm_hs.py \
  > "$HS_OUTPUT_DIR/tp4-runner1.log" 2>&1
```

Do not force the reduced test's 128-token block size on the real model; let native cache geometry resolve. If TP4 runs out of memory, inspect whether failure is weight loading, profiling, or cache allocation before trying TP8. Start without expert parallelism to keep the first distributed test focused.

## Interpretation and follow-ups

- Record commit, checkpoint snapshot, GPU model/count, installed versions, resolved layout/block sizes, cache capacity, startup logs, tensor shapes, and numerical errors. Retain safetensors when a comparison fails.
- The `atol=1e-5, rtol=1e-2` comparison passed the dummy-model tests; it is an initial acceptance criterion for real weights, not an established full-model error bound. Investigate failures rather than silently relaxing it.
- Serial/concurrent agreement checks batching and storage consistency, not absolute agreement with an independent model implementation. Keep comparisons within one engine: sparse-indexer autotuning can cause differences across launches or TP ranks (issues #58636/#58979).
- Once the initial test passes, increase lengths/concurrency, then separately test prefix caching, preemption, compiled/CUDA-graph execution, and generated-token extraction. Multi-token generation has separate known issues (#56417/#56442); keep `max_tokens=1` initially.
- All-layer capture, multimodal inputs, full-model accuracy, and performance remain separate experiments. If switching to an HTTP server, use `--disable-uvicorn-access-log` and keep one server alive across comparison runs.
