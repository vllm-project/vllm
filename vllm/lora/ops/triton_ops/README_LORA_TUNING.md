# LoRA Triton Kernel Tuning Script (`benchmark_lora_tuning.py`)

This document explains how to use `benchmark_lora_tuning.py` to grid-search
LoRA Triton kernel configurations and generate JSON config files that are
consumed at runtime by `vllm.lora.ops.triton_ops.utils.get_lora_op_configs`.

The script lives at:

`vllm/benchmarks/kernels/benchmark_lora_tuning.py`

and is meant as a **standalone tuning tool** for:

- `lora_shrink`
- `lora_expand`
- `fused_moe_lora_gate_up_shrink`
- `fused_moe_lora_gate_up_expand`
- `fused_moe_lora_down_shrink`
- `fused_moe_lora_down_expand`

It runs a predefined search space of Triton kernel parameters, benchmarks
each candidate, and writes the **best config** into JSON files under a
user-specified directory.

---

## 1. Prerequisites

- A CUDA GPU and a working CUDA driver.
- Triton kernels enabled in vLLM (`HAS_TRITON` must be `True`).
- PyTorch with CUDA support.
- (Optional) `transformers` and `safetensors` if you want
  `benchmark_lora_tuning.py` to **auto-infer** dimensions from
  `--model-path` / `--lora-path` (see section 3.6). If these are not
  installed, you can simply pass `--hidden-size` / `--lora-rank`
  explicitly.

At runtime, vLLM will read tuned configs from the directory pointed to:

```bash
export VLLM_TUNED_CONFIG_FOLDER=/path/to/tuned/configs
```

This folder must contain JSON files named according to
`vllm/vllm/lora/ops/triton_ops/README_TUNING.md`, for example:

- `NVIDIA_H200_SHRINK.json`
- `NVIDIA_H200_EXPAND_TRUE.json`
- `NVIDIA_H200_FUSED_MOE_LORA_W13_SHRINK.json`
- `NVIDIA_H200_FUSED_MOE_LORA_W2_EXPAND.json`

`benchmark_lora_tuning.py` generates these files for you.

---

## 2. What the script does

High-level flow inside `main()`:

1. Checks that Triton and CUDA are available.
2. Patches `get_lora_op_configs` so that **during tuning**, the runtime
   LoRA ops use the current candidate kernel config instead of the default.
3. Builds a search space over:
   - `block_m`, `block_n`, `block_k`
   - `num_warps`, `num_stages`, `num_ctas`
   - `split_k`
4. For each `BenchmarkContext` (specified by batch/seq/hidden/rank/etc.):
   - Benchmarks all candidate configs using `bench_optype`.
   - Tracks the best (lowest median time) config.
5. If `--save-json-dir` is set:
   - Writes the best config to a JSON file named according to the kernel type
     and GPU name (`torch.cuda.get_device_name()`).
   - The JSON structure matches `get_lora_op_configs` expectations:
     `config_data[max_loras][num_slices][m][k][n]` for regular LoRA, and
     `config_data[max_loras][num_slices][m][k][n][i]` for fused MoE LoRA.
6. Optionally auto-infers dimensions:
   - When `--model-path` is provided (and `transformers` is installed),
     `hidden_size`, `num_experts`, `moe_intermediate_size`, and `top_k_num`
     can be filled from the base model config if not explicitly set.
   - When `--lora-path` points to a LoRA weights directory, `lora_rank`
     can be inferred from `adapter_config.json` or `.safetensors` files
     if not passed explicitly.

Note: CSV timing data is still streamed through the `csv` module, but in the
current version it is written to `os.devnull`, so **no `.csv` file is created**.
The only persisted outputs are the tuned JSON configs.

---

## 3. Command-line arguments

### 3.1 Core arguments

- `--op-type`  
  LoRA op to tune:

  ```text
  lora_shrink
  lora_expand
  fused_moe_lora_gate_up_shrink
  fused_moe_lora_gate_up_expand
  fused_moe_lora_down_shrink
  fused_moe_lora_down_expand
  ```

- `--dtype`  
  Data type string, e.g.:

  ```text
  torch.float16
  torch.bfloat16
  torch.float32
  ```

- `--batch-size` / `--seq-length`  
  Defines the logical M dimension: `M = batch_size * seq_length`.

- `--hidden-size`  
  Hidden size of the model:

  - For `lora_shrink`: `K = hidden_size`, `N = lora_rank`.
  - For `lora_expand` and fused MoE: `K = lora_rank`, `N = hidden_size`.
  - Required unless it can be **auto-inferred** from `--model-path`
    (see section 3.6).

- `--lora-rank`  
  LoRA rank (N dimension in shrink).
  - Required unless it can be **auto-inferred** from `--lora-path`
    (see section 3.6).

- `--num-loras` / `--num-active-loras`  
  Total number of LoRA adapters and number of active LoRAs.
  If `--num-active-loras` is omitted, it defaults to `num_loras`.

- `--num-slices`  
  Number of slices for the LoRA kernel (e.g. 1 or 2).

### 3.2 Benchmarking behavior

- `--sort-by-lora-id` (int, 0 or 1)  
  Whether to sort tokens by LoRA ID in the benchmark.

- `--arg-pool-size`  
  Number of `BenchmarkTensors` instances reused for timing to reduce caching
  effects. Values like 8–32 are typical. Larger values increase tuning time.

- `--cuda-graph-nops`  
  If set, uses CUDA Graph to benchmark `N` consecutive ops inside a captured
  graph, which reduces Python overhead but increases the per-run cost.

- `--test-correctness`  
  If set, calls the reference implementation to verify correctness **for
  `lora_shrink` and `lora_expand` only**. Correctness testing is not supported
  for fused MoE LoRA ops.

- `--max-configs`  
  Optional limit on the number of kernel configs to benchmark. Useful for
  quick runs:

  ```bash
  --max-configs 512
  ```

### 3.3 Output control

- `--save-json-dir` (recommended)  
  Directory to write tuned JSON configs. For example:

  ```bash
  --save-json-dir /model/hdj/tuning
  ```

  Files are named as:

  - `"{gpu_name}_SHRINK.json"` for `lora_shrink`
  - `"{gpu_name}_EXPAND_{add_inputs}.json"` for `lora_expand`
  - `"{gpu_name}_FUSED_MOE_LORA_W13_SHRINK.json"` / `EXPAND.json`
  - `"{gpu_name}_FUSED_MOE_LORA_W2_SHRINK.json"` / `EXPAND.json`

  The script automatically normalizes `gpu_name` (spaces and `-` replaced by `_`).

- `--json-overwrite`  
  When the target JSON file already exists:

  - If `--json-overwrite` **is not** set, the script will create a new file
    with a `_tuned_{idx}.json` suffix.
  - If `--json-overwrite` **is** set, the existing file is overwritten.

- `--output-csv`  
  Path to an output CSV file (kept for CLI compatibility). Internally, the
  script still streams timing data through the `csv` module but writes it
  to `os.devnull`, so **no `.csv` file is actually created** and timing
  info is not persisted in CSV form.

### 3.4 Fused MoE-specific arguments

These are relevant when `--op-type` is one of the fused MoE LoRA ops:

- `--num-experts`  
  Number of experts in the MoE layer. Default: `8`.

- `--top-k-num`  
  Top-K routing value used in MoE. Default: `2`.

- `--moe-intermediate-size`  
  Intermediate size of the MoE layer. When **saving JSON for fused MoE LoRA**,
  this value becomes the final index `i` in:

  ```text
  config_data[max_loras][num_slices][m][k][n][i]
  ```

  To generate usable fused MoE LoRA JSON for runtime, you must provide the
  correct intermediate size of your model’s MoE FFN (often equal to
  `intermediate_size` or `moe_intermediate_size` in the model config).

### 3.5 Multi-M tuning

- `--m-values`  
  A list of explicit `M` (num_tokens) values to tune in a single run. When
  provided, the script creates one `BenchmarkContext` per `M` with:

  ```text
  batch_size = M
  seq_length = 1
  ```

  and tunes each context sequentially, saving all results into the same JSON.

### 3.6 Dimension auto-inference (optional)

These convenience arguments let the script infer dimensions for you instead
of requiring every value on the command line:

- `--model-path`  
  A base model path or Hugging Face repo id. When tuning fused MoE LoRA ops
  and `transformers` is installed, the script will try to infer:

  - `hidden_size` (e.g. from `hidden_size`, `n_embd`, or `d_model`),
  - `num_experts` (e.g. `num_experts`, `moe_num_experts`, `num_local_experts`,
    `n_routed_experts`),
  - `moe_intermediate_size` (e.g. `moe_intermediate_size`,
    `ffn_hidden_size`, `intermediate_size`),
  - `top_k_num` (e.g. `num_experts_per_tok`, `num_experts_per_token`,
    `moe_top_k`, `top_k`),

  filling in any of these that you did not explicitly set.

- `--lora-path`  
  Directory containing LoRA adapter weights. The script looks for common
  config and weight files such as:

  - `adapter_config.json`, `adapter_config.bin`, `config.json`, and
  - `*.safetensors` (preferring keys that contain `lora_`).

  If a valid rank can be inferred from these files, `lora_rank` is filled
  automatically when you omit `--lora-rank`.

> Note: if auto-inference fails (e.g. files are missing or not readable),
> you must still provide `--hidden-size` / `--lora-rank` explicitly. The
> script will raise a clear error in that case.

---

## 4. Typical usage examples

### 4.1 Basic `lora_shrink` tuning (single workload)

```bash
python3 benchmark_lora_tuning.py \
  --op-type lora_shrink \
  --dtype torch.float16 \
  --batch-size 16 \
  --seq-length 1 \
  --hidden-size 5120 \
  --lora-rank 16 \
  --num-loras 4 \
  --num-active-loras 4 \
  --num-slices 1 \
  --arg-pool-size 8 \
  --max-configs 50 \
  --save-json-dir /model/hdj/tuning \
  --json-overwrite
```

This will:

- Search over the first 50 configs from the search space.
- Tune a single workload (`M=16`, `K=5120`, `N=16`).
- Save the best config into `/model/hdj/tuning/{gpu_name}_SHRINK.json`.

### 4.2 `lora_expand` tuning (with add_inputs flag)

```bash
python3 benchmark_lora_tuning.py \
  --op-type lora_expand \
  --dtype torch.float16 \
  --batch-size 16 \
  --seq-length 1 \
  --hidden-size 5120 \
  --lora-rank 16 \
  --num-loras 4 \
  --num-active-loras 4 \
  --num-slices 1 \
  --expand-add-inputs 0 \
  --arg-pool-size 8 \
  --max-configs 50 \
  --save-json-dir /model/hdj/tuning \
  --json-overwrite
```

This writes a config into:

```text
{gpu_name}_EXPAND_FALSE.json
```

If you set `--expand-add-inputs 1`, the filename becomes
`{gpu_name}_EXPAND_TRUE.json`.

### 4.3 Fused MoE LoRA tuning (gate_up / down)

Gate/Up projection (`w13`):

```bash
python3 benchmark_lora_tuning.py \
  --op-type fused_moe_lora_gate_up_shrink \
  --dtype torch.float16 \
  --batch-size 64 \
  --seq-length 1 \
  --hidden-size 8192 \
  --lora-rank 64 \
  --num-loras 8 \
  --num-active-loras 8 \
  --num-slices 2 \
  --num-experts 8 \
  --top-k-num 2 \
  --moe-intermediate-size 14336 \
  --arg-pool-size 8 \
  --max-configs 256 \
  --save-json-dir /model/hdj/tuning \
  --json-overwrite
```

This will write the tuned config into:

```text
{gpu_name}_FUSED_MOE_LORA_W13_SHRINK.json
```

Similarly for `fused_moe_lora_gate_up_expand`, `fused_moe_lora_down_shrink`,
and `fused_moe_lora_down_expand`, with appropriate filenames as documented
in `README_TUNING.md`.

> Important: for fused MoE LoRA JSON to be usable at runtime, you **must**
> pass a correct `--moe-intermediate-size` so that `get_lora_op_configs(...)`
> can index `[i]` properly.

### 4.4 Longer tuning runs (larger search + workload)

To run a heavier search that may take tens of minutes:

```bash
python3 benchmark_lora_tuning.py \
  --op-type lora_shrink \
  --dtype torch.float16 \
  --batch-size 64 \
  --seq-length 16 \
  --hidden-size 8192 \
  --lora-rank 64 \
  --num-loras 8 \
  --num-active-loras 8 \
  --num-slices 2 \
  --arg-pool-size 8 \
  --cuda-graph-nops 16 \
  --max-configs 512 \
  --save-json-dir /model/hdj/tuning \
  --json-overwrite
```

This keeps the search space fairly large and uses CUDA Graph to amortize
Python overhead, at the cost of longer per-config benchmarking time.

---

## 5. JSON structure and runtime consumption

The script writes JSONs that are loaded at runtime by
`vllm/vllm/lora/ops/triton_ops/utils.py::get_lora_op_configs` via
`load_lora_op_config`. The expected structure is:

```text
config_data[max_loras][num_slices][m][k][n] = cfg
```

for regular LoRA (`shrink`/`expand`), and:

```text
config_data[max_loras][num_slices][m][k][n][i] = cfg
```

for fused MoE LoRA, where:

- `max_loras`: number of LoRA adapters supported.
- `num_slices`: LoRA slices.
- `m`: M dimension (`batch_size * seq_length`).
- `k`, `n`: matrix dimensions as defined by `OpType.mkn(...)`.
- `i`: MoE intermediate size (only for fused MoE LoRA).

At runtime, vLLM will:

1. Compute `(max_loras, num_slices, m, k, n, i)` from the actual workload.
2. Look up the closest matching keys in the JSON.
3. Fall back to default configs if no JSON is available.

By using `benchmark_lora_tuning.py` to populate the JSON files under
`VLLM_TUNED_CONFIG_FOLDER`, you ensure that the LoRA Triton kernels use
*measured* best configs instead of generic defaults.

---

## 6. Relationship to `benchmark_lora.py`

- `benchmark_lora.py` is a **general benchmarking tool** for LoRA, with
  multiple modes (`list_bench`, `range_bench`, `model_bench`) and support for
  saving raw `torch.utils.benchmark.Measurement` data to pickle/CSV.
- `benchmark_lora_tuning.py` is a **focused tuner**:
  - Uses a fixed search space.
  - Integrates directly with runtime `get_lora_op_configs`.
  - Writes only the final tuned configs (JSON) needed for production use.

In a typical workflow, you:

1. Use `benchmark_lora.py` for exploration and performance analysis.
2. Use `benchmark_lora_tuning.py` to generate the JSON files actually used
   by inference.
