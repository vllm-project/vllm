# Runtime Tuning

The Recipes JSON remains the baseline. vLLM Recipes already provide validated
model, hardware, strategy, environment variables, and serving arguments.
Runtime tuning is optional and is intended for parameters whose best value can
depend on the actual deployment resources or expected request workload.

## Information Sources

Runtime tuning combines up to three inputs:

- **vLLM Recipe** — the required baseline from the Recipes JSON API or a direct
  recipe JSON file.
- **Hardware information (optional)** — detected with `--detect-hardware`,
  including effective CPU/NUMA topology and memory availability.
- **Workload information (optional)** — supplied through CLI hints such as input
  and output token lengths, concurrency, TTFT/TPOT objectives, and target QPS.

If no optional hardware or workload information is supplied, the converter keeps
the normal Recipes conversion behavior.

See [Deployment-Time Parameters](#deployment-time-parameters) for the mapping
between these inputs and individual runtime parameters.

## Hardware Information

Hardware detection is enabled only with `--detect-hardware`. It reports the
effective resources available to the process or container, including CPU/NUMA
topology and memory information. The current policy can use those values to
refine `tensor-parallel-size` and `gpu-memory-utilization`.

```bash
python3 tools/recipes/recipe_json_to_vllm_config.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware xeon6 \
  --detect-hardware
```

Hardware detection is optional. Recipe hardware selects the tuning policy; host
hardware detection provides resource information to that policy.

## Workload Information

Workload information is supplied explicitly because the converter runs before a
vLLM server exists. Supported hints include:

- `--input-tokens`
- `--output-tokens`
- `--concurrency`
- `--ttft-sla-ms`
- `--tpot-sla-ms`
- `--target-qps`

Example:

```bash
python3 tools/recipes/recipe_json_to_vllm_config.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware xeon6 \
  --input-tokens 128 \
  --output-tokens 128 \
  --concurrency 32 \
  --ttft-sla-ms 3000 \
  --tpot-sla-ms 100
```

`--output-tokens` is used to estimate steady-state request turnover for
`max-num-batched-tokens`. When both `--target-qps` and `--tpot-sla-ms` are
available, they are also used to estimate prompt arrival pressure per scheduler
step. `--ttft-sla-ms` is collected but is not yet converted directly into a
batch-size formula because the relationship between TTFT and scheduler budget is
model- and hardware-dependent.

## Deployment-Time Parameters

The parameters below may already exist in a recipe. They are candidates for
deployment-time refinement when the recipe value is missing, generic, or based
on a validation environment that differs from the user's target deployment.

| Runtime parameter | Why it may need deployment-time refinement | Main decision input | Current policy |
| --- | --- | --- | --- |
| `tensor-parallel-size` | The effective CPU/NUMA topology available to a container or pod can differ from the system used to validate the recipe. | Hardware topology | Use the largest power-of-two TP value that does not exceed the effective NUMA-node count. |
| `gpu-memory-utilization` | Available memory can differ by machine size, container limits, and other memory use. The vLLM option name is also used by the CPU backend. | Hardware memory + recipe baseline | Calculate a conservative fraction from the most constrained NUMA node. |
| `max-num-seqs` | The useful scheduler concurrency depends on the number of requests expected to be active at the same time. | Workload concurrency | Set `max-num-seqs` to `--concurrency` when supplied. |
| `max-num-batched-tokens` | Each scheduler iteration must share its token budget between active decodes and incoming prefills. | Input/output token shape, concurrency, and optional QPS/TPOT | Calculate decode budget + expected prefill demand, with vLLM scheduler constraints as floors. |
| `data-parallel-size` | The required replica count depends on the requested throughput and the capacity of one replica. | Capacity target | Keep the recipe value today because per-replica SLA capacity is not known. |

## How Each Runtime Parameter Is Calculated

### `tensor-parallel-size`

TP is derived from the effective NUMA topology reported by hardware detection:

```text
TP = largest power of two <= effective NUMA-node count
```

For example, 2 effective NUMA nodes produce TP=2, while 6 nodes currently
produce TP=4. This is a topology-based starting point and avoids automatically
selecting unusual non-power-of-two TP sizes before they are validated.

### `gpu-memory-utilization`

For every effective NUMA memory node, the policy calculates:

```text
node_safe_fraction = available_memory / total_memory - 0.10
safe_fraction = min(0.90, minimum node_safe_fraction)
```

The 10% reserve leaves memory for model/runtime overhead outside the vLLM cache
budget. If the recipe already contains a smaller value, the policy preserves the
smaller value. If it is absent, `0.80` is used as the initial ceiling:

```text
candidate = min(recipe value or 0.80, safe_fraction)
```

### `max-num-seqs`

When `--concurrency` is supplied, it directly represents the requested maximum
number of simultaneously active requests:

```text
max-num-seqs = concurrency
```

If concurrency is not supplied, the converter does not override this parameter.

### `max-num-batched-tokens`

The token budget is workload-derived and no longer assumes a fixed number of
parallel prefills.

First, the policy determines the active sequence count:

```text
active_sequences =
    --concurrency
    or recipe max-num-seqs
    or vLLM default max_num_seqs (128)
```

Decode requests consume approximately one token per active sequence in a
scheduler iteration:

```text
decode_budget = active_sequences
```

The policy then estimates how many new prompts need prefill work per scheduler
step:

```text
prefills_per_step = max(
    1,
    active_sequences / output_tokens,
    target_qps * tpot_sla_ms / 1000
)

prefills_per_step = min(active_sequences, prefills_per_step)
prefill_budget = ceil(input_tokens * prefills_per_step)
```

The final scheduler budget is:

```text
max-num-batched-tokens = max(
    vLLM default max_num_batched_tokens (2048),
    active_sequences,
    decode_budget + prefill_budget
)
```

For example, with 128 input tokens, 128 output tokens, and concurrency 32:

```text
decode_budget      = 32
prefills_per_step  = max(1, 32 / 128) = 1
prefill_budget     = 128
candidate          = max(2048, 32, 32 + 128) = 2048
```

If chunked prefill is explicitly disabled and `max-model-len` is available, the
policy also ensures:

```text
max-num-batched-tokens >= max-model-len
```

### `data-parallel-size`

`--target-qps` alone is not enough to choose DP safely. A correct capacity rule
also needs measured per-replica throughput that still satisfies TTFT/TPOT:

```text
DP = ceil(target_qps / qps_per_replica_at_SLO)
```

Because the converter does not have that measured capacity yet, it intentionally
keeps the recipe DP value rather than guessing.

## Precedence

```text
vLLM defaults
    -> vLLM Recipes baseline
        -> hardware refinement (optional)
            -> workload / SLO refinement (optional)
                -> config.yml + env.sh
```

## Runtime-Tuning Hardware Scope

Runtime tuning is selected from the resolved recipe JSON's `hardware` field,
rather than by inspecting which physical devices happen to be present on the
host. This matters because a GPU server also exposes its host CPU topology.

The current runtime-tuning policy registry contains `xeon6`. A tuning request
for unregistered recipe hardware, such as `b200`, fails before host hardware
detection:

```text
ERROR: Runtime tuning is not supported for recipe hardware 'b200'.
Currently supported: xeon6.
```

Plain recipe conversion remains available for all recipe hardware. The gate only
applies when optional runtime-tuning inputs are requested.

## vLLM CPU Docker Shell

Run hardware detection and runtime tuning inside the target vLLM CPU container
so detected CPU, NUMA, memory, and cgroup limits match the deployment
environment.

From the vLLM source tree:

```bash
mkdir -p recipe-output

docker run --rm -it \
  --entrypoint bash \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  --shm-size=4g \
  -p 8000:8000 \
  -v "$PWD/tools/recipes:/recipes:ro" \
  -v "$PWD/recipe-output:/output" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /output \
  vllm/vllm-openai-cpu:latest-x86_64
```

`/output` is writable; `/recipes` remains read-only.

Inside the container, generate one runtime-tuned initial configuration:

```bash
python3 /recipes/recipe_json_to_vllm_config.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware xeon6 \
  --detect-hardware \
  --input-tokens 128 \
  --output-tokens 128 \
  --concurrency 32 \
  --ttft-sla-ms 3000 \
  --tpot-sla-ms 100 \
  --config-out /output/config.yml \
  --env-out /output/env.sh
```

Deploy the generated initial configuration:

```bash
source /output/env.sh
vllm serve --config /output/config.yml
```

For optional benchmark-backed validation of the scheduler settings, continue
with [SWEEP_TUNING.md](SWEEP_TUNING.md).

## Implementation

- `hardware_detection.py` collects effective CPU/NUMA/memory information.
- `runtime_tuning.py` owns hardware-policy selection and independent parameter
  tuning functions.
- `recipe_json_to_vllm_config.py` resolves the recipe, collects optional inputs,
  applies the selected policy, and generates `config.yml` and `env.sh`.

For benchmark-backed validation of the initial scheduler suggestion, see
[SWEEP_TUNING.md](SWEEP_TUNING.md).
