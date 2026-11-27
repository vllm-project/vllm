# Logprob Drift Evaluation

Some changes to vLLM alter what a model computes without moving any accuracy score: a value
rounded to bf16 too early, a stale KV block reused. This eval catches them from the model's
logprobs, usually before any answer changes.

No build reproduces itself exactly, so outputs can't simply be diffed. Instead the eval asks
whether a candidate build differs from a reference build by more than reruns of the
reference differ from each other.

## Usage

### Run tests with pytest

Record the reference build, typically `main` at the PR's merge base, then the candidate
on the same machine:

```bash
# On the reference build: 2 runs
pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py --drift-out=runs/main

# On the candidate build: 1 run, and more only if it differs from the reference's
pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py \
    --drift-out=runs/pr --drift-reference=runs/main
```

If the candidate's first run differs, the test asks for more reference runs. Record them
with `--drift-runs`, for example `--drift-out=runs/main --drift-runs=4`, and run the
candidate's test again: it resumes from the runs on disk and records more of its own.

Each run starts its own server with empty compile and autotune caches, about two minutes
with the small model. Two builds that read the same cost 3 runs in all, sometimes 6, and a
change costs 9; the candidate's test then fails, and `result.json` holds the comparison. A
reference also serves the next candidate compared with that build on that machine.

### Run standalone evaluation script

```bash
# Start each run's server with empty caches
C=$(mktemp -d)
VLLM_CACHE_ROOT=$C TRITON_CACHE_DIR=$C/triton VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$C/flashinfer \
    vllm serve Qwen/Qwen3-0.6B --max-model-len 2048 --max-num-seqs 32

python -m tests.evals.logprob_drift.logprob_drift_eval record --out runs/main/1.npz
python -m tests.evals.logprob_drift.logprob_drift_eval compare \
    --reference runs/main/*.npz --candidate runs/pr/*.npz
```

`compare` takes more runs of the reference than of the candidate. It exits 0 for same, 1
for changed and 3 when it needs more runs to tell.

For example, replaying the bug fixed in
[#42379](https://github.com/vllm-project/vllm/pull/42379), with the fix as the reference
and its parent as the candidate, on Qwen/Qwen3-0.6B with `--enforce-eager`:

```text
First runs: probe KL 0.001111, NLL -0.036 %, same top token 97.99 %, 48.0 % of outputs differ
Across builds over rerun noise, 5 reference and 4 candidate runs: probes ∞×, decode ∞×, traffic 1.13×
Verdict: CHANGED, ∞× is above the 3.20× needed; identical builds read this way 1 in 126 times
Where: probes, 32 of 32 chunks differ more across builds than between reruns; most, chunk 21, at position 404 (token ids, probabilities)
  reference: 22235 (0.546), 6755 (0.147), 6569 (0.026)
  candidate: 22235 (0.463), 6755 (0.181), 6569 (0.032)
```

The first line says how far apart the builds' first runs are, the second how each kind of
request scores against rerun noise, and the last where the change shows most.

## How it works

Each run sends three kinds of requests:

- **Probes**: 32 chunks of 1,024 WikiText-2 tokens, scored with prompt logprobs on their last
  512 positions, as llama.cpp's `--kl-divergence` does. Both builds see identical tokens, so
  a kernel that rounds differently stands out even when no answer changes.
- **Decode**: the first half of each chunk, continued greedily for 128 tokens, one request
  at a time, so decode kernels run without batching.
- **Traffic**: 250 GSM8K questions behind a shared 5-shot prefix, generated greedily, more
  than fit in one batch. This runs prefix caching, chunked prefill mixed with decode,
  CUDA-graph padding and KV block reuse, which prompt logprobs never reach. Two outputs are
  compared up to and including their first differing token.

Each kind is scored as the mean KL across builds over the mean KL between reruns, about 1×
when nothing changed, and the score is the largest of the three. KL is taken over each
side's top 20 tokens, a lower bound on the true KL. Traffic takes its noise from the
reference's reruns alone, since a bug that corrupts the KV cache at random makes the
candidate's runs disagree with each other.

The verdict is a permutation test. If the builds are the same, the real split of the runs
into reference and candidate is just one of many equally likely splits, so it rarely scores
highest. The builds separate when it scores above every other split. There are no
thresholds to tune, so the rate below holds for any model, settings or GPU, provided
nothing but the build differs between runs:

| Reference runs | Candidate runs | Identical builds separate | Then |
| --- | --- | --- | --- |
| 2 | 1 | 1 time in 3 | same: stop. Separate: record 3 more |
| 4 | 2 | 1 time in 15 | same: stop. Separate: record 3 more |
| 5 | 4 | 1 time in 126 | same, or changed |

Most builds checked are the same, so most checks stop after 3 or 6 runs. For an intended
change, the first stage already shows how large it is and where.

## Configuration Format

```yaml
model_name: "Qwen/Qwen3-0.6B"
server_args: "--max-model-len 2048 --max-num-seqs 32"
env:                            # Environment variables (optional)
  VLLM_USE_FLASHINFER_SAMPLER: "0"
startup_max_wait_seconds: 1200  # Optional
```

Runs record their `server_args`, `env`, model, GPU type and data, and `compare` refuses runs
that differ in any of them. A check covers only the code paths these settings run.

## Limits

- It finds changes, not bugs: an intended numerics change reads as changed too.
- It misses a change smaller than the score needed, which every verdict prints, and a bug
  that strikes too few runs.
- A reference run that landed apart from the others, by however little, can hide a change
  from the candidate's first runs.
- Record both builds back to back on the same machine. If the machine changes in between,
  say a driver update, identical builds can read as changed.
- It compares two builds serving one configuration, so a PR that adds a configuration has
  no reference. It compares probabilities, so a build that only cuts outputs short reads
  the same.
