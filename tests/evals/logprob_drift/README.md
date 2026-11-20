# Logprob Drift Evaluation

Measures how far a candidate build's next-token logprobs move from a reference build's, for the same model on the same GPU type. It catches numerical drift from kernels, quantization and attention backends, often before a task score such as GSM8K accuracy moves.

The eval scores fixed passages by teacher forcing: each passage is sent as a prompt to `/v1/completions` with `prompt_logprobs` and `return_token_ids`, one request at a time, so every position's logprobs are conditioned on the same text in both builds. No engine change is needed; any vLLM build whose completions endpoint has those two fields, and returns a `rank` with each prompt logprob, can be scored.

The passages are the first 64 GSM8K test questions with their worked answers, from the same dataset files the [gsm8k eval](../gsm8k/README.md) downloads, so no data is checked in.

## Usage

### Run the test with pytest

```bash
pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py \
    --config-list-file=configs/models-small.txt
```

For each config, this starts one server, records a reference, compares the same server against it and requires the gates to pass. It checks the eval end to end; it does not compare two builds or check server restarts (see below).

### Compare two builds

Record on the reference build (for a PR, the merge-base), then compare on the candidate (the PR head), with the same model, server arguments and GPU type. Run the script from the repository root:

```bash
# Reference build
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
python -m tests.evals.logprob_drift.logprob_drift_eval record \
    --port 8000 --output reference.json
# Stop the reference server, then start the candidate build
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
python -m tests.evals.logprob_drift.logprob_drift_eval compare \
    --port 8000 --reference reference.json --save-results result.json
```

`record` and `compare` must be given the same `--server-args`, or neither, or `compare` stops with an error. The script stores the string you type; it does not read the server's configuration, so the check catches a forgotten or mistyped argument, not a server started differently. The string is saved in the reference file, so leave secrets such as `--api-key` out of it. `record` scores the passages `--num-runs` times (default 2) and stores the first pass, its metadata (model, vLLM version, `--server-args` if given, a SHA-256 of the passages, top-k) and the drift between the passes as the reference's rerun noise. `compare` scores the same passages once on the candidate and gates the drift. Nothing else may send requests to the server while either runs.

`--top-k` (default 20) must not exceed the server's `--max-logprobs`. The script stops with an error if the server does not return a position's own-token logprob, returns fewer than top-k logprobs, or returns values that cannot be logprobs (above 1e-4, or summing to more than probability 1, as raw logits often do). These checks catch many but not all non-default `--logprobs-mode` settings; serve both builds with the default.

Exit codes: 0 when the gates pass, 1 when one fails, 2 on an error (server unreachable, missing logprobs or files, a different model, tokenization or dataset, or a reference with too few passes for its rerun noise).

## Reading the output

| Metric | Meaning |
| --- | --- |
| Identical positions | Positions whose own-token logprob and top-k logprobs are bit-identical to the reference. |
| Mean NLL change | Relative change of the mean negative log-likelihood of the passages' own tokens. Exact: every position's own-token logprob is returned, also below the top-k. |
| Top-1 agreement | Share of positions where a token tied for first in the reference is also tied for first in the candidate. Ties are seen only among the top-k. |
| Top-k overlap | Mean share of the reference's top-k tokens that are in the candidate's top-k. |
| KL estimate | KL(reference ‖ candidate) over the reference's top-k plus one bucket for the rest of the vocabulary, in nats per position. An estimate, see below. |

### Gates and their defaults

Each gate's bar is the looser of the reference's rerun noise and a tolerance given to `compare`:

- `--max-nll-rel-change` (default 0): allowed absolute relative change of the mean NLL.
- `--min-top1-agreement` (default 1.0): lowest allowed top-1 agreement.
- `--max-kl-estimate` (default 0): allowed absolute KL estimate.

The defaults add nothing to the rerun noise, because the measured rerun noise of this scoring is zero. With one request at a time, repeated scoring passes on one server returned bit-identical logprobs at all 11,744 positions in every run (`Qwen/Qwen2.5-1.5B-Instruct`, vLLM 0.33.0, one RTX 5090). Restarts are covered below. The reruns on vLLM 0.32.0 reported in #61698 were bit-identical too. So by default a change that moves the mean NLL, the top-1 agreement or the KL estimate (in either direction) at all fails, and the report says how large it is. The count of bit-identical positions is reported but not gated.

A fixed absolute bar would hide real bugs: two past numerics fixes, #42379 (RMSNorm weight upcast) and #31931 (MoE router weight cast), changed the logprobs at every position, yet #42379 read a mean NLL change of +0.07% with a top-1 agreement of 99.3% (`Qwen/Qwen3-0.6B`), and #31931 a mean NLL change of +0.07% with a top-1 agreement of 98.1% (`ibm-granite/granite-3.1-1b-a400m-base`). Both pass both halves of a bar such as 1% NLL and 97% top-1. These numbers were measured with an earlier version of this eval.

A PR that is meant to change numerics (a new kernel, a quantization scheme) is expected to fail the default gates. Its author reads the drift and, if it is acceptable, states it in the PR or reruns `compare` with tolerances chosen for that change. When the reference's own passes differ (`rerun_noise.identical` is false in the reference file), the bars widen to the measured noise, which is the largest drift over `--num-runs` minus 1 reruns. A candidate identical to the reference has a drift drawn like the rerun drifts, so each of them is equally likely to be the largest, whatever the noise distribution. The candidate then exceeds the bar with chance 1 in `--num-runs` for each such gate; with 2 passes it fails each such gate about half the time. Over several gates the chance is at most their number divided by `--num-runs`. `compare` keeps that chance at or below 1 in 20: it stops with an error, which names the `--num-runs` needed, unless the reference has 20 passes for each gate whose bar the noise sets (60 for all three), or every tolerance is looser than the measured noise: above it for the NLL change and KL estimate, below it for top-1 agreement. The 1 in 20 is a chosen risk, not a measurement.

The rerun noise is measured inside one server process, but comparing two builds always crosses processes. Choices made at start-up (autotuning, compilation, CUDA-graph capture) are fixed within a process, so noise between server starts does not enter the rerun noise. Restarts are usually bit-identical but not always. On one RTX 5090, two batches of 5 and 15 separately started servers matched within each batch at every position (115 pairs). In 7 comparisons of a reference with a freshly started server, one differed: top-1 agreement 100%, estimated KL 2e-12. Before relying on the zero defaults for a new model or GPU type, compare the reference against a few freshly started servers of the same build. If one fails, the drift it shows is the size of the restart noise, and tolerances above it are needed. The CI test records and compares on one server, so it does not depend on restarts.

## Limits

- **The KL estimate is neither an upper nor a lower bound.** The candidate's probability of a reference top-k token that is outside the candidate's own top-k is unknown; the script gives it an equal share of the candidate's remaining mass, capped at the candidate's k-th probability. When the candidate's top-k holds all its mass, that share is floored at probability 1e-12, so the token adds up to 27.6 nats times its reference probability. The estimate can even be negative. Use it for trends; the NLL change and top-1 agreement are exact. `/v1/completions` cannot return the logprob of a chosen token outside the top-k, so the eval cannot score the reference's tokens on the candidate exactly.
- **Prefill only.** Prompt logprobs are computed in prefill, so they do not pass through the decode path; requests that ask for them also skip prefix-cache reads by default. Each request still decodes one token, which the eval ignores. A bug on the decode path (block reuse, CUDA-graph padding, a stale cache slot) does not show here. Two past bugs of that kind, #30915 (FlexAttention block reuse) and #36580 (CUDA-graph padding), left prompt logprobs at the rerun noise and showed only in batched greedy generation (see the discussion in #61698).
- **Same GPU type and server arguments.** Kernels differ between GPU generations, so a reference recorded on one GPU type does not transfer to another.
- **One request at a time.** Concurrent scoring would add batch-composition noise that this eval does not measure.

## Related work

- #65815 (open) measures the KL of a KV-cache dtype against the `auto` cache within one build, from the client side. This eval compares two builds at the same settings.
- #35961 computes full-vocabulary KL divergence inside the engine against stored reference logits, which needs engine changes. This eval runs against an unmodified server, at the cost of the top-k estimate above.

## Configuration Format

Model configs in the `configs/` directory use this YAML format:

```yaml
model_name: "Qwen/Qwen2.5-1.5B-Instruct"
num_passages: 64                    # GSM8K test passages to score
top_k: 20                           # Logprobs per position
server_args: "--max-model-len 4096" # Arguments for `vllm serve`
env:                                # Environment variables (optional)
  VLLM_LOGGING_LEVEL: "DEBUG"
```
