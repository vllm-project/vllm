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

For each config, this records a reference on 3 freshly started servers, each with its own empty vLLM, Triton and FlashInfer-autotune cache directories (FlashInfer's compiled-kernel cache stays shared). If they disagree, the model has restart noise on that machine and the test checks that `compare` stops with its restart-noise error (see "Server restarts"). If they agree, it compares the last server against the reference, which should pass; the test's docstring gives the small chance that it does not. Which of the two paths a run takes depends on the machine; the CPU tests cover both. It does not compare two builds.

### Compare two builds

Record on the reference build (for a PR, the merge-base), then compare on the candidate (the PR head), with the same model, server arguments and GPU type. Run the script from the repository root:

```bash
# Reference build, on 3 freshly started servers, each with new empty caches
for i in 1 2 3; do
    C=$(mktemp -d)
    VLLM_CACHE_ROOT=$C TRITON_CACHE_DIR=$C/triton VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$C/flashinfer \
        vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 &
    SERVER=$!
    timeout 1200 sh -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done'
    # If record warns that the passes differ, start again in a new file with the --num-runs it names for every server
    python -m tests.evals.logprob_drift.logprob_drift_eval record \
        --port 8000 --output reference.json $([ $i -gt 1 ] && echo --append)
    kill $SERVER; wait $SERVER
done
# Candidate build, on a freshly started server with new empty caches
C=$(mktemp -d)
VLLM_CACHE_ROOT=$C TRITON_CACHE_DIR=$C/triton VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$C/flashinfer \
    vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 &
SERVER=$!
timeout 1200 sh -c 'until curl -sf localhost:8000/health > /dev/null; do sleep 5; done'
python -m tests.evals.logprob_drift.logprob_drift_eval compare \
    --port 8000 --reference reference.json --save-results result.json
kill $SERVER; wait $SERVER
```

`record` and `compare` must be given the same `--server-args`, or neither, or `compare` stops with an error. The script stores the string you type; it does not read the server's configuration, so the check catches a forgotten or mistyped argument, not a server started differently. The string is saved in the reference file, so leave secrets such as `--api-key` out of it. `record` scores the passages `--num-runs` times (default 2) and stores the first pass, its metadata (model, vLLM version, `--server-args` if given, a SHA-256 of the passages, top-k) and the drift between the passes as the reference's rerun noise. `compare` scores the same passages once on the candidate and gates the drift. Nothing else may send requests to the server while either runs.

`--top-k` (default 20) must not exceed the server's `--max-logprobs`. The script stops with an error if the server does not return a position's own-token logprob, returns fewer than top-k logprobs, or returns values that cannot be logprobs (above 1e-4, or summing to more than probability 1, as raw logits often do). These checks catch many but not all non-default `--logprobs-mode` settings; serve both builds with the default.

Exit codes: 0 when the gates pass, 1 when one fails, 2 on an error (server unreachable, missing logprobs or files, a different model, tokenization or dataset, or a reference with too few passes for its rerun noise), 3 when the reference's own servers disagree (restart noise, see below; it is checked first, so it wins over too few passes); `record --append` also exits 3 then, after saving the recording; such a reference cannot gate a candidate, so start a new file. When the servers' own passes differ, rerun noise alone can also make them disagree, and this check then reports it as restart noise.

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

A PR that is meant to change numerics (a new kernel, a quantization scheme) is expected to fail the default gates. Its author reads the drift and, if it is acceptable, states it in the PR or reruns `compare` with tolerances chosen for that change. When the reference's own passes differ (on any of its servers), the bars widen to the measured noise, which is the largest drift over `--num-runs` minus 1 reruns. A candidate identical to the reference has a drift drawn like the rerun drifts, so each of them is equally likely to be the largest, whatever the noise distribution. The candidate then exceeds the bar with chance 1 in `--num-runs` for each such gate; with 2 passes it fails each such gate about half the time. Over several gates the chance is at most their number divided by `--num-runs`. `compare` keeps that chance at or below 1 in 20: it stops with an error, which names the `--num-runs` needed, unless the reference has 20 passes for each gate whose bar the noise sets (60 for all three), or every tolerance is looser than the measured noise: above it for the NLL change and KL estimate, below it for top-1 agreement. The 1 in 20 is a chosen risk, not a measurement, and with several servers it assumes they share one rerun-noise distribution.

### Server restarts

The rerun noise is measured inside one server process, but comparing two builds always crosses processes. A server makes choices at start-up (autotuning, compilation, CUDA-graph capture) that stay fixed for its lifetime, so the noise between server starts never enters the rerun noise. Measured on one RTX 5090, 8 freshly started servers per row, one recording each, grouped by bit-identity:

| vLLM | Model | Caches | States among 8 servers |
| --- | --- | --- | --- |
| 0.34.0 | Qwen/Qwen2.5-1.5B-Instruct | shared | 1 |
| 0.34.0 | Qwen/Qwen2.5-1.5B-Instruct | empty | 4 |
| 0.34.0 | Qwen/Qwen3-0.6B | empty | 1 |
| 0.22.1 | Qwen/Qwen2.5-1.5B-Instruct | empty | 1 |

In the 4-state row most servers differed from the first one by a few positions (KL near 1e-11), but one kept only 397 of 11,744 positions identical (top-1 agreement 99.86%, KL 7e-4), the size of a real numerics fix. A shared cache hides this: every restart replays the first server's start-up choices, while a candidate build cannot reuse them and makes its own. So record the reference on freshly started servers with empty caches: the first with `record --output`, the others with `record --output <same file> --append`, which checks that they run the same model, vLLM version, server arguments, passages and top-k. It cannot tell whether the server was restarted: recording the same running server twice gives two identical recordings when its passes are identical, so start each one afresh. It also cannot tell builds apart that report the same version string, such as source installs that report `dev`. Three servers are a sensible minimum; in the 4-state row, 5 of the 28 pairs of servers were bit-identical (18%), so a 2-server reference would miss it about 1 time in 5.

`compare` then checks the reference before scoring anything. If its servers are bit-identical, the gates take their bars as before (zero, or the worst rerun noise of the servers) and the report says how many servers agreed. If they disagree, it exits 3 and lists the states: a candidate would fail the gates without any change of its own, and telling a change from this noise needs a comparison over repeated server starts, which this eval leaves to a later change (the reference keeps every server's recording, and `recording_distance` gives the distance between two of them, for that purpose). A reference from a single server still works as before, with a note that restart noise is unchecked. A candidate that fails can be in a rare start-up state the reference servers missed: start it again with empty caches and compare once more. A second failure makes a change likely, but an unchanged build can land in the same unseen state twice, so it does not rule out restart noise.

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
