# Text watermarking

Text watermarking embeds a statistical signal in generated token choices. A
detector that knows the generation parameters can test for this signal without
access to the model weights.

The design and scope are discussed in
[RFC #53916](https://github.com/vllm-project/vllm/issues/53916).

## Configuration

Enable watermarking by configuring an algorithm and secret key at engine
startup:

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"gumbel","key":42}'
```

Watermarking is disabled when `--watermark-config` is omitted. Gumbel-max (see
`gumbel`) is the default algorithm within an enabled `WatermarkConfig`. When
watermarking is configured, it is enabled for requests by default.

Requests can opt out without changing the engine-level algorithm or key:

```python
from vllm import SamplingParams

sampling_params = SamplingParams(watermarking=False)
```

The OpenAI-compatible APIs accept the same `watermarking: false` request field.
Deployments that require watermarking must restrict this field to trusted
callers, or strip and validate it at the ingress boundary, so untrusted clients
cannot opt out.

`context_width` controls how many prior tokens seed each watermark decision
and defaults to 4. Larger values make the watermark less robust to
edits because an insertion, deletion, or substitution changes more subsequent
contexts. Values above 16 are allowed but emit a warning.

`allow_target_only_watermarking` defaults to false and only has an effect when
speculative decoding is enabled. It permits speculative decoding with a
watermarking algorithm that does not support it natively, at the cost of
weaker detectability. See
[Speculative decoding](#speculative-decoding).

## Architecture

`WatermarkConfig` selects an algorithm and PRF. Model Runner V2 constructs the
corresponding `Watermarker`, and `GPUWatermarkSampler` invokes it for the final
stochastic token selection after temperature, min-p, top-k, and top-p are
applied. A watermarker can either select a token directly or transform logits
and delegate to vLLM's random sampler.

Detection is separate from generation. vLLM provides detector primitives for
the reference algorithms. `WatermarkDetector` consumes token IDs, so callers
remain responsible for using the tokenizer and watermark profile that match
generation.

## Speculative decoding

Watermarking requires speculative decoding to use probabilistic draft sampling,
standard rejection sampling, and an autoregressive model-based method (`dspark`,
`eagle`, `eagle3`, or `mtp`). Parallel drafting is supported only by `dspark`.

A watermarking algorithm without native speculative-decoding support is
rejected before model loading. Set
`"allow_target_only_watermarking": true` to allow it: accepted draft tokens are
not watermarked, while target-side rejection recovery and bonus sampling remain
watermarked. The watermark signal is diluted in proportion to the share of
output tokens supplied by accepted drafts; rejected drafts do not dilute it
because their recovery tokens are watermarked.

Speculative-decoding token paths do not currently support generation-side
context deduplication. The configured `deduplicate_contexts` policy is not
applied to accepted drafts, rejection-recovery tokens, or bonus tokens.

For `dual_key_gumbel`, `alpha` has no effect under speculative decoding. The
speculative protocol selects the key for each token instead.

## Algorithms

### Gumbel-max

Gumbel-max derives a deterministic pseudorandom value from the key, prior token
context, and every candidate token, then uses the resulting Gumbel noise for
categorical sampling. See
[Aaronson's original presentation](https://simons.berkeley.edu/sites/default/files/2024-10/LLM24-2%20Slides%20-%20Scott%20Aaronson.pdf).

Gumbel-max requires stochastic sampling. Greedy requests (`temperature=0`)
bypass watermarking and emit a warning once per worker.

When a token context is repeated, generation can use ordinary sampling for that
occurrence. Reusing the repeated context leads to a bias over the sequence, as
certain token choices would be correlated. Ordinary sampling at these positions
allows for single-sequence non-distortion (see section G.3 of the
[SynthID-Text supplementary materials](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf)).
The detector independently deduplicates contexts so repeated keyed random
vectors are not treated as independent evidence, meaning that context
deduplication at generation time does not reduce the watermarking signal, unless
a user tampers with the output and removes the first occurrence while leaving an
unwatermarked repetition.

`deduplicate_contexts` controls which history is searched for a repeated
context:

- `"none"` disables context deduplication.
- `"single_turn"`, the default, searches the tokens generated for the current
  request. This gives single-turn non-distortion.
- `"all"` also searches the prompt, which extends non-distortion across the
  turns of a conversation. The first `context_width` generated tokens use
  ordinary sampling, and subsequent tokens whose context already occurs in the
  prompt are not watermarked. Previous turns of a session, or a prompt that
  already contains the structure of the answer such as a tool result the model
  extends, may therefore leave little of the answer watermarked. Given the
  impact of `"all"` on both performance (sampling steps must scan over a
  potentially large amount of previous context) and watermarking detectability
  (we may significantly reduce the amount of watermarked tokens), you should
  only use `"all"` when non-distortion across multiple turns is required.

`deduplicate_contexts_max_history` limits the search to the most recent
positions and defaults to 8,192. Each position is compared over the
`context_width` tokens before it, so the window of tokens read is that much
longer. A smaller value reduces scanning cost but only provides the guarantee
within that window. Set it to `null` to search back to the start of the
generation for `"single_turn"`, or of the request for `"all"`; an unbounded
search costs more as the sequence grows. The setting has no effect when
`deduplicate_contexts` is `"none"`. Values below 1,024 emit a warning because a
short window can miss repetition loops whose contexts recur farther apart.

For example, this checks prompt and completion history within the default
8,192-position window:

```bash
vllm serve MODEL \
  --watermark-config \
  '{"algorithm":"gumbel","key":42,"deduplicate_contexts":"all"}'
```

### Dual-key Gumbel-max

Dual-key Gumbel-max derives independent keys A and B from one configured master
key. During ordinary generation, each token uses key A with probability
`1 - alpha` and key B with probability `alpha`. The algorithm-specific `alpha`
parameter defaults to 0.1. Detection scores every token against both keys.

The same two key streams support speculative decoding without changing its
acceptance rate. In this mode, the speculative protocol selects the key instead
of `alpha`: draft tokens use key A, while rejection recovery and bonus tokens
use key B. The ordinary target-to-draft probability-ratio test remains unchanged,
implementing [SynthID-Text Supplementary Algorithm
6](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf).

Select `dual_key_gumbel` together with probabilistic drafting:

```bash
vllm serve MODEL \
  --speculative-config \
  '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --watermark-config '{"algorithm":"dual_key_gumbel","key":42,"alpha":0.1}'
```

### SynthID-Text

[SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) is planned but
not currently implemented.

## Pseudorandom functions

A watermark PRF turns the secret key, token context, and candidate token into
reproducible random values. Generation and detection must produce identical
values across devices and releases. The values should also be uniform and
independent enough for the sampling algorithm and detector statistics. PRF
selection is an advanced compatibility and performance setting; most users
should keep the default.

Watermarked generation currently supports the `philox` PRF:

- `philox` is based on the counter-based Philox4x32-10
  generator from the [Random123 paper](https://doi.org/10.1145/2063384.2063405).
  It is parallel, vectorizes on accelerators, and avoids CPU transfers, but is
  not a cryptographic PRF and does not provide key-recovery or forgery
  resistance. vLLM versions its input mapping and provides compatibility
  vectors so generation and detection remain interoperable.

## Detection

The detector primitives operate on token IDs and do not require model weights:

```python
from transformers import AutoTokenizer

from vllm.v1.watermarking import GumbelWatermarkDetector

tokenizer = AutoTokenizer.from_pretrained(MODEL)
token_ids = tokenizer.encode(text, add_special_tokens=False)
result = GumbelWatermarkDetector(key=42, prf="philox").detect(token_ids)
print(result.p_value, result.is_watermarked)
```

The detection configuration must match the generation configuration, including
the tokenizer, PRF, watermarking algorithm, algorithm-specific watermarking
configuration, and key. In practice, this information is often unavailable
when checking a piece of text. Deployments should therefore retain the set of
candidate configurations they have served, test the text against each
candidate, and correct for multiple testing, for example with a Bonferroni
correction to the resulting p-values.

Gumbel-max detection scores repeated contexts once by default so identical PRF
random vectors are not treated as independent evidence. Keep
`deduplicate_contexts=True` unless the detector's calibration has been adjusted
for correlated scores.

The reported p-value is calibrated under the assumption that scored PRF inputs
are independent. A deployment uses one fixed key, so repeated structures across
documents reuse the same PRF values and can make the realized false-positive
rate key-dependent even when contexts are deduplicated within each document.
Measure the false-positive rate on representative unwatermarked traffic with the
deployed key before relying on `is_watermarked` for decisions.

A minimal HTTP detector is available in
`examples/basic/online_serving/watermark_detection_server.py`:

```bash
python examples/basic/online_serving/watermark_detection_server.py \
  --tokenizer MODEL --key 42 --prf philox
```

```bash
curl http://localhost:8000/detect \
  -H 'Content-Type: application/json' \
  -d '{"text":"Text to inspect"}'
```

Scores and p-values expose information about the per-token watermark signal.
Repeated queries can use this information to construct text that imitates
watermarked output or to modify watermarked text so it is no longer detected.

## Limitations

- Watermarking is currently available only with Model Runner V2.
- Not all watermarking algorithms have native speculative-decoding support.
- Beam search expands candidates from model log probabilities and does not apply
  Gumbel-max watermarking.
- Models that replace the vLLM sampler with a custom sampler cannot use
  configured watermarking.
