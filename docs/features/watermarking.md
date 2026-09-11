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

`context_width` controls how many prior output tokens seed each watermark
decision and defaults to 4. Larger values make the watermark less robust to
edits because an insertion, deletion, or substitution changes more subsequent
contexts. Values above 16 are allowed but emit a warning.

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
vectors are not treated as independent evidence. When the first occurrence was
watermarked, this does not reduce the watermarking signal unless a user tampers
with the output and removes that occurrence while leaving an unwatermarked
repetition.

`deduplicate_contexts` controls the generation history scope:

- `"none"` disables context deduplication.
- `"single_turn"` enables context deduplication within the generation of a
  single request. This allows for single-turn non-distortion.
- `"all"` enables context deduplication across the entire request context. This
  allows for non-distortion in a multi-turn setting. For large contexts this may
  impact inference speed and may lead to substantially fewer generated tokens
  being watermarked. It is only recommended when single-sequence non-distortion
  over multiple turns is strictly required.

`deduplicate_contexts_max_history` limits deduplication to the most recent
positions and defaults to 8,192. A smaller value reduces scanning cost but only
provides the guarantee within that window. Set it to `null` to remove the limit
and scan to the beginning of the current generation for `"single_turn"`, or the
beginning of the supplied request context for `"all"`. An unbounded scan can
increase inference latency as the sequence grows.

For example, this checks prompt and completion history within the default
8,192-position window:

```bash
vllm serve MODEL \
  --watermark-config \
  '{"algorithm":"gumbel","key":42,"deduplicate_contexts":"all"}'
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

The tokenizer, algorithm, PRF, key, and context width must match generation.
Gumbel-max detection scores repeated contexts once by default so identical PRF
random vectors are not treated as independent evidence. Keep
`deduplicate_contexts=True` unless the detector's calibration has been adjusted
for correlated scores.

The detector must also know the server's `deduplicate_contexts` scope. Text
generated with `"all"` keys its first `context_width` tokens on the prompt and
leaves every context that already occurred in the prompt unwatermarked, so the
detector needs the prompt to recompute the same contexts and to skip the same
positions. Pass it as `context_prefix`; its tokens are never scored:

```python
detector = GumbelWatermarkDetector(key=42, prf="philox", history_scope="all")
result = detector.detect(completion_ids, context_prefix=prompt_ids)
```

With the default `"single_turn"` scope, and with `"none"`, the completion alone
is the correct input and a `context_prefix` is rejected. For `"all"` the prefix
may be omitted when the prompt is unavailable: the p-value stays calibrated,
but positions that were never watermarked are scored and detection is weaker,
so the detector logs a warning.

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
- Gumbel-max cannot be configured with speculative decoding.
- Beam search expands candidates from model log probabilities and does not apply
  Gumbel-max watermarking.
- Models that replace the vLLM sampler with a custom sampler cannot use
  configured watermarking.
