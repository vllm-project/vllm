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

Watermarking is disabled when `--watermark-config` is omitted. Gumbel is the
default algorithm within an enabled `WatermarkConfig`.

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

Gumbel-max derives a deterministic pseudorandom value from the key, prior
generated-token context, and every candidate token, then uses the resulting
Gumbel noise for categorical sampling. See
[Aaronson's original presentation](https://simons.berkeley.edu/sites/default/files/2024-10/LLM24-2%20Slides%20-%20Scott%20Aaronson.pdf).

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

vLLM implements two PRFs:

- `philox` is the default. It is based on the counter-based Philox4x32-10
  generator from the [Random123 paper](https://doi.org/10.1145/2063384.2063405).
  It is parallel, vectorizes on accelerators, and avoids CPU transfers, but is
  not a cryptographic PRF. vLLM versions its input mapping and provides
  compatibility vectors so generation and detection remain interoperable.
- `hmac_sha256` is a cryptographically secure reference implementation based on
  [HMAC](https://doi.org/10.1007/3-540-68697-5_1) and standardized by
  [RFC 2104](https://www.rfc-editor.org/rfc/rfc2104). It provides a conservative,
  portable reference but copies inputs to the CPU and is not suitable for
  performance-sensitive generation.

To override the default, set `prf` in `--watermark-config` and use the same PRF
for detection.

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
Gumbel detection scores repeated contexts once by default so identical PRF
random vectors are not treated as independent evidence. This can be disabled
with `deduplicate_contexts=False`.

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

## Limitations

- Watermarking is currently available only with Model Runner V2.
- Gumbel-max does not support speculative decoding.
- Greedy requests (`temperature=0`) are not watermarked.
