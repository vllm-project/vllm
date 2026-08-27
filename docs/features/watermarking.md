# Text watermarking

Text watermarking embeds a statistical signal in generated token choices. A
detector that knows the generation parameters can test for this signal without
access to the model weights.

The design and scope are discussed in
[RFC #53916](https://github.com/vllm-project/vllm/issues/53916).

## Configuration

Enable watermarking by configuring an algorithm, secret key, and pseudorandom
function (PRF) at engine startup:

```bash
vllm serve MODEL \
  --watermark-config '{"algorithm":"gumbel","key":42,"prf":"philox"}'
```

Watermarking is disabled when `--watermark-config` is omitted. Gumbel is the
default algorithm and Philox is the default PRF within an enabled
`WatermarkConfig`.

## Architecture

`WatermarkConfig` selects an algorithm and PRF. Model Runner V2 constructs the
corresponding `Watermarker`, and `GPUWatermarkSampler` invokes it for the final
stochastic token selection after temperature, min-p, top-k, and top-p are
applied. A watermarker can either select a token directly or transform logits
and delegate to vLLM's random sampler.

Detection is separate from generation. `WatermarkDetector` consumes token IDs,
so callers remain responsible for using the tokenizer and watermark profile
that match generation.

## Algorithms

### Gumbel-max

Gumbel-max derives a deterministic pseudorandom value from the key, prior
generated-token context, and every candidate token, then uses the resulting
Gumbel noise for categorical sampling. See the
[formal treatment](https://arxiv.org/abs/2307.15593) and
[Aaronson's original presentation](https://simons.berkeley.edu/sites/default/files/2024-10/LLM24-2%20Slides%20-%20Scott%20Aaronson.pdf).

### SynthID-Text

[SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) is planned but
not currently implemented. Its sampling and speculative-decoding algorithms
are described further in the
[supplementary material](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-08025-4/MediaObjects/41586_2024_8025_MOESM1_ESM.pdf).

## Pseudorandom functions

`philox` is the default. vLLM defines a versioned Philox4x32-10 input mapping
and compatibility vectors so generated text remains detectable across future
releases. See the [Philox paper](https://doi.org/10.1145/2063384.2063405).

`hmac_sha256` is a cryptographically secure reference implementation following
[RFC 2104](https://www.rfc-editor.org/rfc/rfc2104) and the
[original HMAC paper](https://doi.org/10.1007/3-540-68697-5_1). It copies inputs
to the CPU and is not suitable for performance-sensitive generation.

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
- The example detector is a demonstration, not a production key-management or
  multi-tokenizer search service.
