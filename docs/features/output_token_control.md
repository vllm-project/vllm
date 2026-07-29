# Output Token Control

vLLM extends the standard OpenAI sampling API with several parameters that give
fine-grained control over *which* tokens the model may generate and *when*
generation stops. These parameters are set on `SamplingParams` for offline use
or passed via `extra_body` in the OpenAI-compatible API.

## Bad Words (`bad_words`)

Prevents specific words or phrases from appearing in the model output. Only the
last token of a matching token sequence is suppressed: if the model has already
generated all but the final token of a forbidden phrase, that final token is
masked out.

### Offline (Python API)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

sampling_params = SamplingParams(
    temperature=0.8,
    bad_words=["violence", "hate speech"],
)

outputs = llm.generate("Tell me a story.", sampling_params)
print(outputs[0].outputs[0].text)
```

### Online (OpenAI-compatible API)

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="Tell me a story.",
    extra_body={"bad_words": ["violence", "hate speech"]},
)
print(response.choices[0].text)
```

## Allowed Token IDs (`allowed_token_ids`)

Restricts generation to a whitelist of token IDs. All other tokens are masked to
`-inf` before sampling. Useful when the output must be drawn from a fixed
vocabulary, such as classification labels or constrained structured outputs.

### Offline (Python API)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# Tokenize the labels you want to allow
tokenizer = llm.get_tokenizer()
yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id = tokenizer.encode("No", add_special_tokens=False)[0]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1,
    allowed_token_ids=[yes_id, no_id],
)

outputs = llm.generate("Is the sky blue? Answer Yes or No.", sampling_params)
print(outputs[0].outputs[0].text)
```

### Online (OpenAI-compatible API)

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="Is the sky blue? Answer Yes or No.",
    max_tokens=1,
    extra_body={"allowed_token_ids": [9642, 2360]},  # token IDs for "Yes" / "No"
)
print(response.choices[0].text)
```

## Logprob Token IDs (`logprob_token_ids`)

Returns log-probabilities for a specific set of token IDs at each generation
step, without having to request logprobs for the full vocabulary
(`logprobs=-1`). This is more efficient when you only need probabilities for a
small, known set of tokens — for example, scoring classification labels or
monitoring specific output tokens.

The logprobs for `logprob_token_ids` are returned alongside the sampled token's
logprob in `RequestOutput.outputs[i].logprobs`.

!!! note
    `logprob_token_ids` is capped at `VLLM_MAX_LOGPROB_TOKEN_IDS` token IDs
    (default 100). Exceeding this limit raises a validation error.

### Offline (Python API)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

tokenizer = llm.get_tokenizer()
label_ids = [
    tokenizer.encode(label, add_special_tokens=False)[0]
    for label in ["positive", "negative", "neutral"]
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1,
    logprob_token_ids=label_ids,
)

outputs = llm.generate("Classify the sentiment: 'I love this!'", sampling_params)
logprobs = outputs[0].outputs[0].logprobs[0]
for token_id, lp in logprobs.items():
    print(f"token {token_id}: logprob={lp.logprob:.4f}")
```

### Online (OpenAI-compatible API)

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="Classify the sentiment: 'I love this!'",
    max_tokens=1,
    extra_body={"logprob_token_ids": [6928, 8225, 21277]},  # label token IDs
)
print(response.choices[0].logprobs)
```

## Repetition Detection (`repetition_detection`)

Detects repetitive N-gram patterns in the generated output and ends generation
early when such a pattern is found. Language models can sometimes produce
repetitive token sequences (e.g. `abcdabcdabcd…` or repeated emoji) and only
stop when they reach the maximum output length. Repetition detection terminates
these sequences early, saving both time and tokens.

Configure it with `RepetitionDetectionParams`:

| Field | Type | Description |
|-------|------|-------------|
| `max_pattern_size` | `int` | Maximum N-gram length to detect. Set to `0` to disable. Must be used with `min_count`. |
| `min_pattern_size` | `int` | Minimum N-gram length to check. Defaults to `1` when set to `0`. Must be ≤ `max_pattern_size`. |
| `min_count` | `int` | Number of consecutive repetitions required to trigger early stopping. Must be ≥ 2. |

### Offline (Python API)

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import RepetitionDetectionParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=512,
    repetition_detection=RepetitionDetectionParams(
        max_pattern_size=20,  # detect N-grams up to length 20
        min_pattern_size=1,   # check all lengths from 1 to 20
        min_count=3,          # stop if a pattern repeats 3+ times
    ),
)

outputs = llm.generate("Continue this sequence:", sampling_params)
print(outputs[0].outputs[0].text)
```

### Online (OpenAI-compatible API)

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="Continue this sequence:",
    max_tokens=512,
    extra_body={
        "repetition_detection": {
            "max_pattern_size": 20,
            "min_pattern_size": 1,
            "min_count": 3,
        }
    },
)
print(response.choices[0].text)
```
