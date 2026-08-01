# Output Token Control

vLLM supports several sampling parameters, beyond those in the OpenAI API, that give fine-grained control over which tokens can be generated and when generation terminates:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `bad_words` | `list[str]` | Prevent specific words or phrases from being generated |
| `allowed_token_ids` | `list[int]` | Restrict generation to a whitelist of token IDs |
| `logprob_token_ids` | `list[int]` | Return log probabilities for a specific set of token IDs |
| `repetition_detection` | [RepetitionDetectionParams][vllm.sampling_params.RepetitionDetectionParams] | Detect repetitive N-gram output and terminate generation early |

All four parameters are available in offline inference via [SamplingParams][vllm.SamplingParams], and in the OpenAI-compatible server via `extra_body` on both the Chat Completions and Completions endpoints.

## Preventing words with `bad_words`

`bad_words` takes a list of words or phrases that are not allowed to appear in the generated output. Matching happens at the token level: the last token of a banned sequence is blocked whenever generating it would complete that sequence, so multi-token phrases and different tokenizations (e.g. with and without a leading space) are handled.

### Offline

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")

sampling_params = SamplingParams(bad_words=["purple", "as an AI"])
outputs = llm.generate("My favorite color is", sampling_params)
print(outputs[0].outputs[0].text)
```

### Online

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = client.models.list().data[0].id

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is your favorite color?"}],
    extra_body={"bad_words": ["purple", "as an AI"]},
)
```

!!! note
    Empty strings are not allowed in `bad_words` and will be rejected with a validation error.

## Restricting output with `allowed_token_ids`

`allowed_token_ids` restricts generation to a whitelist of token IDs: the engine constructs a logits processor that retains scores only for the listed tokens. This is useful for forcing the model to answer from a fixed set of options (e.g. classification labels or multiple-choice letters).

### Offline

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")
tokenizer = llm.get_tokenizer()

# Only allow the tokens "Yes" and "No"
allowed = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["Yes", "No"]]

sampling_params = SamplingParams(allowed_token_ids=allowed, max_tokens=1)
outputs = llm.generate("Is the sky blue? Answer Yes or No:", sampling_params)
print(outputs[0].outputs[0].text)
```

### Online

```python
# client, model as in the example above; allowed as in the offline example
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Is the sky blue? Answer Yes or No."}],
    max_tokens=1,
    extra_body={"allowed_token_ids": allowed},
)
```

!!! note
    `allowed_token_ids` must be non-empty and every ID must be within the model's vocabulary; out-of-vocabulary IDs are rejected with a validation error. For constraining output to a schema or grammar rather than a token whitelist, see [structured outputs](structured_outputs.md).

## Targeted log probabilities with `logprob_token_ids`

`logprob_token_ids` returns log probabilities for exactly the specified token IDs at each sampled position, in addition to the sampled token itself. This is much more efficient than requesting the full vocabulary with `logprobs=-1` when you only care about a few tokens — a common pattern in scoring and classification tasks where you want to compare the probabilities of specific label tokens.

### Offline

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")
tokenizer = llm.get_tokenizer()

label_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["Yes", "No"]]

sampling_params = SamplingParams(logprob_token_ids=label_ids, max_tokens=1)
outputs = llm.generate("Is the sky blue? Answer Yes or No:", sampling_params)
print(outputs[0].outputs[0].logprobs)
```

### Online

```python
# client, model as in the example above; label_ids as in the offline example
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Is the sky blue? Answer Yes or No."}],
    max_tokens=1,
    logprobs=True,
    extra_body={"logprob_token_ids": label_ids},
)
print(response.choices[0].logprobs)
```

!!! note
    - In the Chat Completions API, `logprobs` must be set to `true` when using `logprob_token_ids`; in the Completions API, `logprobs` must be set to a value.
    - The list length is capped at 128 token IDs per request.
    - `logprob_token_ids` is not supported together with beam search.
    - In offline `SamplingParams`, when a numeric `logprobs` value is also set, it must equal `len(logprob_token_ids)`.

## Early termination with `repetition_detection`

LLMs can sometimes fall into generating repetitive, unhelpful token patterns (e.g. `abcdabcdabcd...`), stopping only when they hit the maximum output length. `repetition_detection` detects repeating N-gram patterns in the output tokens and terminates generation early, saving time and tokens.

It takes a [RepetitionDetectionParams][vllm.sampling_params.RepetitionDetectionParams] with the following fields:

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `max_pattern_size` | `int` | `0` | Maximum size of N-gram pattern to detect. Set to `0` to disable. Must be used together with `min_count`. |
| `min_pattern_size` | `int` | `0` | Minimum N-gram pattern size to check. `0` defaults to `1`. Must be `<= max_pattern_size`. |
| `min_count` | `int` | `0` | Number of times a pattern must repeat to trigger detection. Must be `>= 2`. |

### Offline

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import RepetitionDetectionParams

llm = LLM(model="Qwen/Qwen3-0.6B")

sampling_params = SamplingParams(
    max_tokens=1024,
    repetition_detection=RepetitionDetectionParams(
        max_pattern_size=8,
        min_pattern_size=1,
        min_count=4,
    ),
)
outputs = llm.generate("Repeat the word 'token' forever:", sampling_params)
```

### Online

```python
# client, model as in the example above
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Repeat the word 'token' forever."}],
    extra_body={
        "repetition_detection": {
            "max_pattern_size": 8,
            "min_pattern_size": 1,
            "min_count": 4,
        }
    },
)
```

!!! note
    Detection operates on token IDs, not text. Choose `max_pattern_size` to cover the token length of the patterns you want to catch, and `min_count` high enough that legitimate repetition (e.g. lists or refrains) is not cut off.
