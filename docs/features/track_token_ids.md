# Tracked Token Logprobs

vLLM supports efficient tracking of log probabilities (logprobs) for specific tokens without requiring full vocabulary logprobs. This feature is useful for classification tasks, constrained generation, and any use case where you need accurate probabilities for a small set of predefined tokens.

## Motivation

When working with LLMs, you may need logprobs for specific tokens (e.g., class labels, binary choices, special tokens). Previously, users had to choose between two suboptimal approaches:

| Approach | Description | Problems |
|----------|-------------|----------|
| **Top-k logprobs** (`logprobs=k`) | Hope target tokens appear in top-k | ❌ Unreliable: target tokens may not be in top-k |
| **Full vocabulary** (`logprobs=-1`) | Retrieve all ~150k logprobs | ❌ Memory overhead (~1.2 MB per token)<br>❌ Large GPU→CPU transfer<br>❌ Wasteful: discard 99.99% of data |

The `track_token_ids` parameter solves this by allowing you to specify exactly which tokens to track, achieving:

- **99.99% memory reduction** compared to full vocabulary retrieval
- **Reliable results**: tracked tokens are always included regardless of their rank
- **Minimal overhead**: just indexing specific columns from the logprobs tensor

## Use Cases

### Classification Tasks

Track probabilities for class labels to get reliable classification confidence:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
tokenizer = llm.get_tokenizer()

# Get token IDs for class labels
classes = ["Technology", "Politics", "Sports", "Art"]
track_ids = []
for label in classes:
    ids = tokenizer.encode(label, add_special_tokens=False)
    track_ids.extend(ids)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=10,
    track_token_ids=track_ids,
)

outputs = llm.generate(
    ["Classify this article: 'The new GPU architecture shows 2x performance...'"],
    sampling_params,
)

# Access tracked logprobs
tracked = outputs[0].outputs[0].tracked_logprobs
for token_id, logprobs_over_time in tracked.items():
    token_str = tokenizer.decode([token_id])
    print(f"{token_str}: {logprobs_over_time}")
```

### Binary Decision Making

Perfect for yes/no, true/false, or binary classification scenarios:

```python
# Medical diagnosis, content moderation, fact-checking, etc.
yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
no_ids = tokenizer.encode("No", add_special_tokens=False)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=50,
    track_token_ids=yes_ids + no_ids,
)

outputs = llm.generate(["Is this content safe? Content: ..."], sampling_params)
tracked = outputs[0].outputs[0].tracked_logprobs
# Analyze confidence between Yes/No across generation steps
```

### AI-Generated Content Detection

Monitor probabilities for detection-related tokens throughout generation:

```python
# Track "real" vs "ai-generated" probabilities
real_ids = tokenizer.encode("real", add_special_tokens=False)
ai_ids = tokenizer.encode("ai-generated", add_special_tokens=False)
synthetic_ids = tokenizer.encode("synthetic", add_special_tokens=False)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=100,
    track_token_ids=real_ids + ai_ids + synthetic_ids,
)

outputs = llm.generate(["Is this image real or AI-generated?"], sampling_params)

# Analyze confidence evolution across generation
tracked = outputs[0].outputs[0].tracked_logprobs
for token_id, logprobs_over_time in tracked.items():
    token_str = tokenizer.decode([token_id])
    print(f"{token_str}: step-by-step logprobs = {logprobs_over_time}")
```

### Research and Analysis

Track specific vocabulary items for hypothesis testing or model behavior analysis:

```python
# Study model confidence for specific tokens across long sequences
interesting_tokens = ["however", "therefore", "because", "but"]
track_ids = []
for token in interesting_tokens:
    track_ids.extend(tokenizer.encode(token, add_special_tokens=False))

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=500,
    track_token_ids=track_ids,
)
```

## API Reference

### SamplingParams

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    # ... other parameters ...
    track_token_ids=[1234, 5678, 9012],  # List of token IDs to track
)
```

**Parameter:**

- `track_token_ids` (`list[int] | None`): List of token IDs to track logprobs for at every generation step. These tokens will have their logprobs recorded even if they don't appear in the top-k. Default is `None` (disabled).

### Output Format

Tracked logprobs are available in the `CompletionOutput`:

```python
@dataclass
class CompletionOutput:
    # ... existing fields ...
    tracked_logprobs: dict[int, list[float]] | None
    """Logprobs for tracked tokens across all generation steps.
    
    Format: {token_id: [logprob_step0, logprob_step1, ...]}
    Only present if track_token_ids was specified in SamplingParams.
    """
```

**Example output structure:**

```python
# For track_token_ids=[100, 200, 300] with 5 generation steps:
tracked_logprobs = {
    100: [-2.3, -1.8, -2.1, -3.0, -2.5],  # logprobs at each step
    200: [-5.1, -4.9, -5.3, -4.7, -5.0],
    300: [-8.2, -7.9, -8.5, -8.1, -8.3],
}
```

## Combining with Regular Logprobs

You can use `track_token_ids` alongside regular top-k logprobs:

```python
sampling_params = SamplingParams(
    logprobs=5,                    # Get top-5 logprobs as usual
    track_token_ids=[100, 200],    # Also track these specific tokens
)

outputs = llm.generate(prompts, sampling_params)

# Access both
for output in outputs:
    completion = output.outputs[0]
    
    # Regular top-k logprobs (list of dicts, one per step)
    top_k = completion.logprobs
    
    # Tracked token logprobs (dict of lists)
    tracked = completion.tracked_logprobs
```

## Edge Cases

- **Empty track list** (`track_token_ids=[]`): Returns an empty dict `{}`
