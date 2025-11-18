# Tutorial 05: Sampler Implementation

## Learning Objectives

1. Understand the role of sampling in LLM token generation
2. Master different sampling strategies (greedy, temperature, top-k, top-p)
3. Explore beam search and its implementation
4. Learn about logit processors and penalties
5. Debug sampling issues and optimize performance

## Overview

The Sampler is the final component in the token generation pipeline. After the model produces logits for next token predictions, the sampler applies various strategies to select which token(s) to generate, balancing quality, diversity, and computational efficiency.

## Sampling Fundamentals

### The Sampling Problem

```
Model Output (Logits):
┌─────────────────────────────────────────┐
│ Token    │ Logit  │ Probability (after │
│          │        │    softmax)         │
├──────────┼────────┼────────────────────┤
│  "the"   │  8.5   │     45.2%          │
│  "a"     │  7.2   │     22.1%          │
│  "an"    │  6.8   │     17.3%          │
│  "his"   │  5.9   │      8.7%          │
│  "her"   │  5.1   │      4.2%          │
│  ...     │  ...   │      ...           │
└──────────┴────────┴────────────────────┘

Question: Which token should we generate?
```

Different sampling strategies answer this differently:

1. **Greedy**: Always pick "the" (highest probability)
2. **Temperature**: Sample from modified distribution
3. **Top-k**: Sample only from top k tokens
4. **Top-p (Nucleus)**: Sample from smallest set with cumulative prob ≥ p
5. **Beam Search**: Explore multiple hypotheses simultaneously

## Core Components

### 1. Sampler Class

**File**: `/vllm/v1/sample/sampler.py` (lines 20-100)

```python
class Sampler(nn.Module):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. Compute logprobs if requested
    2. Convert logits to float32
    3. Apply allowed token ids whitelist
    4. Apply bad words exclusion
    5. Apply logit processors (min tokens, logit bias)
    6. Apply penalties (repetition, frequency, presence)
    7. Sample next tokens
    8. Gather logprobs of sampled tokens
    9. Return SamplerOutput
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode)
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode

    def forward(
        self,
        logits: torch.Tensor,               # [batch_size, vocab_size]
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        """
        Sample next tokens from logits.

        Args:
            logits: Model output logits
            sampling_metadata: Sampling parameters for each request
            predict_bonus_token: Whether to predict extra token (speculative)
            logprobs_mode_override: Override default logprobs mode

        Returns:
            SamplerOutput with sampled token IDs and logprobs
        """

        # Step 1: Compute raw logprobs if needed
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        num_logprobs = sampling_metadata.max_num_logprobs

        if num_logprobs is not None:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                raw_logprobs = logits.clone()

        # Step 2: Use float32 for sampling (important for numerical stability)
        logits = logits.to(torch.float32)

        # Step 3-6: Apply logits processors
        logits = self.apply_logits_processors(
            logits,
            sampling_metadata,
            predict_bonus_token
        )

        # Step 7: Sample next token
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)

        # Step 8: Convert to long (int64)
        sampled = sampled.long()

        # Step 9: Gather logprobs for sampled tokens
        if num_logprobs is not None:
            logprobs = self.gather_logprobs(
                raw_logprobs if processed_logprobs is None else processed_logprobs,
                sampled,
                num_logprobs
            )
        else:
            logprobs = None

        # Return output
        return SamplerOutput(
            sampled_tokens=sampled,
            logprobs=logprobs
        )
```

### 2. Sampling Metadata

Contains per-request sampling parameters:

```python
@dataclass
class SamplingMetadata:
    """
    Metadata containing sampling parameters for each request.
    """

    # Per-request sampling parameters
    temperature: torch.Tensor          # [num_requests]
    top_p: torch.Tensor                # [num_requests]
    top_k: torch.Tensor                # [num_requests]
    min_p: torch.Tensor                # [num_requests]

    # Penalties
    repetition_penalty: torch.Tensor   # [num_requests]
    frequency_penalty: torch.Tensor    # [num_requests]
    presence_penalty: torch.Tensor     # [num_requests]

    # Request-specific
    prompt_tokens: list[list[int]]     # For penalty calculation
    output_tokens: list[list[int]]     # Already generated tokens

    # Logprobs
    max_num_logprobs: int | None = None

    # Constraints
    allowed_token_ids: list[list[int]] | None = None
    bad_words_ids: list[list[int]] | None = None
```

## Sampling Strategies

### 1. Greedy Sampling

Always select the highest probability token:

```python
def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling: select token with highest probability.

    Args:
        logits: [batch_size, vocab_size]

    Returns:
        sampled_tokens: [batch_size]
    """

    # Simply take argmax
    sampled = torch.argmax(logits, dim=-1)

    return sampled
```

**Characteristics**:
- Deterministic (same input → same output)
- Fast (single operation)
- Can produce repetitive text
- Good for factual tasks

**Example**:

```
Logits: [8.5, 7.2, 6.8, 5.9, 5.1]
        ↓
Greedy: Always picks index 0 (highest)

Output: "the the the the..." (repetitive!)
```

### 2. Temperature Sampling

Controls randomness by scaling logits before softmax:

```python
def temperature_sample(
    logits: torch.Tensor,
    temperature: float
) -> torch.Tensor:
    """
    Sample with temperature scaling.

    Args:
        logits: [batch_size, vocab_size]
        temperature: Scaling factor (0.0-2.0+)
            - Lower (0.1-0.5): More deterministic
            - 1.0: No change
            - Higher (1.5-2.0): More random

    Returns:
        sampled_tokens: [batch_size]
    """

    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sample from distribution
    sampled = torch.multinomial(probs, num_samples=1)

    return sampled.squeeze(-1)
```

**Effect of Temperature**:

```
Original Logits: [8.5, 7.2, 6.8, 5.9, 5.1]

Temperature = 0.1 (Low - Deterministic):
Probabilities: [99.8%, 0.1%, 0.05%, 0.03%, 0.02%]
→ Almost always picks first token

Temperature = 1.0 (Normal):
Probabilities: [45.2%, 22.1%, 17.3%, 8.7%, 4.2%]
→ Balanced sampling

Temperature = 2.0 (High - Random):
Probabilities: [28.3%, 23.1%, 20.5%, 15.2%, 10.1%]
→ More uniform, more diversity
```

### 3. Top-K Sampling

Sample only from the k most likely tokens:

```python
def top_k_sample(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Top-k sampling: sample only from k most likely tokens.

    Args:
        logits: [batch_size, vocab_size]
        k: Number of top tokens to consider
        temperature: Temperature for sampling

    Returns:
        sampled_tokens: [batch_size]
    """

    # Get top k logits and indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # Apply temperature
    top_k_logits = top_k_logits / temperature

    # Softmax over top k
    top_k_probs = F.softmax(top_k_logits, dim=-1)

    # Sample from top k
    sampled_idx = torch.multinomial(top_k_probs, num_samples=1)

    # Map back to original vocabulary
    sampled = torch.gather(top_k_indices, dim=-1, index=sampled_idx)

    return sampled.squeeze(-1)
```

**Example**:

```
Original Vocabulary (50,000 tokens):
  Token 123: prob = 45.2%
  Token 456: prob = 22.1%
  Token 789: prob = 17.3%
  Token 234: prob =  8.7%
  Token 567: prob =  4.2%
  ... (remaining 49,995 tokens)

Top-k (k=3):
  Only consider: [Token 123, Token 456, Token 789]
  Renormalize: [53.6%, 26.2%, 20.2%]
  Sample from these 3 only

Benefits:
  - Eliminates very unlikely tokens
  - Reduces nonsensical outputs
  - Maintains diversity among top tokens
```

### 4. Top-P (Nucleus) Sampling

Sample from smallest set of tokens whose cumulative probability ≥ p:

```python
def top_p_sample(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Top-p (nucleus) sampling: sample from smallest set with cumulative prob ≥ p.

    Args:
        logits: [batch_size, vocab_size]
        p: Cumulative probability threshold (0.0-1.0)
        temperature: Temperature for sampling

    Returns:
        sampled_tokens: [batch_size]
    """

    # Apply temperature
    logits = logits / temperature

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: first position where cumulative prob > p
    # Remove tokens with cumulative probability above p
    sorted_indices_to_remove = cumulative_probs > p

    # Keep at least one token
    sorted_indices_to_remove[..., 0] = False

    # Shift right to keep first token that exceeds p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Set logits to -inf for removed tokens
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # Sample from remaining tokens
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_sorted_idx = torch.multinomial(probs, num_samples=1)

    # Map back to original indices
    sampled = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)

    return sampled.squeeze(-1)
```

**Example**:

```
Sorted Probabilities:
  Token A: 40% (cumulative: 40%)
  Token B: 25% (cumulative: 65%)
  Token C: 15% (cumulative: 80%)  ← p=0.8 cutoff here
  Token D: 10% (cumulative: 90%)  ← Excluded
  Token E:  5% (cumulative: 95%)  ← Excluded
  ...

Top-p (p=0.8):
  Nucleus = {Token A, Token B, Token C}
  Renormalize: [50%, 31.25%, 18.75%]
  Sample from nucleus only

Benefits:
  - Adaptive cutoff (more tokens when distribution is flat)
  - Prevents sampling very unlikely tokens
  - Works well across different contexts
```

### 5. Combined Top-K + Top-P

Best practice: combine both for quality:

```python
def top_k_top_p_sample(
    logits: torch.Tensor,
    k: int,
    p: float,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Combined top-k and top-p sampling.

    Args:
        logits: [batch_size, vocab_size]
        k: Top-k threshold
        p: Top-p threshold
        temperature: Temperature scaling

    Returns:
        sampled_tokens: [batch_size]
    """

    # Apply temperature first
    logits = logits / temperature

    # Step 1: Apply top-k filtering
    if k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        # Create mask for non-top-k tokens
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
        logits = logits_filtered

    # Step 2: Apply top-p filtering
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above p
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    # Step 3: Sample
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)

    return sampled.squeeze(-1)
```

## Logit Processors and Penalties

### 1. Repetition Penalty

Reduces probability of already-generated tokens:

```python
def apply_repetition_penalty(
    logits: torch.Tensor,
    previous_tokens: list[int],
    penalty: float
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage repeating tokens.

    Args:
        logits: [vocab_size]
        previous_tokens: List of previously generated token IDs
        penalty: Penalty factor (> 1.0 discourages, < 1.0 encourages)

    Returns:
        Modified logits
    """

    for token_id in previous_tokens:
        # If logit is positive, divide by penalty
        # If logit is negative, multiply by penalty
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty

    return logits
```

**Example**:

```
Original logits for "the": 8.5
Previous tokens: ["the", "the", "the"]  (appeared 3 times)
Penalty: 1.2

After penalty: 8.5 / 1.2 = 7.08

Result: Less likely to generate "the" again
```

### 2. Frequency Penalty

Penalizes based on frequency of occurrence:

```python
def apply_frequency_penalty(
    logits: torch.Tensor,
    token_counts: dict[int, int],
    penalty: float
) -> torch.Tensor:
    """
    Apply frequency penalty based on token counts.

    Args:
        logits: [vocab_size]
        token_counts: {token_id: count}
        penalty: Penalty factor

    Returns:
        Modified logits
    """

    for token_id, count in token_counts.items():
        # Subtract penalty * count
        logits[token_id] -= penalty * count

    return logits
```

### 3. Presence Penalty

Binary penalty for any token that has appeared:

```python
def apply_presence_penalty(
    logits: torch.Tensor,
    present_tokens: set[int],
    penalty: float
) -> torch.Tensor:
    """
    Apply presence penalty for tokens that have appeared.

    Args:
        logits: [vocab_size]
        present_tokens: Set of token IDs that have appeared
        penalty: Penalty factor

    Returns:
        Modified logits
    """

    for token_id in present_tokens:
        logits[token_id] -= penalty

    return logits
```

### Combined Penalties

vLLM applies all penalties together:

```python
def apply_all_penalties(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    request_idx: int
) -> torch.Tensor:
    """Apply all penalties for a request"""

    # Get tokens for this request
    prompt_tokens = sampling_metadata.prompt_tokens[request_idx]
    output_tokens = sampling_metadata.output_tokens[request_idx]
    all_tokens = prompt_tokens + output_tokens

    # Repetition penalty
    if sampling_metadata.repetition_penalty[request_idx] != 1.0:
        logits = apply_repetition_penalty(
            logits,
            all_tokens,
            sampling_metadata.repetition_penalty[request_idx]
        )

    # Frequency penalty
    if sampling_metadata.frequency_penalty[request_idx] != 0.0:
        token_counts = {}
        for token in all_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        logits = apply_frequency_penalty(
            logits,
            token_counts,
            sampling_metadata.frequency_penalty[request_idx]
        )

    # Presence penalty
    if sampling_metadata.presence_penalty[request_idx] != 0.0:
        present_tokens = set(all_tokens)
        logits = apply_presence_penalty(
            logits,
            present_tokens,
            sampling_metadata.presence_penalty[request_idx]
        )

    return logits
```

## Beam Search

### Algorithm

Beam search maintains k hypotheses and explores them in parallel:

```python
class BeamSearchSampler:
    """
    Beam search sampler that maintains multiple hypotheses.
    """

    def __init__(self, beam_width: int, max_length: int):
        self.beam_width = beam_width
        self.max_length = max_length

    def search(
        self,
        model,
        initial_tokens: torch.Tensor,
    ) -> list[tuple[list[int], float]]:
        """
        Beam search decoding.

        Args:
            model: Language model
            initial_tokens: Starting tokens

        Returns:
            List of (token_sequence, score) tuples
        """

        # Initialize beams: [(tokens, score)]
        beams = [(initial_tokens.tolist(), 0.0)]

        for step in range(self.max_length):
            candidates = []

            # Expand each beam
            for tokens, score in beams:
                # Get logits for next token
                logits = model(torch.tensor(tokens))

                # Get top beam_width tokens
                top_logprobs, top_tokens = torch.topk(
                    F.log_softmax(logits[-1], dim=-1),
                    self.beam_width
                )

                # Create new candidates
                for logprob, token in zip(top_logprobs, top_tokens):
                    new_tokens = tokens + [token.item()]
                    new_score = score + logprob.item()
                    candidates.append((new_tokens, new_score))

            # Keep top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]

            # Check for termination
            all_finished = all(self._is_finished(tokens) for tokens, _ in beams)
            if all_finished:
                break

        return beams

    def _is_finished(self, tokens: list[int]) -> bool:
        """Check if sequence is finished (e.g., EOS token)"""
        return tokens[-1] == self.eos_token_id
```

**Beam Search Visualization**:

```
Step 0: Start
  Beam: ["The"]

Step 1: Expand
  Candidates:
    ["The", "cat"] (score: -1.2)
    ["The", "dog"] (score: -1.5)
    ["The", "car"] (score: -2.1)
    ["The", "man"] (score: -2.3)

  Keep top 2 (beam_width=2):
    ["The", "cat"] (score: -1.2)
    ["The", "dog"] (score: -1.5)

Step 2: Expand each beam
  From ["The", "cat"]:
    ["The", "cat", "is"] (score: -2.1)
    ["The", "cat", "sat"] (score: -2.3)

  From ["The", "dog"]:
    ["The", "dog", "is"] (score: -2.8)
    ["The", "dog", "ran"] (score: -2.9)

  Keep top 2:
    ["The", "cat", "is"] (score: -2.1)
    ["The", "cat", "sat"] (score: -2.3)

... continue until max_length or all beams end
```

## Hands-On Exercises

### Exercise 1: Compare Sampling Strategies

**Objective**: See how different strategies affect output

```python
def compare_sampling_strategies(prompt: str, model, tokenizer):
    """Compare outputs from different sampling strategies"""

    strategies = {
        "Greedy": {"temperature": 0.0},
        "Low Temp": {"temperature": 0.5, "top_p": 0.9},
        "Normal": {"temperature": 1.0, "top_p": 0.9},
        "High Temp": {"temperature": 1.5, "top_p": 0.9},
        "Top-k": {"temperature": 1.0, "top_k": 50},
        "Top-p": {"temperature": 1.0, "top_p": 0.9},
    }

    print(f"Prompt: {prompt}\n")

    for name, params in strategies.items():
        output = model.generate(
            prompt,
            max_tokens=50,
            **params
        )

        print(f"{name:12s}: {output}")
        print()
```

**Task**: Run with prompt "Once upon a time" and analyze outputs.

### Exercise 2: Implement Min-P Sampling

**Objective**: Implement a new sampling strategy

```python
def min_p_sample(
    logits: torch.Tensor,
    min_p: float,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Min-p sampling: remove tokens with prob < (min_p * max_prob).

    Args:
        logits: [batch_size, vocab_size]
        min_p: Minimum probability threshold (0.0-1.0)
        temperature: Temperature scaling

    Returns:
        sampled_tokens: [batch_size]
    """

    # TODO: Implement this!
    # Steps:
    # 1. Apply temperature
    # 2. Compute probabilities
    # 3. Find max probability
    # 4. Remove tokens with prob < (min_p * max_prob)
    # 5. Sample from remaining tokens

    # YOUR CODE HERE
    pass
```

**Task**: Implement and test min-p sampling.

### Exercise 3: Visualize Sampling Distributions

**Objective**: Understand how parameters affect distributions

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_sampling_effect(logits, temperatures=[0.5, 1.0, 2.0]):
    """Visualize how temperature affects distribution"""

    fig, axes = plt.subplots(1, len(temperatures), figsize=(15, 4))

    for ax, temp in zip(axes, temperatures):
        # Apply temperature
        scaled_logits = logits / temp
        probs = F.softmax(torch.tensor(scaled_logits), dim=-1).numpy()

        # Plot
        ax.bar(range(len(probs)), probs)
        ax.set_title(f'Temperature = {temp}')
        ax.set_xlabel('Token ID')
        ax.set_ylabel('Probability')

    plt.tight_layout()
    plt.show()

# Test with example logits
logits = [8.5, 7.2, 6.8, 5.9, 5.1, 3.2, 2.1, 1.5]
visualize_sampling_effect(logits)
```

**Task**: Run and observe the flattening effect of higher temperatures.

## Common Pitfalls and Solutions

### Pitfall 1: Temperature = 0 Division

**Problem**: Setting temperature to exactly 0 causes division by zero.

```python
# BAD
logits = logits / 0.0  # ❌ Division by zero!
```

**Solution**: Use greedy sampling for temperature ≈ 0:

```python
# GOOD
def safe_temperature_sample(logits, temperature):
    if temperature < 1e-5:
        # Effectively greedy
        return torch.argmax(logits, dim=-1)
    else:
        # Normal temperature sampling
        return temperature_sample(logits, temperature)
```

### Pitfall 2: Forgetting to Renormalize After Filtering

**Problem**: After filtering tokens, probabilities don't sum to 1.

```python
# BAD
top_k_probs = probs[top_k_indices]  # ❌ Doesn't sum to 1!
sampled = torch.multinomial(top_k_probs, 1)
```

**Solution**: Always renormalize after filtering:

```python
# GOOD
top_k_logits = logits[top_k_indices]
top_k_probs = F.softmax(top_k_logits, dim=-1)  # ✓ Sums to 1
sampled = torch.multinomial(top_k_probs, 1)
```

### Pitfall 3: Incorrect Penalty Application

**Problem**: Applying penalties incorrectly can cause bias.

```python
# BAD: Always subtracting penalty
for token in previous_tokens:
    logits[token] -= penalty  # ❌ Wrong for negative logits!
```

**Solution**: Apply penalties correctly based on sign:

```python
# GOOD: Correct penalty application
for token in previous_tokens:
    if logits[token] > 0:
        logits[token] /= penalty
    else:
        logits[token] *= penalty
```

## Performance Optimization

### 1. Batch Sampling

Sample multiple requests together:

```python
def batch_sample(
    logits: torch.Tensor,  # [batch_size, vocab_size]
    temperatures: torch.Tensor,  # [batch_size]
    top_p: torch.Tensor,  # [batch_size]
) -> torch.Tensor:
    """Efficiently sample a batch of requests"""

    batch_size, vocab_size = logits.shape

    # Apply temperature (broadcasting)
    logits = logits / temperatures.unsqueeze(1)

    # Vectorized top-p filtering
    # ... (implementation details)

    # Batch multinomial sampling
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)

    return sampled.squeeze(-1)
```

### 2. Fused Kernel for Top-K + Top-P

```python
# Instead of separate top-k and top-p operations,
# use fused kernel for better performance

@torch.jit.script
def fused_top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Fused top-k and top-p filtering"""

    # Single pass through logits
    # Apply both filters simultaneously
    # ... (optimized implementation)

    return filtered_logits
```

### 3. Cache-Friendly Penalty Application

```python
def optimized_penalty_application(
    logits: torch.Tensor,
    penalties: torch.Tensor,  # Pre-computed penalties per token
) -> torch.Tensor:
    """Apply pre-computed penalties efficiently"""

    # Single operation instead of loop
    logits = logits - penalties

    return logits
```

## Advanced Topics

### Constrained Decoding

Force output to match constraints (e.g., JSON schema):

```python
class ConstrainedSampler:
    """Sampler with constraints on valid tokens"""

    def __init__(self, constraint_checker):
        self.constraint_checker = constraint_checker

    def sample(self, logits, current_tokens):
        # Get valid next tokens based on constraints
        valid_tokens = self.constraint_checker.get_valid_tokens(current_tokens)

        # Mask invalid tokens
        mask = torch.ones_like(logits) * float('-inf')
        mask[valid_tokens] = 0
        logits = logits + mask

        # Sample from valid tokens only
        return torch.argmax(logits)
```

### Speculative Sampling

Sample draft tokens quickly, then verify:

```python
def speculative_sample(
    draft_model,
    target_model,
    prompt_tokens,
    k: int = 4  # Number of speculative tokens
):
    """
    Speculative sampling for faster generation.

    1. Draft model generates k tokens quickly
    2. Target model verifies in parallel
    3. Accept/reject based on probabilities
    """

    # Draft k tokens with small model
    draft_tokens = []
    for _ in range(k):
        draft_logits = draft_model(prompt_tokens + draft_tokens)
        draft_token = sample(draft_logits)
        draft_tokens.append(draft_token)

    # Verify with target model (parallel)
    target_logits = target_model(prompt_tokens + draft_tokens)

    # Accept/reject each draft token
    accepted_tokens = []
    for i, draft_token in enumerate(draft_tokens):
        target_prob = F.softmax(target_logits[i], dim=-1)[draft_token]
        draft_prob = ...  # From draft model

        if random.random() < target_prob / draft_prob:
            accepted_tokens.append(draft_token)
        else:
            # Rejection, sample new token from adjusted distribution
            break

    return accepted_tokens
```

## References

### Source Code Files

- **Sampler**: `/vllm/v1/sample/sampler.py`
- **Top-K/Top-P**: `/vllm/v1/sample/ops/topk_topp_sampler.py`
- **Penalties**: `/vllm/v1/sample/ops/penalties.py`
- **Bad Words**: `/vllm/v1/sample/ops/bad_words.py`
- **Sampling Metadata**: `/vllm/v1/sample/metadata.py`

### Key Papers

- "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) - Nucleus sampling
- "Hierarchical Neural Story Generation" (Fan et al., 2018) - Sampling strategies
- "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)

### Configuration

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 16
```

## Summary

In this tutorial, you learned:

- Different sampling strategies and their trade-offs
- How to implement greedy, temperature, top-k, and top-p sampling
- Logit processors and penalties for controlling generation
- Beam search for exploring multiple hypotheses
- Performance optimization techniques
- Advanced topics like constrained and speculative sampling

Sampling is the final step that converts model predictions into actual text. Understanding sampling strategies helps you control output quality, diversity, and coherence.

## Next Steps

- **Tutorial 06**: KV Cache Management - Complete cache lifecycle
- **Tutorial 07**: Request Batching Strategies - Optimize throughput
- **Module 6**: Advanced Generation Techniques
