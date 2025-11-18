# Lab 03: Custom Sampling Strategies

## Overview
Learn to implement and experiment with different sampling strategies for text generation. This lab explores temperature, top-k, top-p (nucleus), beam search, and custom sampling logic to control generation quality and diversity.

## Learning Objectives
1. Understand different sampling strategies and their effects
2. Implement temperature scaling and its impact on randomness
3. Configure top-k and top-p (nucleus) sampling
4. Experiment with beam search for deterministic generation
5. Create custom sampling strategies with logit processors

## Estimated Time
1.5-2 hours

## Prerequisites
- Completion of Lab 01 (Basic Inference)
- Understanding of probability distributions
- Basic knowledge of language model generation

## Lab Structure
```
lab03_custom_sampling/
├── README.md           # This file
├── starter.py          # Starter code with TODOs
├── solution.py         # Complete solution
├── test_lab.py         # pytest tests
└── requirements.txt    # Dependencies
```

## Instructions

### Step 1: Setup Environment
```bash
pip install -r requirements.txt
```

### Step 2: Implement Sampling Strategies

#### TODO 1: Greedy Decoding
Implement deterministic greedy decoding (temperature=0).

#### TODO 2: Temperature Sampling
Experiment with different temperature values (0.1, 0.8, 1.5, 2.0).

#### TODO 3: Top-K Sampling
Implement top-k sampling with different k values.

#### TODO 4: Top-P (Nucleus) Sampling
Configure nucleus sampling with various p thresholds.

#### TODO 5: Beam Search
Implement beam search for higher quality outputs.

### Step 3: Run Experiments
```bash
python starter.py
```

### Step 4: Verify with Tests
```bash
pytest test_lab.py -v
```

## Expected Output

```
=== vLLM Custom Sampling Lab ===

Prompt: "Once upon a time"

[Greedy Decoding]
Temperature: 0.0
Output: Once upon a time, there was a young girl...

[Low Temperature]
Temperature: 0.3
Output: Once upon a time, there was a beautiful princess...

[High Temperature]
Temperature: 1.5
Output: Once upon a time, dragons flew across magical skies...

[Top-K Sampling (k=10)]
Output: Once upon a time, in a faraway kingdom...

[Nucleus Sampling (p=0.9)]
Output: Once upon a time, adventure called to brave souls...

[Beam Search (n=4)]
Output: Once upon a time, there lived a wise king who ruled...
```

## Key Concepts

### Temperature
- Controls randomness in sampling
- Low (0.1-0.5): More focused, deterministic
- Medium (0.7-1.0): Balanced creativity
- High (1.5-2.0): More random, creative

### Top-K Sampling
- Samples from top K most probable tokens
- Lower K: More conservative
- Higher K: More diverse

### Top-P (Nucleus) Sampling
- Samples from smallest set of tokens with cumulative probability >= p
- Dynamically adjusts vocabulary size
- p=0.9 is typical for balanced output

### Beam Search
- Maintains N best sequences
- More coherent but less diverse
- Good for tasks requiring accuracy

## Troubleshooting

### Issue: All outputs look the same
**Solution**: Increase temperature or top-p value.

### Issue: Gibberish output
**Solution**: Reduce temperature or increase top-k/top-p constraints.

### Issue: Beam search too slow
**Solution**: Reduce beam width or max tokens.

## Going Further

1. Implement repetition penalty
2. Add presence/frequency penalties
3. Create custom logit processors
4. Compare sampling strategies quantitatively
5. Implement constrained generation

## References
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [vLLM Sampling Parameters](https://docs.vllm.ai/)
