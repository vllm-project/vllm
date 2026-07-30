# DFlash / Domino

DFlash is a parallel drafter for speculative decoding: it produces all K draft
tokens in a single forward pass of the draft model, avoiding the sequential
overhead of autoregressive drafters.

[Domino](https://arxiv.org/abs/2605.29707) extends DFlash with a lightweight
causal correction head (a GRU encoder + low-rank MLP) that refines the parallel
base logits using causal state from previously drafted tokens. The correction
operates in logit space, so no additional forward passes through the draft model
or LM head are required.

## Usage

Domino is configured as a `projector_type` sub-mode of DFlash. Use a
Domino-trained checkpoint with `method="dflash"`:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "dflash",
        "model": "your-username/Qwen3-8B-Domino-b16",
        "num_speculative_tokens": 16,
    },
)
```

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative-config '{
        "method": "dflash",
        "model": "your-username/Qwen3-8B-Domino-b16",
        "num_speculative_tokens": 16
    }'
```

When the checkpoint's `dflash_config.projector_type` is `"domino"`, vLLM
automatically loads the Domino correction head weights and uses them during
draft generation.

## Training Domino draft models

Domino draft models are trained using the
[vllm-project/speculators](https://github.com/vllm-project/speculators) library.
See the [speculators guide](speculators.md) for details.

## Pre-trained models

- See the [vllm-project/speculators](https://github.com/vllm-project/speculators)
  repository for available Domino checkpoints.
- Public DFlash checkpoints (without Domino head) are available on Hugging Face,
  e.g. `z-lab/Qwen3-8B-DFlash-b16`.
