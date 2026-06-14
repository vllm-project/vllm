# 📄 Technical Specification  
## vLLM Integration for BitNet b1.58 (BF16 Correctness Mode)

---

## 1. Objective

Enable **correct (not optimized) inference** of  
`microsoft/bitnet-b1.58-2B-4T-bf16` in vLLM.

### Success Criteria
- Model loads in vLLM without `trust_remote_code`
- Generates text correctly
- Output is **numerically close** to Hugging Face reference
- Supports:
  - standard prompts
  - chat-template prompts

### Non-Goals (Phase 1)
- ❌ No ternary (-1,0,1) kernel support
- ❌ No packed 1.58-bit weight support
- ❌ No custom CUDA kernels
- ❌ No performance optimization

---

## 2. High-Level Architecture

We implement BitNet as a **new native vLLM architecture**, not a wrapper.

### Key Components

```
BitNetForCausalLM
 └── BitNetModel
      ├── Embedding
      ├── N × BitNetDecoderLayer
      │     ├── Attention (RoPE + GQA)
      │     ├── MLP (ReLU²)
      │     ├── SubLN (custom norm)
      │     └── BitLinear (BF16 version)
      └── Final Norm
 └── LM Head (tied embeddings)
```

---

## 3. Model Characteristics (from HF config)

| Attribute | Value |
|----------|------|
| Hidden size | 2560 |
| Layers | 30 |
| Heads | 20 |
| KV heads | 5 (GQA) |
| FFN size | 6912 |
| Activation | ReLU² |
| Norm | SubLN |
| Bias | None |
| RoPE theta | 500000 |
| Context length | 4096 |
| Dtype | BF16 |
| Embedding | tied |

---

## 4. Development Strategy

### Step 1 — Build as vLLM Plugin (Out-of-tree)
Use:
```python
ModelRegistry.register_model(...)
```

### Step 2 — Upstream later into vLLM core

---

## 5. Code Structure

### 5.1 Directory Layout

```
bitnet_vllm/
├── __init__.py
├── modeling_bitnet_vllm.py
├── layers/
│   ├── bitlinear.py
│   ├── attention.py
│   ├── mlp.py
│   ├── norm.py
│   └── activations.py
├── weight_loader.py

scripts/
tests/
```

---

## 6. Core Components

---

## 6.1 BitLinear (BF16 Version)

### Purpose
Stub implementation of BitNet linear layer.

### Requirements
- Use standard BF16 GEMM
- No bias
- Preserve HF weight shapes/names

### Interface

```python
class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, prefix):
        ...
    def forward(self, x):
        return x @ weight.T
```

### Important
- DO NOT implement ternary logic
- DO NOT quantize
- MUST match HF parameter names

---

## 6.2 Activation: ReLU²

```python
def relu2(x):
    return F.relu(x) ** 2
```

---

## 6.3 Normalization (SubLN)

### Unknown: MUST replicate HF behavior exactly

Tasks:
- Inspect HF model code
- Identify:
  - pre/post norm
  - scaling formula
  - epsilon usage

### Risk
⚠️ Highest source of mismatch

---

## 6.4 Attention

### Requirements
- Use vLLM attention backend
- Causal mask
- RoPE (theta=500000)
- GQA:
  - 20 query heads
  - 5 KV heads

### Implementation Notes
- Likely reuse LLaMA attention structure
- Adapt head reshaping

---

## 6.5 MLP

Structure:
```
x → proj_up → relu² → proj_down
```

No bias.

---

## 6.6 Decoder Layer

Structure (example, verify against HF):

```
x
 ├── norm
 ├── attention
 ├── residual add
 ├── norm
 ├── mlp
 └── residual add
```

⚠️ Must confirm exact ordering

---

## 6.7 Embedding + LM Head

- Shared weights
- Shape: `[vocab_size, hidden_size]`

---

## 7. Weight Loading

---

## 7.1 Tasks

- Load `.safetensors`
- Map HF → vLLM parameter names
- Ensure:
  - QKV split correctness
  - MLP weights correct
  - embedding tied correctly

---

## 7.2 Quantization Config Handling

HF config includes:

```json
"quantization_config": {
  "quant_method": "bitnet"
}
```

### Phase 1 Rule:
👉 Ignore completely

---

## 8. vLLM Integration

---

## 8.1 Plugin Registration

```python
from vllm import ModelRegistry

ModelRegistry.register_model(
    "BitNetForCausalLM",
    "bitnet_vllm.modeling_bitnet_vllm:BitNetForCausalLM"
)
```

---

## 8.2 vLLM Requirements

- All modules must accept `prefix`
- Forward must match vLLM expectations
- Use vLLM attention APIs

---

## 9. Tokenizer Integration

---

## 9.1 Expected Behavior

- Use standard HF tokenizer
- No custom tokenizer required

### Special tokens:
- BOS: 128000
- EOS: 128001
- EOT: 128009

---

## 9.2 Chat Template

- Must match HF exactly
- Use for parity testing

---

## 10. Testing Plan

---

## 10.1 Reference Script (HF)

Create:

```
scripts/hf_reference.py
```

Outputs:
- token IDs
- logits checksum
- short generation

---

## 10.2 vLLM Tests

### A. Load Test
- model initializes
- weights load

### B. Generation Test
- simple prompt
- chat prompt

### C. Parity Test

Compare:
- top-10 logits
- generated tokens

---

## 11. Milestones

---

### Milestone 1 — HF baseline

---

### Milestone 2 — vLLM model loads

---

### Milestone 3 — forward pass works

---

### Milestone 4 — generation works

---

### Milestone 5 — parity achieved

---

### Milestone 6 — upstream PR ready

---

## 12. Risks

---

### 🔴 High Risk
- SubLN mismatch
- weight mapping errors
- attention shape mismatch

---

### 🟡 Medium Risk
- RoPE incorrect scaling
- GQA implementation bug

---

### 🟢 Low Risk
- tokenizer
- embedding tying

---

## 13. Future Work (Phase 2)

- Implement BitNet quantization backend
- Add packed weight loader
- Add custom CUDA kernels
- Integrate with vLLM quantization system

---

## 14. Instructions for AI Agent (Cursor / Antigravity)

---

### Primary Tasks

1. Implement model skeleton from LLaMA reference
2. Replace:
   - linear → BitLinear
   - activation → relu2
   - norm → SubLN
3. Implement weight loader
4. Register model
5. Write parity tests

---

### Constraints

- DO NOT use `trust_remote_code`
- DO NOT implement quantization
- DO NOT optimize performance
- PRIORITIZE correctness over speed

---

## 15. Definition of Done

- vLLM runs:
  ```
  vllm serve microsoft/bitnet-b1.58-2B-4T-bf16
  ```
- Produces valid outputs
- Matches HF generation (approx)
- Ready for upstream PR

