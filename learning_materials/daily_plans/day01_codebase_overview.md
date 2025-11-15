# Day 1: vLLM Codebase Overview & First Exploration

> **Goal**: Get comfortable with vLLM structure, run first examples, understand request flow
> **Time**: 4-6 hours
> **Prerequisites**: Environment setup complete
> **Deliverables**: Architecture diagram, traced request flow, notes on key components

---

## 📅 Daily Schedule

### Morning Session (2-3 hours): Code Exploration

**9:00-9:30** - Codebase Structure Overview
**9:30-10:30** - Run Examples & Understand Outputs
**10:30-11:00** - Break + Review Notes
**11:00-12:00** - Trace Request Flow (Python Layer)

### Afternoon Session (2-3 hours): Deep Dive

**14:00-15:00** - Key Components Analysis
**15:00-16:00** - CUDA Layer Introduction
**16:00-16:30** - Break
**16:30-17:30** - Hands-On Exercises

### Evening (Optional, 1 hour): Review & Prepare

**19:00-20:00** - Review notes, create flashcards, preview Day 2

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Navigate the vLLM codebase confidently
- [ ] Run basic inference examples
- [ ] Explain the high-level request flow
- [ ] Identify where key components live (scheduler, block manager, attention)
- [ ] Understand the Python ↔ C++/CUDA boundary

---

## 📂 Morning: Codebase Structure (9:00-12:00)

### Task 1: Directory Tree Exploration (30 min)

```bash
cd ~/projects/vllm

# View directory structure
tree -L 2 -d

# Expected output:
# .
# ├── benchmarks/          # Performance benchmarks
# ├── csrc/               # C++ and CUDA source code ⭐
# ├── docs/               # Documentation
# ├── examples/           # Example scripts ⭐
# ├── tests/              # Test suite
# ├── vllm/               # Main Python package ⭐
# │   ├── attention/      # Attention implementations
# │   ├── core/           # Scheduler, block manager
# │   ├── engine/         # LLM engine
# │   ├── entrypoints/    # API servers
# │   ├── model_executor/ # Model execution
# │   └── ...
```

**📝 Exercise**: Create a mind map of directories

```
vllm/
├── 🐍 Python API (entrypoints/)
├── 🧠 Engine (engine/)
├── 📊 Scheduler (core/)
├── 💾 Attention (attention/)
├── 🔧 Model Executor (model_executor/)
└── ⚡ CUDA Kernels (csrc/)
```

### Task 2: Run Your First Example (45 min)

**File**: `examples/offline_inference.py`

```bash
# Read the example first
cat examples/offline_inference.py

# Run it
python examples/offline_inference.py

# Expected output:
# Processed prompts: 100%|████████| 4/4 [00:03<00:00,  1.33it/s]
# Prompt: 'Hello, my name is', Generated: ' John Smith and I am...'
# ...
```

**📝 Code Reading**:

Open `examples/offline_inference.py` and annotate:

```python
from vllm import LLM, SamplingParams  # ← Main API

# Step 1: Create LLM instance
# Q: What happens during __init__?
llm = LLM(model="facebook/opt-125m")

# Step 2: Define sampling parameters
# Q: What options are available?
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Step 3: Generate
# Q: How does batching work internally?
outputs = llm.generate(prompts, sampling_params)

# Step 4: Access results
for output in outputs:
    print(output.outputs[0].text)
```

**🔍 Questions to Answer**:
1. Where is the `LLM` class defined? → Find in codebase
2. What models does vLLM support? → Check supported models
3. How does vLLM download models? → Trace model loading

**Answers**:
```bash
# Q1: LLM class location
grep -r "class LLM" vllm/
# Answer: vllm/entrypoints/llm.py

# Q2: Supported models
ls vllm/model_executor/models/
# Answer: llama.py, opt.py, gpt2.py, mistral.py, etc.

# Q3: Model loading
grep -r "download" vllm/model_executor/
# Trace: Uses HuggingFace transformers library
```

### Task 3: Trace Request Flow - Python Layer (60 min)

**Goal**: Understand what happens from `llm.generate()` to model inference

**Step-by-Step Tracing**:

#### **File 1**: `vllm/entrypoints/llm.py:LLM.generate()`

```python
class LLM:
    def generate(self, prompts, sampling_params):
        # 1. Prepare inputs
        inputs = self._prepare_inputs(prompts)

        # 2. Process through engine
        outputs = self.llm_engine.generate(inputs, sampling_params)

        return outputs

# 📌 KEY: Delegates to LLMEngine
# NEXT: Check LLMEngine.generate()
```

#### **File 2**: `vllm/engine/llm_engine.py:LLMEngine.generate()`

```python
class LLMEngine:
    def generate(self, inputs, sampling_params):
        # 1. Add requests to scheduler
        for prompt in inputs:
            self._add_request(prompt, sampling_params)

        # 2. Execute steps until all complete
        while not all_finished:
            outputs = self.step()  # ← Main execution step

        return outputs

# 📌 KEY: Scheduler manages requests, step() runs one iteration
# NEXT: Check step() method
```

#### **File 3**: `vllm/engine/llm_engine.py:LLMEngine.step()`

```python
def step(self):
    # 1. Schedule: Decide which requests to process
    scheduler_outputs = self.scheduler.schedule()

    # 2. Execute: Run model on scheduled requests
    model_outputs = self.model_executor.execute_model(
        scheduler_outputs
    )

    # 3. Process outputs: Sample tokens, update sequences
    outputs = self._process_outputs(model_outputs)

    return outputs

# 📌 KEY: Three phases: Schedule → Execute → Process
# COMPONENTS: Scheduler, ModelExecutor, Sampler
```

**📊 Create Flow Diagram**:

```
User Call: llm.generate(prompts)
           ↓
    [LLM entrypoint]
           ↓
    [LLMEngine.generate()]
           ↓
    ┌──────────────┐
    │ Add requests │
    │ to scheduler │
    └──────────────┘
           ↓
    ┌─────────────────────┐
    │ While not finished: │
    │   step()            │
    └─────────────────────┘
           ↓
    ┌────────────────────┐
    │ Scheduler.schedule()│ ← Select batches
    └────────────────────┘
           ↓
    ┌──────────────────────┐
    │ ModelExecutor.execute│ ← GPU inference
    └──────────────────────┘
           ↓
    ┌──────────────────────┐
    │ Process outputs      │ ← Sample tokens
    │ Update sequences     │
    └──────────────────────┘
```

### 🎯 Morning Checkpoint

**Quiz Yourself**:
1. What are the 3 main steps in `LLMEngine.step()`?
2. Where is scheduling logic located?
3. What component runs the actual model inference?

**Answers**:
1. Schedule → Execute → Process
2. `vllm/core/scheduler.py`
3. `ModelExecutor` in `vllm/model_executor/`

---

## 🔬 Afternoon: Deep Dive into Components (14:00-17:30)

### Task 4: Scheduler Deep Dive (60 min)

**File**: `vllm/core/scheduler.py:Scheduler`

**What does the scheduler do?**
1. Manages running/waiting/finished sequences
2. Decides which sequences to run in next step (batching)
3. Allocates GPU blocks via block manager
4. Handles preemption when out of memory

**Key Method**: `schedule()`

```python
# vllm/core/scheduler.py:Scheduler.schedule()

def schedule(self):
    # 1. Try to schedule waiting requests
    while self.waiting:
        seq = self.waiting[0]

        # Can we allocate blocks for this sequence?
        if not self._can_allocate(seq):
            break  # Out of memory

        # Allocate blocks
        self._allocate(seq)

        # Move to running
        self.running.append(seq)
        self.waiting.pop(0)

    # 2. Schedule running sequences for next step
    scheduled_seqs = self.running

    # 3. If out of memory, preempt lowest priority
    if not enough_memory:
        self._preempt_lowest_priority()

    return SchedulerOutputs(scheduled_seqs)
```

**📝 Exercise**: Trace a scenario

```
Initial state:
- Waiting: [Seq A (100 tokens), Seq B (50 tokens)]
- Running: []
- Available blocks: 10 (block_size=16)

Step 1: schedule()
  - Seq A needs 7 blocks → Allocate → Move to Running
  - Seq B needs 4 blocks, but only 3 left → Stays in Waiting
  - Result: Running=[Seq A], Waiting=[Seq B]

Step 2: Seq A generates 10 more tokens (110 total)
  - Needs 1 more block → Allocate
  - Result: A uses 8 blocks, 2 available

Step 3: schedule()
  - Seq B needs 4 blocks, but only 2 available
  - Preempt Seq A? Or wait?
  - Decision based on policy (FCFS, priority, etc.)
```

### Task 5: Block Manager Exploration (60 min)

**File**: `vllm/core/block_manager_v2.py:BlockManager`

**Navigate the code**:

```bash
# Open the file
code vllm/core/block_manager_v2.py

# Or read in terminal
less vllm/core/block_manager_v2.py
```

**Key Concepts**:

```python
class BlockManager:
    def __init__(self, block_size, num_gpu_blocks, num_cpu_blocks):
        self.block_size = 16  # tokens per block
        self.num_gpu_blocks = 1000  # e.g., for 40GB GPU

    def allocate(self, seq_id, num_blocks):
        """Allocate blocks for a sequence"""
        pass

    def free(self, seq_id):
        """Free all blocks for a sequence"""
        pass

    def can_allocate(self, num_blocks):
        """Check if enough free blocks"""
        pass
```

**📊 Visualization Exercise**:

Draw the state of block manager:

```
GPU Memory (20 blocks total, block_size=16):
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ A0 │ B0 │ A1 │ C0 │ B1 │ A2 │ C1 │FREE│FREE│FREE│
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

Block Tables:
A: [0, 2, 5]     → 48 tokens
B: [1, 4]        → 32 tokens
C: [3, 6]        → 32 tokens

Free blocks: [7, 8, 9, ...]
```

### Task 6: CUDA Layer Introduction (45 min)

**Explore CUDA kernels**:

```bash
# List CUDA files
ls csrc/attention/

# Expected:
# attention_kernels.cu          ← Main kernels ⭐
# attention_generic.cuh          ← Generic implementation
# dtype_float16.cuh              ← FP16 specialization
# dtype_float32.cuh              ← FP32 specialization
```

**Read kernel signature**:

```bash
# View kernel entry point
grep -A 20 "paged_attention_v1_kernel" csrc/attention/attention_kernels.cu
```

**Understanding**:

```cuda
// Kernel signature (simplified)
__global__ void paged_attention_v1_kernel(
    float* out,              // Output
    const float* q,          // Query
    const float* k_cache,    // Key cache (paged)
    const float* v_cache,    // Value cache (paged)
    const int* block_tables, // Block table mapping
    const int* context_lens, // Sequence lengths
    float scale              // Attention scale
) {
    // Compute attention with paged K/V cache
}
```

**📝 Questions**:
1. Why is `k_cache` called "paged"? → Review Part 1
2. What does `block_tables` contain? → Mapping logical→physical blocks
3. How many threads per block? → Check kernel launch config

### Task 7: Hands-On Exercise (60 min)

**Exercise 1**: Modify example to print intermediate info

Create `my_first_vllm_test.py`:

```python
#!/usr/bin/env python3
"""
Day 1 Exercise: Instrument vLLM to understand execution
"""

from vllm import LLM, SamplingParams
import time

def test_with_instrumentation():
    print("=" * 50)
    print("Step 1: Creating LLM")
    print("=" * 50)
    start = time.time()
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=512,
        gpu_memory_utilization=0.5
    )
    print(f"✓ LLM created in {time.time() - start:.2f}s")

    print("\n" + "=" * 50)
    print("Step 2: Generating Text")
    print("=" * 50)

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In the year 2050,",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"✓ Generated in {time.time() - start:.2f}s")

    print("\n" + "=" * 50)
    print("Step 3: Results")
    print("=" * 50)

    for i, output in enumerate(outputs):
        print(f"\nPrompt {i+1}: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print(f"Tokens: {len(output.outputs[0].token_ids)}")
        print(f"Finish reason: {output.outputs[0].finish_reason}")

if __name__ == "__main__":
    test_with_instrumentation()
```

Run it:
```bash
python my_first_vllm_test.py
```

**Exercise 2**: Experiment with parameters

```python
# Try different settings:
# 1. Temperature: 0.1, 0.5, 1.0, 2.0
# 2. Max tokens: 10, 50, 200
# 3. Top_p: 0.5, 0.9, 0.99

# Observe differences in output
```

**Exercise 3**: Trace with debugging

```python
# Add breakpoint in vLLM code
# File: vllm/engine/llm_engine.py

def step(self):
    import pdb; pdb.set_trace()  # ← Add this
    scheduler_outputs = self.scheduler.schedule()
    # ...

# Run again and explore:
# (pdb) p scheduler_outputs
# (pdb) p self.scheduler.running
# (pdb) p self.scheduler.waiting
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Codebase Structure**
- Main Python package: `vllm/`
- CUDA kernels: `csrc/`
- Examples and tests

✅ **Request Flow**
- LLM entrypoint → Engine → Scheduler → Executor

✅ **Key Components**
- Scheduler: Batching and resource management
- BlockManager: Memory allocation
- ModelExecutor: GPU inference
- Attention: PagedAttention kernels

✅ **Python ↔ C++ Boundary**
- Python wrappers in `vllm/`
- C++ implementations in `csrc/`
- PyTorch custom ops for binding

### Knowledge Check (Quiz)

**Question 1**: What are the three phases in `LLMEngine.step()`?
<details>
<summary>Answer</summary>
1. Schedule (Scheduler.schedule())
2. Execute (ModelExecutor.execute_model())
3. Process outputs (sampling, updating sequences)
</details>

**Question 2**: What does the BlockManager do?
<details>
<summary>Answer</summary>
Manages allocation and deallocation of fixed-size memory blocks for KV cache storage. Maps logical blocks to physical GPU memory blocks.
</details>

**Question 3**: Why is vLLM's attention called "paged"?
<details>
<summary>Answer</summary>
Because it uses virtual memory-style paging: KV cache is split into fixed-size blocks (pages) that can be non-contiguous in physical memory, mapped via a page table (block table).
</details>

### Daily Reflection

**What went well?**
- [ ] I can navigate the codebase
- [ ] I understand the basic request flow
- [ ] I ran examples successfully

**What was challenging?**
- [ ] Understanding async execution
- [ ] Complexity of scheduler logic
- [ ] CUDA kernel code (normal, we'll dive deeper!)

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 2

Tomorrow you'll dive deeper into:
- **Attention Mechanism**: How vLLM implements attention
- **PagedAttention Details**: Theory and implementation (Part 1)
- **Memory Management**: Block allocation strategies
- **First Profile**: Using Nsight to see kernel execution

**Preparation**:
- Review PagedAttention Part 1 (theory)
- Refresh understanding of attention mechanism
- Ensure profiling tools work (nsys, ncu)

---

## 📚 Additional Resources

**Today's Reading**:
- [ ] vLLM blog post: [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html)
- [ ] README.md in vLLM repo
- [ ] `docs/` folder overview

**Optional Deep Dive**:
- [ ] Read `vllm/engine/llm_engine.py` completely
- [ ] Explore `vllm/core/scheduler.py`
- [ ] Browse model implementations in `vllm/model_executor/models/`

---

**Congratulations on completing Day 1! 🎉**

**Time to commit your notes to memory and prepare for tomorrow's deeper dive!**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
