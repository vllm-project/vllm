# Day 3: Request Lifecycle - From API to Output Token

> **Goal**: Understand complete request flow from API call to generated tokens, trace execution through all components
> **Time**: 6-8 hours
> **Prerequisites**: Day 1-2 completed, debug build ready, basic understanding of transformers
> **Deliverables**: Annotated request flow diagram, traced code with notes, custom logging implementation

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Request Flow Analysis

**9:00-9:45** - High-Level Request Flow Review
**9:45-11:00** - Python Layer: API to Engine (LLM → LLMEngine)
**11:00-11:30** - Break + Draw Flow Diagram
**11:30-12:30** - Engine Layer: Scheduling and Execution Loop

### Afternoon Session (3-4 hours): Deep Component Dive

**14:00-15:00** - Scheduler: Request Batching & Memory Allocation
**15:00-16:00** - Model Executor: Forward Pass Orchestration
**16:00-16:30** - Break
**16:30-18:00** - Hands-On: Trace Real Request with Logging

### Evening (Optional, 1 hour): Integration

**19:00-20:00** - Create comprehensive flow diagram, prepare questions for Week 1 review

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Trace a complete request from `llm.generate()` to output tokens
- [ ] Explain each component's role in the pipeline
- [ ] Understand how batching and scheduling work
- [ ] Identify bottlenecks and optimization points
- [ ] Add custom instrumentation to track requests
- [ ] Read and understand the core engine code

---

## 🚀 Morning: Request Flow Deep Dive (9:00-12:30)

### Task 1: High-Level Flow Review (45 min)

**The Complete Journey of a Request**:

```
User Input: "Hello, my name is"
           ↓
┌─────────────────────────────────────────┐
│ 1. API Layer (vllm/entrypoints/llm.py) │
│    - LLM.generate()                      │
│    - Input validation & preprocessing    │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 2. Engine Layer (vllm/engine/)          │
│    - LLMEngine.generate()                │
│    - Create SequenceGroup                │
│    - Add to scheduler queue              │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 3. Execution Loop (LLMEngine.step())    │
│    ┌───────────────────────────────┐   │
│    │ a) Scheduler.schedule()        │   │
│    │    - Select requests to run    │   │
│    │    - Allocate KV cache blocks  │   │
│    └───────────────────────────────┘   │
│    ┌───────────────────────────────┐   │
│    │ b) ModelExecutor.execute()     │   │
│    │    - Prepare inputs            │   │
│    │    - Run model forward         │   │
│    │    - Get logits                │   │
│    └───────────────────────────────┘   │
│    ┌───────────────────────────────┐   │
│    │ c) Process outputs             │   │
│    │    - Sample next token         │   │
│    │    - Update sequences          │   │
│    │    - Check completion          │   │
│    └───────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               ↓
           Output: "Hello, my name is John"
```

**Key Insight**: Each request goes through multiple iterations of the loop. Each iteration:
- Prefill phase: Process input tokens in parallel
- Decode phase: Generate one token at a time

### Task 2: API Layer Deep Dive (75 min)

**File**: `vllm/entrypoints/llm.py`

Let's read the actual code:

```bash
# Open in editor
code vllm/entrypoints/llm.py

# Or read specific lines
sed -n '1,100p' vllm/entrypoints/llm.py
```

**Key Class: LLM**

```python
# vllm/entrypoints/llm.py (simplified)

class LLM:
    """Main entry point for offline inference."""

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """
        Initialize vLLM engine.

        Key steps:
        1. Load tokenizer
        2. Determine model configuration
        3. Initialize LLMEngine
        4. Allocate GPU blocks
        """
        # 1. Create engine config
        self.llm_engine = LLMEngine.from_engine_args(
            EngineArgs(
                model=model,
                dtype=dtype,
                max_model_len=max_model_len,
                # ...
            )
        )

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """
        Generate completions for prompts.

        Args:
            prompts: Input text(s)
            sampling_params: Sampling configuration

        Returns:
            List of RequestOutput with generated texts
        """
        # 1. Normalize inputs
        if isinstance(prompts, str):
            prompts = [prompts]

        # 2. Create sampling params if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()

        # 3. Add requests to engine
        for i, prompt in enumerate(prompts):
            self._add_request(
                request_id=str(i),
                prompt=prompt,
                sampling_params=sampling_params,
            )

        # 4. Run until all requests complete
        return self._run_engine(use_tqdm=True)

    def _add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> None:
        """Add request to engine queue."""
        # Tokenize input
        token_ids = self.tokenizer.encode(prompt)

        # Add to engine
        self.llm_engine.add_request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
        )

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        """Execute engine steps until all requests complete."""
        outputs: List[RequestOutput] = []

        with tqdm(total=len(self.llm_engine.scheduler.waiting)) as pbar:
            while self.llm_engine.has_unfinished_requests():
                # ⭐ KEY: Single execution step
                step_outputs = self.llm_engine.step()

                # Collect completed requests
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
                        pbar.update(1)

        return outputs
```

**📝 Exercise**: Trace `generate()` call

Given: `llm.generate(["Hello", "Goodbye"], sampling_params)`

1. `_add_request()` called twice → 2 requests in queue
2. `_run_engine()` starts loop
3. Each `step()` processes batch of requests
4. Loop continues until both complete

**Breakpoint Practice**:

```python
# Add to vllm/entrypoints/llm.py:_add_request()

def _add_request(self, request_id, prompt, sampling_params):
    token_ids = self.tokenizer.encode(prompt)

    # ← Add breakpoint here
    import pdb; pdb.set_trace()

    print(f"Request {request_id}: {len(token_ids)} tokens")
    # Inspect: p token_ids, p sampling_params

    self.llm_engine.add_request(...)
```

### Task 3: Engine Layer - The Heart (60 min)

**File**: `vllm/engine/llm_engine.py`

**Key Class: LLMEngine**

```python
# vllm/engine/llm_engine.py (simplified structure)

class LLMEngine:
    """
    Core engine managing:
    - Request scheduling
    - Model execution
    - Output processing
    """

    def __init__(self, ...):
        """Initialize engine components."""
        # 1. Create model executor (runs model on GPU)
        self.model_executor = GPUExecutor(
            model_config=model_config,
            cache_config=cache_config,
        )

        # 2. Create scheduler (manages requests & memory)
        self.scheduler = Scheduler(
            scheduler_config=scheduler_config,
            cache_config=cache_config,
        )

        # 3. Create output processor
        self.output_processor = OutputProcessor(
            tokenizer=self.tokenizer,
        )

    def add_request(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> None:
        """
        Add new request to scheduler.

        Creates SequenceGroup to track the request.
        """
        # 1. Create Sequence
        seq = Sequence(
            seq_id=self._get_next_seq_id(),
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
        )

        # 2. Create SequenceGroup (can have multiple seqs for beam search)
        seq_group = SequenceGroup(
            request_id=request_id,
            seqs=[seq],
            sampling_params=sampling_params,
        )

        # 3. Add to scheduler's waiting queue
        self.scheduler.add_seq_group(seq_group)

    def step(self) -> List[RequestOutput]:
        """
        ⭐ CORE METHOD ⭐

        Single execution step:
        1. Schedule which requests to run
        2. Execute model forward pass
        3. Sample tokens and update state

        Returns outputs for finished requests.
        """

        # === PHASE 1: SCHEDULE ===
        scheduler_outputs = self.scheduler.schedule()

        # scheduler_outputs contains:
        # - scheduled_seq_groups: Requests to run this step
        # - blocks_to_swap_in/out: Memory management
        # - blocks_to_copy: For beam search
        # - num_batched_tokens: Total tokens in batch

        if scheduler_outputs.is_empty():
            return []  # Nothing to do

        # === PHASE 2: EXECUTE MODEL ===
        output = self.model_executor.execute_model(
            seq_group_metadata_list=scheduler_outputs.seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )

        # output is SamplerOutput containing:
        # - logits or sampled token IDs
        # - probabilities if requested

        # === PHASE 3: PROCESS OUTPUTS ===
        request_outputs = self._process_model_outputs(
            output=output,
            scheduler_outputs=scheduler_outputs,
        )

        # Update sequences with new tokens
        # Check for completion (EOS token or max length)
        # Free memory for finished sequences

        return request_outputs

    def _process_model_outputs(
        self,
        output: SamplerOutput,
        scheduler_outputs: SchedulerOutputs,
    ) -> List[RequestOutput]:
        """
        Process model outputs:
        1. Sample next tokens
        2. Update sequences
        3. Return completed requests
        """
        request_outputs = []

        # For each sequence in the batch
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            # Get sampled token for this sequence
            sample = output.samples[seq_group.seq_id]

            # Append token to sequence
            seq_group.sequences[0].append_token_id(
                token_id=sample.token_id,
                logprob=sample.logprob,
            )

            # Check if finished
            if self._is_finished(seq_group):
                # Mark as finished
                seq_group.is_finished = True

                # Free allocated blocks
                self.scheduler.free_seq_group(seq_group)

                # Create output
                request_outputs.append(
                    RequestOutput.from_seq_group(seq_group)
                )

        return request_outputs

    def _is_finished(self, seq_group: SequenceGroup) -> bool:
        """Check if sequence is complete."""
        seq = seq_group.sequences[0]

        # Finished if:
        # 1. Generated EOS token
        if seq.get_last_token_id() == self.tokenizer.eos_token_id:
            return True

        # 2. Reached max length
        if len(seq.token_ids) >= self.max_model_len:
            return True

        # 3. Generated max_tokens from sampling params
        if seq.get_output_len() >= seq_group.sampling_params.max_tokens:
            return True

        return False
```

**📊 Visualize Step Execution**:

```
Step 0 (Prefill):
  Input: "Hello, my name is" [5 tokens]
  Schedule: Process all 5 tokens in parallel
  Execute: Forward pass through model
  Output: Logits for next token
  Sample: "John" (token_id=2215)
  State: "Hello, my name is John" [6 tokens]

Step 1 (Decode):
  Input: Previous state + new token position
  Schedule: Process 1 new token
  Execute: Forward pass (using KV cache)
  Output: Logits for next token
  Sample: "Smith" (token_id=3421)
  State: "Hello, my name is John Smith" [7 tokens]

Step 2 (Decode):
  ... repeat until EOS or max_tokens ...
```

---

## 🔬 Afternoon: Component Deep Dive (14:00-18:00)

### Task 4: Scheduler Internals (60 min)

**File**: `vllm/core/scheduler.py`

**Understanding Scheduler.schedule()**:

```python
# vllm/core/scheduler.py (simplified)

class Scheduler:
    """
    Manages request scheduling and memory allocation.

    Key responsibilities:
    - Maintain waiting/running/swapped queues
    - Allocate KV cache blocks via BlockManager
    - Decide batch composition each step
    - Handle preemption when out of memory
    """

    def __init__(self, scheduler_config, cache_config):
        # Queues for different states
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: Deque[SequenceGroup] = deque()

        # Block manager for memory
        self.block_manager = BlockSpaceManager(
            block_size=cache_config.block_size,
            num_gpu_blocks=cache_config.num_gpu_blocks,
            num_cpu_blocks=cache_config.num_cpu_blocks,
        )

        # Configuration
        self.max_num_seqs = scheduler_config.max_num_seqs
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens

    def schedule(self) -> SchedulerOutputs:
        """
        Main scheduling logic.

        Steps:
        1. Try to swap in swapped requests
        2. Schedule running requests (decode)
        3. Schedule waiting requests (prefill)
        4. Handle preemption if needed
        """

        # Lists to populate
        scheduled_seq_groups: List[SequenceGroup] = []
        num_batched_tokens = 0

        # === STEP 1: Swap in ===
        blocks_to_swap_in = []
        while self.swapped:
            seq_group = self.swapped[0]

            # Can we swap in? (enough GPU blocks)
            if not self._can_swap_in(seq_group):
                break

            self._swap_in(seq_group, blocks_to_swap_in)
            self.running.append(seq_group)
            self.swapped.popleft()

        # === STEP 2: Schedule running (decode) ===
        # Running requests get priority
        running_scheduled = []

        for seq_group in self.running:
            # Each running seq generates 1 token
            num_tokens = 1

            # Check batch size limit
            if num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                break

            # Allocate new block if needed
            if not self._can_allocate(seq_group):
                # Out of memory - need to preempt
                self._preempt(seq_group)
                continue

            self._allocate(seq_group)
            running_scheduled.append(seq_group)
            num_batched_tokens += num_tokens

        # === STEP 3: Schedule waiting (prefill) ===
        # Try to schedule new requests
        waiting_scheduled = []

        while self.waiting:
            seq_group = self.waiting[0]

            # Prefill processes all prompt tokens
            num_tokens = seq_group.get_seqs()[0].get_len()

            # Check limits
            if len(scheduled_seq_groups) >= self.max_num_seqs:
                break
            if num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                break

            # Can we allocate blocks?
            if not self._can_allocate(seq_group):
                break  # No more memory

            self._allocate(seq_group)
            self.waiting.popleft()
            self.running.append(seq_group)
            waiting_scheduled.append(seq_group)
            num_batched_tokens += num_tokens

        scheduled_seq_groups = running_scheduled + waiting_scheduled

        # === STEP 4: Create outputs ===
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=[],
            blocks_to_copy=[],
        )

    def _can_allocate(self, seq_group: SequenceGroup) -> bool:
        """Check if we can allocate blocks for this sequence."""
        num_required_blocks = self._get_num_required_blocks(seq_group)
        return self.block_manager.can_allocate(num_required_blocks)

    def _allocate(self, seq_group: SequenceGroup) -> None:
        """Allocate blocks for sequence."""
        self.block_manager.allocate(seq_group)

    def _preempt(self, seq_group: SequenceGroup) -> None:
        """Preempt sequence (swap out or recompute)."""
        # Decide: swap to CPU or recompute from scratch
        if self.swap_space_available():
            self._swap_out(seq_group)
            self.swapped.append(seq_group)
        else:
            # Recompute - move back to waiting
            self._free(seq_group)
            self.waiting.appendleft(seq_group)

        self.running.remove(seq_group)
```

**📝 Exercise: Trace Scheduling Scenario**

```
Initial State:
  Waiting: [A (prompt=50 tokens), B (prompt=30 tokens)]
  Running: [C (generated 10 tokens so far)]
  GPU Blocks: 100 total, 80 used by C, 20 free
  max_num_batched_tokens: 64

Step 1: schedule()

  1. Swap in: (none in swapped)

  2. Schedule running:
     - C: needs 1 token (decode)
     - num_batched_tokens = 1
     - Check allocation: C needs 1 more block (has grown)
       - Can allocate? Yes (20 free blocks)
     - Allocate and schedule C
     - Result: scheduled=[C], num_batched_tokens=1

  3. Schedule waiting:
     - A: needs 50 tokens (prefill)
     - num_batched_tokens = 1 + 50 = 51 ≤ 64 ✓
     - Check allocation: A needs 4 blocks (50/16 rounded up)
       - Can allocate? Yes (19 free blocks remaining)
     - Allocate and schedule A
     - Move A to running
     - Result: scheduled=[C, A], num_batched_tokens=51

     - B: needs 30 tokens
     - num_batched_tokens = 51 + 30 = 81 > 64 ✗
     - Cannot fit in batch - stays in waiting

  Output: SchedulerOutputs(
    scheduled=[C, A],
    num_batched_tokens=51
  )

New State:
  Waiting: [B]
  Running: [C (11 tokens), A (50 tokens)]
  Batched: C (1 decode) + A (50 prefill) = 51 tokens
```

### Task 5: Model Executor (60 min)

**File**: `vllm/model_executor/model_runner.py`

**Key Method: execute_model()**

```python
# vllm/model_executor/model_runner.py

class ModelRunner:
    """Executes model on GPU."""

    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        """
        Execute model forward pass.

        Steps:
        1. Prepare input tensors
        2. Handle block operations (swap, copy)
        3. Run model forward
        4. Sample next tokens
        """

        # === STEP 1: Prepare inputs ===
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list
        )

        # input_tokens: [num_tokens] - All tokens in batch
        # input_positions: [num_tokens] - Position of each token
        # input_metadata: Metadata for attention (block tables, etc.)

        # === STEP 2: Memory operations ===
        if blocks_to_swap_in:
            self._swap_blocks(blocks_to_swap_in, "in")

        if blocks_to_swap_out:
            self._swap_blocks(blocks_to_swap_out, "out")

        if blocks_to_copy:
            self._copy_blocks(blocks_to_copy)

        # === STEP 3: Model forward ===
        hidden_states = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.kv_cache,
            input_metadata=input_metadata,
        )

        # hidden_states: [num_tokens, hidden_dim]

        # === STEP 4: Sample ===
        sampler_output = self.model.sampler(
            hidden_states=hidden_states,
            sampling_metadata=self._prepare_sampling_metadata(
                seq_group_metadata_list
            ),
        )

        return sampler_output

    def _prepare_inputs(self, seq_group_metadata_list):
        """
        Prepare batched inputs from multiple sequences.

        Combines prefill and decode requests into single batch.
        """
        input_tokens = []
        input_positions = []
        slot_mapping = []
        block_tables = []

        for seq_group_meta in seq_group_metadata_list:
            seq_data = seq_group_meta.seq_data

            if seq_group_meta.is_prompt:  # Prefill
                # All prompt tokens
                tokens = seq_data.get_prompt_token_ids()
                positions = list(range(len(tokens)))

            else:  # Decode
                # Just the last token
                tokens = [seq_data.get_last_token_id()]
                positions = [seq_data.get_len() - 1]

            input_tokens.extend(tokens)
            input_positions.extend(positions)

            # Block table for this sequence
            block_table = seq_group_meta.block_tables[seq_id]
            block_tables.append(block_table)

        # Convert to tensors
        input_tokens = torch.tensor(input_tokens, device="cuda")
        input_positions = torch.tensor(input_positions, device="cuda")

        return input_tokens, input_positions, metadata
```

**Understanding Batching**:

```
Batch contents:
  Seq A (prefill): tokens=[101, 102, 103, 104, 105]  (5 tokens)
  Seq B (decode):  tokens=[999]                       (1 token)
  Seq C (decode):  tokens=[888]                       (1 token)

Batched input_tokens:    [101, 102, 103, 104, 105, 999, 888]  (7 tokens)
Batched input_positions: [0,   1,   2,   3,   4,   0,   5  ]

Model processes all 7 tokens in single forward pass!
```

### Task 6: Hands-On Request Tracing (90 min)

**Exercise 1: Add Comprehensive Logging**

Create `trace_request.py`:

```python
#!/usr/bin/env python3
"""
Day 3: Trace complete request lifecycle with logging
"""

import logging
from vllm import LLM, SamplingParams

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)

# Enable vLLM internal logging
logging.getLogger("vllm").setLevel(logging.DEBUG)

def main():
    print("=" * 60)
    print("TRACING REQUEST LIFECYCLE")
    print("=" * 60)

    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=256,
        gpu_memory_utilization=0.3,
    )

    prompt = "The future of AI is"
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=15,
    )

    print(f"\n📥 Input: {prompt!r}")

    outputs = llm.generate([prompt], sampling_params)

    print(f"\n📤 Output: {outputs[0].outputs[0].text!r}")
    print(f"\n📊 Stats:")
    print(f"  - Prompt tokens: {len(outputs[0].prompt_token_ids)}")
    print(f"  - Generated tokens: {len(outputs[0].outputs[0].token_ids)}")
    print(f"  - Total tokens: {len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids)}")

if __name__ == "__main__":
    main()
```

**Exercise 2: Custom Instrumentation**

Add detailed prints to vLLM code:

```python
# File: vllm/engine/llm_engine.py

def step(self):
    print(f"\n{'='*50}")
    print(f"STEP {self.step_counter}")
    print(f"{'='*50}")

    # Schedule
    print("📅 SCHEDULING...")
    scheduler_outputs = self.scheduler.schedule()
    print(f"  → Scheduled {len(scheduler_outputs.scheduled_seq_groups)} seq groups")
    print(f"  → Total tokens in batch: {scheduler_outputs.num_batched_tokens}")

    if scheduler_outputs.is_empty():
        return []

    # Execute
    print("🚀 EXECUTING MODEL...")
    output = self.model_executor.execute_model(...)
    print(f"  → Model executed, got {len(output.samples)} samples")

    # Process
    print("⚙️  PROCESSING OUTPUTS...")
    request_outputs = self._process_model_outputs(output, scheduler_outputs)
    print(f"  → Finished {len(request_outputs)} requests")

    self.step_counter += 1
    return request_outputs
```

**Exercise 3: Step-by-Step Debugger Trace**

```python
# Use debugger to inspect each step

import pdb

# In vllm/engine/llm_engine.py:step()
def step(self):
    pdb.set_trace()  # ← Breakpoint

    scheduler_outputs = self.scheduler.schedule()

    # When hit:
    # (Pdb) p self.scheduler.waiting
    # (Pdb) p self.scheduler.running
    # (Pdb) n  # Next line
    # (Pdb) p scheduler_outputs.scheduled_seq_groups
    # (Pdb) c  # Continue to next step
```

**Exercise 4: Visualize Request State Machine**

```python
# Create request_state_machine.py

from vllm import LLM, SamplingParams
from vllm.sequence import SequenceStatus

# Monkey-patch to track state changes
original_set_status = None

def track_status_change(self, status):
    print(f"  Seq {self.seq_id}: {self.status} → {status}")
    original_set_status(self, status)

# Apply patch
from vllm import sequence
original_set_status = sequence.Sequence.set_status
sequence.Sequence.set_status = track_status_change

# Run inference
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=5))

# Output shows state transitions:
# Seq 0: WAITING → RUNNING
# Seq 0: RUNNING → RUNNING
# Seq 0: RUNNING → FINISHED_STOPPED
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Complete Request Flow**
- From LLM.generate() through engine to output
- Three-phase execution: Schedule → Execute → Process
- How requests transition through queues

✅ **Scheduler Deep Dive**
- Waiting/Running/Swapped queue management
- Block allocation and preemption
- Batching strategy (prefill + decode)

✅ **Model Executor**
- Input preparation and batching
- Forward pass orchestration
- Integration with KV cache

✅ **Request Lifecycle States**
- WAITING → RUNNING → FINISHED
- Memory allocation/deallocation
- Completion detection

### Knowledge Check (Quiz)

**Question 1**: What are the three phases of `LLMEngine.step()`?
<details>
<summary>Answer</summary>
1. **Schedule**: Scheduler.schedule() decides which requests to process
2. **Execute**: ModelExecutor.execute_model() runs model forward pass
3. **Process**: Sample tokens, update sequences, check completion
</details>

**Question 2**: What's the difference between prefill and decode?
<details>
<summary>Answer</summary>
- **Prefill**: Process all prompt tokens in parallel (first step for new request)
- **Decode**: Generate one token at a time (subsequent steps)
Prefill is compute-bound, decode is memory-bound (KV cache access).
</details>

**Question 3**: How does the scheduler prioritize requests?
<details>
<summary>Answer</summary>
Priority order:
1. Swap in (swapped requests)
2. Running requests (decode) - already allocated
3. Waiting requests (prefill) - new allocations
This ensures running requests make progress.
</details>

**Question 4**: Can prefill and decode be batched together?
<details>
<summary>Answer</summary>
Yes! vLLM batches them together as long as total tokens ≤ max_num_batched_tokens. Example: 3 decode (3 tokens) + 1 prefill (50 tokens) = 53 tokens in one batch.
</details>

**Question 5**: When does a sequence move from WAITING to RUNNING?
<details>
<summary>Answer</summary>
When the scheduler successfully:
1. Has capacity (batch size limits)
2. Allocates required GPU blocks for KV cache
3. Selects it in schedule() method
Then it moves from waiting queue to running list.
</details>

### Daily Reflection

**What went well?**
- [ ] Traced complete request flow
- [ ] Understood scheduler logic
- [ ] Added custom logging successfully

**What was challenging?**
- [ ] Complex batching logic
- [ ] Multiple queue management
- [ ] State transitions

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 4

Tomorrow's deep dive:
- **PagedAttention Algorithm**: Detailed theory and benefits
- **Memory Management**: Why paging is revolutionary
- **Implementation Details**: Code walkthrough of attention kernels
- **Hands-On**: Implement simplified PagedAttention in PyTorch

**Preparation**:
- Review attention mechanism (Q, K, V matrices)
- Read about virtual memory paging (OS concepts)
- Ensure you understand KV cache basics

---

## 📚 Additional Resources

**Code Reading**:
- [ ] Complete `vllm/engine/llm_engine.py`
- [ ] Complete `vllm/core/scheduler.py`
- [ ] `vllm/model_executor/model_runner.py`
- [ ] `vllm/sequence.py` (Sequence and SequenceGroup classes)

**Concepts**:
- [ ] [Continuous Batching Blog Post](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [ ] Attention mechanism refresher
- [ ] GPU memory management basics

**Optional Deep Dive**:
- [ ] Explore `vllm/core/block_manager_v2.py`
- [ ] Read about different scheduling policies
- [ ] Compare with other inference engines (TGI, TRT-LLM)

---

**Congratulations on mastering request flow! 🎉**

**You now understand how vLLM orchestrates LLM inference!**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
