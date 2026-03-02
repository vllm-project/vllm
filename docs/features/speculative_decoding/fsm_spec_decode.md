# FSM Speculation

FSM (Finite State Machine) speculation is a speculative decoding method where the FSM proposes draft tokens and no rejection happens for deterministic tokens. It constrains generation to follow specific patterns defined by a state machine. It fast-forwards through deterministic token sequences, reducing latency for structured outputs like JSON or templated text.

```python
from vllm import LLM, SamplingParams
from vllm.custom_fsm import CustomFSM

# Create FSM
fsm = CustomFSM()
tokenizer = ...  # Get tokenizer from your model

# Define structure: "hello world" then "test" or "example"
hello_token = tokenizer.encode("hello", add_special_tokens=False)[0]
world_token = tokenizer.encode(" world", add_special_tokens=False)[0]
test_token = tokenizer.encode(" test", add_special_tokens=False)[0]
example_token = tokenizer.encode(" example", add_special_tokens=False)[0]
eos_token = tokenizer.eos_token_id

# Build FSM graph
fsm.graph[0] = {hello_token: 1}
fsm.graph[1] = {world_token: 2}
fsm.graph[2] = {test_token: 3, example_token: 4}  # Branch
fsm.graph[3] = {eos_token: 5}
fsm.graph[4] = {eos_token: 6}

# Save FSM
fsm_path = "/tmp/fsm.json"
fsm.save(fsm_path)

# Use with vLLM
prompts = ["Generate text:"]
sampling_params = SamplingParams(temperature=0, max_tokens=10)

llm = LLM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    speculative_config={
        "method": "fsm",
        "fsm_path": fsm_path,
        "num_speculative_tokens": 3,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Generated: {output.outputs[0].text}")
```

## FSM Structure

FSMs are represented as nested dictionaries:

```python
fsm.graph[current_state][token_id] = next_state
```

Special token values:

- Token IDs ≥ 0: Specific token required for state transition
- `-1`: Wildcard, any token allowed
- EOS token: Terminates generation

## Examples

### Branching

```python
# "hello" then either "world" or "universe"
fsm.graph[0] = {hello_token: 1}
fsm.graph[1] = {world_token: 2, universe_token: 3}  # Branch
fsm.graph[2] = {eos_token: 4}
fsm.graph[3] = {eos_token: 5}
```

### Wildcard

```python
# "hello" then any token, then "goodbye"
fsm.graph[0] = {hello_token: 1}
fsm.graph[1] = {-1: 2}  # Wildcard: any token allowed
fsm.graph[2] = {goodbye_token: 3}
fsm.graph[3] = {eos_token: 4}
```

## Configuration

| Parameter | Description |
|-----------|-------------|
| `method` | Must be `"fsm"` |
| `fsm_path` | Path to JSON file with FSM graph |
| `num_speculative_tokens` | Max tokens to speculate |
