# Weighted Fair Queuing (WFQ) Usage Guide

## Overview

Weighted Fair Queuing (WFQ) is an advanced scheduling policy for vLLM that provides proportional fairness based on request weights. It ensures that requests with higher weights receive proportionally more resources while preventing starvation of low-weight requests.

## When to Use WFQ

Use WFQ when you need:

- **Proportional Resource Allocation**: Different request classes should receive resources proportional to their importance
- **Quality-of-Service (QoS) Guarantees**: Premium users or critical workloads need prioritized service
- **Fair Resource Sharing**: Multiple tenants/applications share the same vLLM instance
- **Preventing Starvation**: Unlike strict priority scheduling, WFQ guarantees all requests make progress

## Configuration

### Setting the Scheduling Policy

Configure WFQ by setting the `policy` parameter in `SchedulerConfig`:

```python
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.config.scheduler import SchedulerConfig

# Create scheduler config with WFQ policy
scheduler_config = SchedulerConfig.default_factory(
    policy="wfq",  # Enable Weighted Fair Queuing
    max_num_seqs=128,
    max_num_batched_tokens=2048,
)

# Create VllmConfig with WFQ scheduler
vllm_config = VllmConfig(
    scheduler_config=scheduler_config,
    # ... other config parameters
)

# Initialize LLM with WFQ
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    vllm_config=vllm_config,
)
```

### Using the CLI

For the vLLM API server, use the `--scheduler-policy` flag:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --scheduler-policy wfq
```

## Setting Request Weights

### Python API

Specify the `weight` parameter when creating requests:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    scheduler_policy="wfq",
)

# High-priority request (weight=2.0)
prompts_high = ["Critical user query"]
sampling_params_high = SamplingParams(
    max_tokens=100,
    weight=2.0,  # 2x higher priority
)

# Normal-priority request (weight=1.0, default)
prompts_normal = ["Regular user query"]
sampling_params_normal = SamplingParams(
    max_tokens=100,
    weight=1.0,  # Default weight
)

# Low-priority request (weight=0.5)
prompts_low = ["Background task"]
sampling_params_low = SamplingParams(
    max_tokens=100,
    weight=0.5,  # 0.5x priority
)

# Generate with mixed priorities
outputs_high = llm.generate(prompts_high, sampling_params_high)
outputs_normal = llm.generate(prompts_normal, sampling_params_normal)
outputs_low = llm.generate(prompts_low, sampling_params_low)
```

### OpenAI-Compatible API

When using the OpenAI-compatible API, pass `weight` in the request:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

# High-priority completion
response = client.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="Critical user query",
    max_tokens=100,
    extra_body={"weight": 2.0},  # Custom parameter
)
```

**Note**: The OpenAI client may require using `extra_body` for custom parameters. Alternatively, extend the API to accept `weight` as a first-class parameter.

### Chat Completions

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100,
    extra_body={"weight": 1.5},
)
```

## Weight Semantics

### Weight Values

- **Default**: `weight=1.0` (standard priority)
- **Higher Weight**: `weight > 1.0` (higher priority, scheduled earlier)
- **Lower Weight**: `0 < weight < 1.0` (lower priority, scheduled later)
- **Invalid**: `weight ≤ 0` (automatically reset to default `1.0`)

### Proportional Fairness

WFQ provides proportional fairness, meaning:

- A request with `weight=2.0` receives 2x the resources of a `weight=1.0` request
- A request with `weight=0.5` receives 0.5x (half) the resources of a `weight=1.0` request

### Virtual Time Computation

Internally, WFQ computes:

```text
virtual_start = max(global_virtual_time, request_arrival_time)
virtual_finish = virtual_start + (tokens_needed / weight)
```

Requests are scheduled in order of `virtual_finish` time (earliest first).

## Use Cases and Examples

### Multi-Tenant Serving

Assign different weights to different tenants:

```python
# Premium tier: weight=2.0
tenant_a_requests = generate_with_weight(prompts_a, weight=2.0)

# Standard tier: weight=1.0
tenant_b_requests = generate_with_weight(prompts_b, weight=1.0)

# Free tier: weight=0.5
tenant_c_requests = generate_with_weight(prompts_c, weight=0.5)
```

**Result**: Tenant A gets 2x resources compared to Tenant B, and 4x compared to Tenant C.

### Application-Level Priorities

Different application features have different priorities:

```python
# Real-time chat: weight=3.0 (highest)
chat_completions = generate_with_weight(chat_prompts, weight=3.0)

# Document summarization: weight=1.0 (standard)
summaries = generate_with_weight(doc_prompts, weight=1.0)

# Batch analytics: weight=0.5 (lowest)
analytics = generate_with_weight(analytics_prompts, weight=0.5)
```

### User-Based QoS

Assign weights based on user subscription level:

```python
def get_user_weight(user_id):
    user = get_user(user_id)
    if user.subscription == "enterprise":
        return 3.0
    elif user.subscription == "pro":
        return 2.0
    elif user.subscription == "standard":
        return 1.0
    else:  # free tier
        return 0.5

# Generate with user-specific weight
weight = get_user_weight(current_user_id)
output = llm.generate(prompt, SamplingParams(max_tokens=100, weight=weight))
```

## Fairness Guarantees

### No Starvation

WFQ guarantees that all requests eventually make progress, regardless of weight:

- Low-weight requests are not indefinitely blocked by high-weight requests
- Global virtual time advances monotonically, ensuring fairness over time

### Preemption Handling

When a request is preempted (e.g., due to KV cache eviction):

- WFQ preserves the request's virtual times
- The request resumes without penalty
- Fairness is maintained across preemption events

## Performance Characteristics

| Operation | Time Complexity |
|-----------|----------------|
| Add request | O(log n) |
| Pop request | O(log n) |
| Peek request | O(1) |

Where `n` is the number of requests in the queue.

## Comparison with Other Policies

| Feature | FCFS | Priority | WFQ |
|---------|------|----------|-----|
| Fairness | First-come | Priority order | Proportional by weight |
| Starvation | No | Yes (low priority) | No |
| Complexity | O(1) add/pop | O(log n) add/pop | O(log n) add/pop |
| Use Case | Simple queuing | Strict priorities | Multi-tenant, QoS |

## Best Practices

### 1. Weight Calibration

Start with a small range of weights (e.g., 0.5, 1.0, 2.0) and adjust based on observed behavior:

```python
WEIGHTS = {
    "critical": 2.0,
    "normal": 1.0,
    "background": 0.5,
}
```

### 2. Avoid Extreme Weights

Very large or very small weights can lead to imbalanced scheduling:

- **Avoid**: `weight=1000.0` or `weight=0.001`
- **Prefer**: `weight` in range `[0.1, 10.0]`

### 3. Monitor Fairness

Track metrics to ensure fairness:

```python
# Log virtual finish times
logger.info(f"Request {req_id}: vfinish={req.virtual_finish_time:.2f}")

# Monitor queue waiting times by weight
avg_wait_by_weight = compute_avg_wait_time_by_weight(requests)
```

### 4. Backward Compatibility

If `weight` is not specified, it defaults to `1.0`:

```python
# These are equivalent
SamplingParams(max_tokens=100)
SamplingParams(max_tokens=100, weight=1.0)
```

## Limitations

### 1. Single-Queue Fairness

WFQ provides fairness within a single vLLM instance. For multi-instance deployments, consider:

- Load balancer-level weight routing
- Instance-level quotas

### 2. Prompt Length Sensitivity

WFQ considers total tokens (prompt + output). Very long prompts may affect fairness:

- Long prompt + high weight → scheduled very early
- Short prompt + low weight → may wait longer

### 3. No Time-Based Guarantees

WFQ provides proportional fairness, not latency guarantees:

- Cannot guarantee "request completes in X seconds"
- Fairness is relative to other requests in the queue

## Troubleshooting

### Issue: All requests have same priority

**Symptom**: WFQ behaves like FCFS

**Solution**: Ensure weights are actually different:

```python
# Verify weights are set
assert req1.weight != req2.weight
```

### Issue: Low-weight requests starving

**Symptom**: Low-weight requests never get scheduled

**Cause**: This should not happen with correct WFQ implementation

**Debug**:

```python
# Check virtual times
logger.debug(f"Global VT: {queue._virtual_time}")
logger.debug(f"Request VF: {req.virtual_finish_time}")
```

### Issue: Invalid weight warning

**Symptom**: Logs show "invalid weight, using default"

**Solution**: Ensure `weight > 0`:

```python
# Bad
SamplingParams(weight=0.0)   # Reset to 1.0
SamplingParams(weight=-1.0)  # Reset to 1.0

# Good
SamplingParams(weight=0.5)   # Valid
SamplingParams(weight=1.0)   # Valid
```

## API Reference

### SchedulerConfig

```python
SchedulerConfig(
    policy: Literal["fcfs", "priority", "wfq"] = "fcfs",
    # ... other parameters
)
```

### Request

```python
Request(
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    weight: float = 1.0,  # Default weight
    # ... other parameters
)
```

### SamplingParams

```python
SamplingParams(
    max_tokens: int,
    weight: float = 1.0,  # WFQ weight parameter
    # ... other parameters
)
```

## Further Reading

- [WFQ Algorithm](<https://en.wikipedia.org/wiki/Weighted_fair_queueing>)
- [Generalized Processor Sharing (GPS)](<https://en.wikipedia.org/wiki/Generalized_processor_sharing>)
- vLLM Documentation: Advanced Scheduling

---

**Version**: vLLM v0.x.x

**Status**: Experimental Feature

**Feedback**: Please report issues at <https://github.com/vllm-project/vllm/issues>
