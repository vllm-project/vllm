# Scenario 04: Discussion Points - Request Batching & Scheduling

## Key Topics

### 1. Continuous vs Static Batching
**Question:** "Explain the difference between continuous and static batching."
- Static: Wait for all sequences to complete
- Continuous: Add new sequences as slots free up
- Benefit: 2-3x throughput improvement

### 2. Priority Scheduling
**Question:** "How do you prevent low-priority requests from starving?"
- Fairness mechanism: Boost priority after timeout
- Weighted round-robin
- SLA-based scheduling

### 3. Batch Size Optimization
**Question:** "How do you choose optimal batch size?"
- Trade-off: Latency vs throughput
- Dynamic adjustment based on queue depth
- Consider GPU memory constraints

## Red/Green Flags
**Red:** Doesn't understand continuous batching, no fairness consideration
**Green:** Proposes iteration-level batching, discusses priority with fairness, mentions vLLM
