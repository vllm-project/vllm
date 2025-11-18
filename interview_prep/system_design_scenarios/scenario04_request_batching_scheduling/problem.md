# Scenario 04: Request Batching & Scheduling System

## Problem Statement

Design an intelligent request batching and scheduling system for LLM inference that maximizes throughput while maintaining latency SLAs, handles variable-length sequences, and supports priority-based scheduling.

## Requirements

### Functional Requirements
1. **Dynamic Batching:** Form batches from incoming requests
2. **Continuous Batching:** Add/remove sequences as they complete
3. **Priority Handling:** Support high/medium/low priority requests
4. **Fairness:** Prevent starvation of low-priority requests

### Non-Functional Requirements
1. **Latency:** P99 < 100ms (high priority), < 500ms (low priority)
2. **Throughput:** Maximize tokens/second
3. **Fairness:** Low-priority requests shouldn't wait >30s
4. **Utilization:** GPU utilization > 80%

## Key Challenges
- Variable output lengths (can't predict duration)
- Trade-off between latency and throughput (batch size)
- Priority vs fairness balance
- Batch size optimization

## Difficulty: ★★★★☆ (Hard)
