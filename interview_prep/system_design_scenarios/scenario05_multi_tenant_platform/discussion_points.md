# Scenario 05: Discussion Points - Multi-Tenant Platform

## Key Topics

### 1. Resource Isolation
**Question:** "How do you ensure one tenant doesn't affect another's performance?"
- Dedicated vs shared resources
- GPU quotas and limits
- CPU/memory isolation (cgroups)

### 2. SLA Enforcement
**Question:** "How do you guarantee different SLAs for different tiers?"
- Priority-based scheduling
- Dedicated capacity for Gold tier
- Queue management per tier

### 3. Cost Attribution
**Question:** "How do you accurately bill each tenant?"
- GPU time tracking
- Token-level metering
- Request counting
- Fair allocation of shared costs

## Red/Green Flags
**Red:** No isolation strategy, can't explain cost attribution
**Green:** Multi-level isolation, detailed billing model, LoRA for custom models
