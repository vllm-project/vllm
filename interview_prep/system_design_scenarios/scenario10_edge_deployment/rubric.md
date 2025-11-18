# Scenario 10: Evaluation Rubric

## Scoring (100 points)
- Model Compression (30 pts): INT4, pruning, distillation, TensorRT
- Edge/Cloud Orchestration (25 pts): Complexity routing, fallback logic
- Resource Constraints (20 pts): Memory, power, latency management
- OTA Updates (15 pts): Delta compression, atomic updates, rollback
- Offline Capability (10 pts): Caching, degraded operation

## Level Expectations
**L6 (85-100):** Multi-stage compression, intelligent routing, power management, OTA
**L5 (70-84):** Good compression, basic routing, simple updates
**L4 (55-69):** INT4 quantization, understands constraints

## Key Discriminators
1. Compression strategy (must achieve <10GB)
2. Edge vs cloud decision logic
3. Power management awareness
4. OTA update mechanism

## Device Constraints Understanding
- Memory: 8-16GB
- Power: <30W
- Latency: <500ms P99
- Model size: <10GB on disk
