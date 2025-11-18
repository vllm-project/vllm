# Scenario 10: Discussion Points - Edge LLM Deployment

## Key Topics

### 1. Model Compression
**Question:** "How do you fit a 7B model on an edge device with 8GB RAM?"
- INT4 quantization: 14GB → 3.5GB
- Pruning: Remove 20% weights
- Distillation: 7B → 3B if needed
- TensorRT optimization

### 2. Edge vs Cloud Decision
**Question:** "When should you use edge vs fallback to cloud?"
- Complexity classification
- Context length limits
- Battery level consideration
- Offline capability requirements

### 3. Power Management
**Question:** "How do you optimize for battery life?"
- Adjust GPU frequency
- Reduce batch size
- Aggressive cloud fallback when low battery
- Sleep mode between requests

### 4. OTA Updates
**Question:** "How do you update models on thousands of devices?"
- Delta compression (80% size reduction)
- Background downloads
- Atomic swap
- Automatic rollback on failure

## Red/Green Flags
**Red:** No compression strategy, can't explain edge/cloud split, ignores power
**Green:** Multi-stage compression (INT4 + pruning + TensorRT), intelligent routing, power-aware, OTA updates
