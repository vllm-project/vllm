# Scenario 08: Discussion Points - Cost Optimization

## Key Topics

### 1. Spot Instance Strategy
**Question:** "How do you use spot instances safely?"
- Mix: Reserved baseline + spot burst
- Diversify instance types
- Handle interruptions (2-min warning)
- Cost: 70% savings vs on-demand

### 2. Model Optimization
**Question:** "How do you reduce model costs?"
- Quantization (INT4): 4x memory reduction
- Model routing: right-sized models
- Distillation: smaller models

### 3. Request Optimization
**Question:** "How do you reduce per-request costs?"
- Response caching (30% hit rate)
- Request deduplication
- Aggressive batching

## Red/Green Flags
**Red:** Only considers scaling, ignores spot instances, no cost calculations
**Green:** Multi-tier optimization, spot instance strategy, accurate cost analysis (50%+ reduction)
