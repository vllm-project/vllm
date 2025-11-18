# Scenario 09: Discussion Points - A/B Testing Framework

## Key Topics

### 1. Traffic Splitting
**Question:** "How do you assign users to variants consistently?"
- Hash-based randomization (user ID)
- Ensures consistency
- Avoids bias

### 2. Quality Measurement
**Question:** "How do you measure LLM output quality?"
- Human evaluation (slow, expensive, gold standard)
- LLM-as-judge (fast, scalable)
- Task-specific metrics (BLEU, ROUGE)

### 3. Statistical Significance
**Question:** "How do you know if variant is truly better?"
- T-test for latency
- Chi-square for categorical
- p-value < 0.05 threshold
- Confidence intervals

### 4. Safety Mechanisms
**Question:** "What if variant performs poorly?"
- Automatic rollback on error spike
- Gradual rollout (10% → 50% → 100%)
- Continuous monitoring

## Red/Green Flags
**Red:** Random assignment without consistency, no quality metrics, no safety
**Green:** Hash-based assignment, LLM-as-judge, statistical rigor, automatic rollback
