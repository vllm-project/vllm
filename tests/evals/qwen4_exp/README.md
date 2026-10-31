# Qwen4Exp accuracy evaluation

This suite starts a Qwen3.8-Flash-Next-FP8 OpenAI-compatible server once and
uses EvalScope to evaluate GSM8K and AIME25. A dataset can set
`max_score_trials` to conditionally run another fixed-seed trial when its
initial score is below the configured floor; the test applies the same floor
to the combined score.

```bash
# B200
pytest -s -v tests/evals/qwen4_exp/test_accuracy.py \
  --config-list-file=configs/models-b200.txt

# H200
pytest -s -v tests/evals/qwen4_exp/test_accuracy.py \
  --config-list-file=configs/models-h200.txt
```
