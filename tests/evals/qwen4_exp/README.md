# Qwen4Exp accuracy evaluation

This suite starts a Qwen3.8-Flash-Next-FP8 OpenAI-compatible server once and
uses EvalScope to evaluate GSM8K and AIME25.

```bash
# B200
pytest -s -v tests/evals/qwen4_exp/test_accuracy.py \
  --config-list-file=configs/models-b200.txt

# H200
pytest -s -v tests/evals/qwen4_exp/test_accuracy.py \
  --config-list-file=configs/models-h200.txt
```
