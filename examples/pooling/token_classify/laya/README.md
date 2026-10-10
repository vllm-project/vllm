# Laya typed decisions

[Laya](https://github.com/NandhaKishorM/laya) answers typed questions (`choice`, `score`, `noul`) about a state in one encoder forward pass. vLLM serves it as a `token_classify` model: each question is one prompt, and the output has one row per option marker, `[p_option, *p_action]`. Option probabilities are temperature-calibrated like Laya's own `Agent`. Pass `use_activation: false` to get raw logits.

## Convert a checkpoint

Laya checkpoints keep their config and tokenizer in subfolders, so convert them first:

```bash
python convert_checkpoint.py convaiinnovations/laya ./laya-vllm
python convert_checkpoint.py convaiinnovations/laya-multilingual ./laya-multilingual-vllm
```

## Offline

```bash
python laya_offline.py --model ./laya-vllm
```

## `/v1/systemone` and structured chat

[`structured_server.py --backend laya`](../../../features/structured_diffusion/README.md#laya) serves the Jev `/v1/systemone` API and the schema-in-system-message `/v1/chat/completions` API in front of vLLM's `/pooling` endpoint, the same as for DiffusionGemma:

```bash
vllm serve ./laya-vllm --served-model-name laya
python examples/features/structured_diffusion/structured_server.py --backend laya \
    --upstream http://127.0.0.1:8000 --model laya --tokenizer ./laya-vllm --port 8011

curl http://127.0.0.1:8011/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Hi, we were billed twice for March. Refund the duplicate today or we cancel.",
  "questions": {
    "churn_risk": {"type": "noul", "instructions": "Is this customer likely to churn?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "payments, refunds", "technical": "bugs, outages"}}
  }
}'
```

Laya's `Router`, which picks the English or multilingual checkpoint based on the script of the input, is not included. Run one vLLM server per checkpoint and route between them.
