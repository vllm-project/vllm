# Generative Scoring

The `/generative_scoring` endpoint uses a CausalLM model (e.g., Llama, Qwen, Mistral) to compute the probability of specified token IDs appearing as the next token. Each item (document) is concatenated with the query to form a prompt, and the model predicts how likely each label token is as the next token after that prompt. This lets you score items against a query — for example, asking "Is this the capital of France?" and scoring each city by how likely the model is to answer "Yes".

This endpoint is automatically available when the server is started with a generative model (task `"generate"`). It is separate from the pooling-based [Score API](../../models/pooling_models/scoring.md#score-api), which uses cross-encoder, bi-encoder, or late-interaction models.

**Requirements:**

- The `label_token_ids` parameter is **required** and must contain **at least 1 token ID**.
- When 2 label tokens are provided, the score equals `P(label_token_ids[0]) / (P(label_token_ids[0]) + P(label_token_ids[1]))` (softmax over the two labels).
- When more labels are provided, the score is the softmax-normalized probability of the first label token across all label tokens.

## How it works

1. **Prompt Construction**: For each item, builds `prompt = query + item` (or `item + query` if `item_first=true`)
2. **Forward Pass**: Runs the model on each prompt to get next-token logits
3. **Probability Extraction**: Extracts logprobs for the specified `label_token_ids`
4. **Softmax Normalization**: Applies softmax over only the label tokens (when `apply_softmax=true`)
5. **Score**: Returns the normalized probability of the first label token

## Finding Token IDs

To find the token IDs for your labels, use the tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id = tokenizer.encode("No", add_special_tokens=False)[0]
print(f"Yes: {yes_id}, No: {no_id}")
```

## Example

```bash
curl -X POST http://localhost:8000/generative_scoring \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "query": "Is this city the capital of France?",
    "items": ["Paris", "London", "Berlin"],
    "label_token_ids": [9454, 2753]
  }'
```

Here, each item is appended to the query to form prompts like `"Is this city the capital of France? Paris"`, `"... London"`, etc. The model then predicts the next token, and the score reflects the probability of "Yes" (token 9454) vs "No" (token 2753).

??? console "Response"

    ```json
    {
      "id": "generative-scoring-abc123",
      "object": "list",
      "created": 1234567890,
      "model": "Qwen/Qwen3-0.6B",
      "data": [
        {"index": 0, "object": "score", "score": 0.95},
        {"index": 1, "object": "score", "score": 0.12},
        {"index": 2, "object": "score", "score": 0.08}
      ],
      "usage": {"prompt_tokens": 45, "total_tokens": 48, "completion_tokens": 3}
    }
    ```
