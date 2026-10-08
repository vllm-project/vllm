# Structured Decisions

The `/v1/systemone` endpoint answers a set of typed questions about a state and
returns a probability for every allowed answer.

The endpoint is off unless the server starts with
`--enable-structured-decisions`. With the flag, startup fails for a model that
does not support structured decisions, and with `--logprobs-mode raw_logits` or
`processed_logits`, since `label_mass` needs log probabilities.

## How it works

1. Each question is one read. The user message is the state, followed by the
   question and its options, labeled `A`, `B`, `C`, ... in order, and
   "Answer with the letter of one option only."
2. The read is the reply's first token. Each label must be one token there,
   or the request returns 400.
3. The read returns the logprobs of that question's label tokens. Softmax over
   the labels gives the answer's probabilities.

Thinking is off unless `chat_template_kwargs` turns it on.

This endpoint does not serve diffusion models yet.

Every read of a request starts with the state, so with
`--enable-prefix-caching` the state is prefilled once and each further
question prefills only its own text, then reads one token.

## Question types

| type | criteria | answer |
| --- | --- | --- |
| `choice` | map of option name to a description or `null` | `choice`, `probabilities` by option name, `confidence` |

A choice has at least one option. A question id is a non-empty string.

## Example

```bash
curl -s http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "state": {"ticket": "My card was charged twice for one order."},
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
          "billing": "payments and refunds",
          "shipping": "deliveries",
          "security": "account access"
        }
      }
    },
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Response, with probabilities that depend on the model:

```json
{
  "id": "decision-...",
  "object": "structured_decision",
  "created": 1790000000,
  "model": "Qwen/Qwen3-0.6B",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.97, "shipping": 0.01, "security": 0.02},
      "confidence": 0.90
    }
  },
  "usage": {"input_tokens": 142, "output_tokens": 1},
  "diagnostics": {
    "team": {"label_mass": 0.93, "argmax_is_label": true}
  }
}
```

`diagnostics.label_mass` is the probability the model put on the labels across
its whole vocabulary. A low value means most of the probability went to tokens
that are not labels. `confidence` is the chosen option's probability over the
whole vocabulary: its share of the labels times `label_mass`.

## Labels

Options are labeled `A` to `Z` in the order the request lists them, so a
`choice` has at most 26 options.

## Limits

| limit | value |
| --- | --- |
| questions per request | 64 |
| options per `choice` | 26 |

## Request fields

| field | meaning |
| --- | --- |
| `state` | what the questions are about: a string, or JSON that is sent as its JSON text |
| `questions` | question id to `{type, instructions, criteria}`, asked in this order |
| `instructions` | optional context placed ahead of the questions |
| `chat_template_kwargs` | passed to the chat template, for example `{"enable_thinking": false}` |

A question with any field other than `type`, `instructions` and `criteria` is
rejected with a 400.
