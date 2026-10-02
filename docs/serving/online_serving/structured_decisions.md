# Structured Decisions

The `/v1/systemone` endpoint answers a set of typed questions about a state and
returns a probability for every allowed answer.

The endpoint is off unless the server starts with
`--enable-structured-decisions`. The endpoint only serves models that support
structured decisions, and other models will return 501.

## How it works

1. Each question is one read. The user message is the state, followed by the
   question and its allowed answers, each with a single-token label such as
   `K`.
2. The assistant reply is prefilled up to the question's label (`id:`), and
   the read generates one token.
3. The read returns the logprobs of that question's label tokens. Softmax over
   the labels gives the answer's probabilities.

This endpoint does not serve diffusion models yet.

Every read of a request starts with the state, so with
`--enable-prefix-caching` the state is prefilled once and each further
question prefills only its own text and answer prefix, then reads one token.
Each read sees only its own question. With every question in one prompt, later
questions lost accuracy: on Qwen3-0.6B, 81% at the first question and 60% at
the fourth.

## Question types

| type | criteria | answer |
| --- | --- | --- |
| `choice` | map of option name to a description or `null` | `choice`, `probabilities` by option name, `confidence` |

A choice has at least one option. A question id is a non-empty string made of any
characters except `:` and newline.

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

At startup the server tries each label from `A` to `ZZ` in the answer format
(`id: label`) and keeps those that are one token. The label's
token can include a leading space or the colon before it. The server groups
the labels by the rest of that token's text and keeps the largest group, so
every label is tokenized the same way. A question takes single letters first
and two-letter labels only when it has more options than letters. Labels are
shuffled with a seed from a hash of the question, so the first option is not
always `A`, and a repeated question gets the same prompt.

## Limits

| limit | value |
| --- | --- |
| questions per request | 64 |
| options per `choice` | the number of labels the server keeps, at most 128 |

## Request fields

| field | meaning |
| --- | --- |
| `state` | what the questions are about: a string, or JSON that is sent as its JSON text |
| `questions` | question id to `{type, instructions, criteria}`, asked in this order |
| `instructions` | optional context placed ahead of the questions |
| `chat_template_kwargs` | passed to the chat template, for example `{"enable_thinking": false}` |
| `seed` | optional, changes each question's label shuffle. Averaging answers over several seeds reduces label bias. |

A question with any field other than `type`, `instructions` and `criteria` is
rejected with a 400.
