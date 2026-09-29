# Structured Decisions

The `/v1/systemone` endpoint answers a set of typed questions about a state and
returns a probability for every allowed answer. The endpoint follows the
request and answer shapes of Jev's decision API.

It is available when the server runs a generative model (task `"generate"`)
with a chat template.

## How it works

1. The system prompt lists every question and its allowed answers, each with a
   single-token label (`A`, `B`, ...). The user message is the state.
2. Each question is one read: the prompt with the assistant reply prefilled up
   to that question's `id:`, and one generated token.
3. The read returns the logprobs of that question's label tokens. Softmax over
   the labels gives the answer's probabilities.

Every read of a request shares the system prompt and the state, so with
`--enable-prefix-caching` the state is prefilled once and each further
question costs about one token.

## Question types

| type | criteria | answer |
| --- | --- | --- |
| `choice` | map of option name to a description or `null` | `choice`, `probabilities` by option name, `confidence` |

A choice has 2 to 26 options. A question id is a non-empty string made of any
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
      "confidence": 0.97
    }
  },
  "usage": {"input_tokens": 142, "output_tokens": 1},
  "diagnostics": {
    "team": {"label_mass": 0.93, "argmax_is_label": true}
  }
}
```

`diagnostics.label_mass` is the probability the model put on the labels across
its whole vocabulary. A low value means the model wanted to reply with
something other than a label, so the answer deserves less trust.

## Request fields

| field | meaning |
| --- | --- |
| `state` | what the questions are about: a string, or JSON that is sent as its JSON text |
| `questions` | question id to `{type, instructions, criteria}`, asked in this order |
| `instructions` | optional context placed ahead of the questions |
| `chat_template_kwargs` | passed to the chat template, for example `{"enable_thinking": false}` |

`depends_on`, `ask_if` and `alone` on a question are rejected with a 400 until
they are implemented.
