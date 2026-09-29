# Structured Decisions

The `/v1/systemone` endpoint answers a set of typed questions about a state and
returns a probability for every allowed answer. The endpoint follows the
request and answer shapes of Jev's decision API.

It is available when the server runs a generative model (task `"generate"`)
with a chat template.

## How it works

1. A decision template renders the system prompt: every question and its
   allowed answers, each with a single-token label (`A`, `B`, ...). The user
   message is the state.
2. Each question is one read: the prompt with the assistant reply prefilled up
   to that question's answer prefix (`id:` by default), and one generated token.
3. The read returns the logprobs of that question's label tokens. Softmax over
   the labels gives the answer's probabilities.

This endpoint does not serve diffusion models yet.

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

## Decision templates

The system prompt comes from a Jinja template, rendered in the same sandboxed
environment as chat templates. The server uses its built-in template unless it
starts with `--decision-template`. That flag takes a file path or the template
inline. A request may send its own `decision_template` when the server runs
with `--trust-request-chat-template`.

The template receives `instructions` (a string or `None`) and `questions`, a
list of objects with `id`, `type`, `instructions` and `options`. Each option has
`label`, `name` and `description`.

A template may define an `answer_prefix(question)` macro: the reply text that
comes right before the question's label. The server prefills each read up to
that text, so a template that asks for a different reply format defines the
matching macro. Without the macro the prefix is `id:`.

```jinja
{% macro answer_prefix(question) %}{{ question.id }} ->{% endmacro %}
Classify the ticket.
{% for q in questions %}
{{ q.instructions }}
{% for o in q.options %}
{{ o.label }} = {{ o.name }}{% if o.description %}: {{ o.description }}{% endif %}

{% endfor %}
{% endfor %}
Reply with one line per question, as: id -> label
```

Every label must be one token after the prefix and a space for the model's
tokenizer. A request whose template breaks that gets a 400 naming the question.

## Limits

| limit | value |
| --- | --- |
| questions per request | 64 |
| options per `choice` | 26, one letter label each |

## Request fields

| field | meaning |
| --- | --- |
| `state` | what the questions are about: a string, or JSON that is sent as its JSON text |
| `questions` | question id to `{type, instructions, criteria}`, asked in this order |
| `instructions` | optional context placed ahead of the questions |
| `decision_template` | a Jinja decision template for this request, used only with `--trust-request-chat-template` |
| `chat_template_kwargs` | passed to the chat template, for example `{"enable_thinking": false}` |

A question with any field other than `type`, `instructions` and `criteria` is
rejected with a 400.
