# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured decisions in front of a vLLM DiffusionGemma server.

POST /v1/systemone takes Jev's request body: {"model", "state", "questions"}.
"questions" maps an id to {"type", "instructions", "criteria"}, where "type"
is "noul", "choice" or "score" and the criteria shape follows the type:
  noul:   optional {"true": ..., "false": ...} descriptions
  choice: option name -> description or null
  score:  ordered list of levels
Answers take Jev's shapes, with this server's diagnostics alongside:
  noul:   {"noul": p}
  choice: {"choice", "probabilities", "confidence"}
  score:  {"score", "legend", "probabilities", "confidence"}
The request body may also carry the schema keys "instructions", "samples",
"auto_max", "auto_threshold", "steps", "think", "ask", "chunk_rows",
"chunk_prompt" and "sequential" as extensions. Images go ahead of the
state, either as multipart/form-data with the JSON body in a part named
"request" and each image as a file part, or as an "images" array of data
URLs in the JSON body.

POST /v1/chat/completions makes the same decision from an OpenAI-shaped
call. The system message is the schema JSON below and the user message is
the state. The reply's `content` is the JSON answer set.

POST /v1/raw/chat/completions passes the body to vLLM's chat completions
unchanged, for plain generation through this port.

With API_KEY set in the environment, every POST needs "Authorization:
Bearer <key>". --tls-port adds an HTTPS listener with a self-signed
certificate kept in --cert-dir, for clients that need a secure origin.

Each answer is one distribution per question, from one denoise step over a
seeded canvas, averaged over a few noise draws. This server handles the
canvas, tokenizer, slot resolution, noise draws and averaging.

Schema (system message):
  {"questions": [
     {"id": "urgent", "type": "noul", "instructions": "..."},
     {"id": "bucket", "type": "choice", "instructions": "...",
      "options": [{"name": "billing", "description": "..."}, ...]},
     {"id": "tone", "type": "score", "instructions": "...",
      "levels": ["calm", "annoyed", "furious"]}],
   "instructions": "optional context",
   "samples": "auto" | N, "auto_threshold": 0.1, "auto_max": 4,
   "steps": 1, "think": 0}

A question may also declare:
  "depends_on": [ids]        answered after those, with their answers in
                             its prompt
  "ask_if": {id: [answers]}  asked only when that question's answer is
                             among them (a skipped answer is null)
  "alone": true              a read of its own
Questions run in stages by these dependencies. Each stage is one joint
read. Later stages continue the earlier answers, prefilled for a text
state and restated for an image.

Up to ten questions answer as "id: label" lines. Past that the id runs
straight into the label, space separated, one row fewer per question. The
server splits a schema whose answer template does not fit the canvas into
chunks that run together, each with its own question list. "chunk_rows"
sets the rows per chunk, "ask" picks a subset of question ids for one
read, "sequential": true runs the chunks in order with the earlier answers
prefilled, and "chunk_prompt": "shared" lists every question in each
chunk's prompt. "think": N lets the model write up to N tokens in its
thought channel, as an ordinary generation, and the read then runs with
that thought in its prompt. With images the model writes the thought with
the image in view and the server seeds it into the canvas ahead of the
answer, so the canvas bounds it. The noise draws of a decision share one
thought.

Serve the model with a canvas that holds the answer template, for example:
  vllm serve google/diffusiongemma-26B-A4B-it \
      --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 \
      --enable-prefix-caching
then run this in front of it:
  python structured_server.py --upstream http://127.0.0.1:8000 \
      --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64 --port 8011
"""

import argparse
import json
import math
import os
import random
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pybase64 as base64
from transformers import AutoTokenizer

ARGS = None
TOK = None
API_KEY = os.environ.get(
    "API_KEY", ""
)  # when set, POST routes need "Authorization: Bearer <key>"
CANVAS_LEN = 64  # the served canvas length. A request may be narrower.
CANVAS_STEP = 16  # request widths are multiples of this
VOCAB = 262144
TURN_CLOSE = 106
PAD = 0
TOPK = 20
MAX_QUESTIONS = 64  # per request
MAX_SAMPLES = 32  # reads per question, fixed or auto
MAX_PARALLEL = 16  # question groups read at once
# the empty thought block the chat template leaves to the model
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"
SCAFFOLD = None
THOUGHT_OPEN = None
THOUGHT_CLOSE = None


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class SchemaError(ValueError):
    pass


def parse_schema(value):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("questions"), list)
        or not value["questions"]
    ):
        raise SchemaError("schema: needs a non-empty questions array")
    if len(value["questions"]) > MAX_QUESTIONS:
        raise SchemaError(f"schema: at most {MAX_QUESTIONS} questions")
    qs = []
    seen = set()
    for q in value["questions"]:
        qid = str(q.get("id", "")).strip()
        if not qid or ":" in qid or "\n" in qid:
            raise SchemaError(
                f"question id {qid!r} must be non-empty, no ':' or newline"
            )
        if qid in seen:
            raise SchemaError(f"duplicate question id {qid!r}")
        seen.add(qid)
        kind = q.get("type")
        if kind in ("noul", "bool", "boolean"):
            kind = "noul"
            crit = q.get("criteria") or {}
            choices = [("yes", crit.get("true")), ("no", crit.get("false"))]
            labels = ["yes", "no"]
        elif kind == "choice":
            opts = q.get("options") or []
            choices = [
                (o["name"], o.get("description"))
                if isinstance(o, dict)
                else (str(o), None)
                for o in opts
            ]
            labels = [chr(ord("A") + i) for i in range(len(choices))]
        elif kind == "score":
            choices = [(str(level), None) for level in (q.get("levels") or [])]
            labels = (
                [str(i + 1) for i in range(len(choices))]
                if len(choices) <= 9
                else [chr(ord("A") + i) for i in range(len(choices))]
            )
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        if len(choices) < 2:
            raise SchemaError(f"question {qid!r}: needs at least two alternatives")
        if len(choices) > 26:
            raise SchemaError(f"question {qid!r}: at most 26 alternatives")
        deps = q.get("depends_on") or []
        ask_if = q.get("ask_if") or {}
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise SchemaError(
                f"question {qid!r}: depends_on must be a list of question ids"
            )
        if not isinstance(ask_if, dict) or not all(
            isinstance(v, list) and v for v in ask_if.values()
        ):
            raise SchemaError(
                f"question {qid!r}: ask_if must map a question id to a "
                "non-empty list of its answers"
            )
        qs.append(
            {
                "id": qid,
                "type": kind,
                "instructions": str(q.get("instructions", "")),
                "choices": choices,
                "labels": labels,
                "depends_on": list(dict.fromkeys(list(deps) + list(ask_if))),
                "ask_if": ask_if,
                "alone": bool(q.get("alone", False)),
            }
        )
    by_id = {q["id"]: q for q in qs}
    for q in qs:
        for dep in q["depends_on"]:
            if dep not in by_id or dep == q["id"]:
                raise SchemaError(
                    f"question {q['id']!r}: depends on unknown question {dep!r}"
                )
        for dep, vals in q["ask_if"].items():
            names = [c[0] for c in by_id[dep]["choices"]]
            if any(v not in names for v in vals):
                raise SchemaError(
                    f"question {q['id']!r}: ask_if values for {dep!r} "
                    f"must be among {names}"
                )
    schedule(qs)  # refuses a cycle
    samples = value.get("samples", "auto")
    if samples == "auto":
        policy = {
            "mode": "auto",
            "max": max(1, min(int(value.get("auto_max", 4)), MAX_SAMPLES)),
            "threshold": float(value.get("auto_threshold", 0.1)),
        }
    elif isinstance(samples, int) and samples >= 1:
        policy = {"mode": "fixed", "n": min(samples, MAX_SAMPLES)}
    else:
        raise SchemaError('schema: samples must be a positive count or "auto"')
    ask = value.get("ask")
    if ask is not None:
        if not isinstance(ask, list) or not ask or any(a not in seen for a in ask):
            raise SchemaError("schema: ask must list question ids from this schema")
        for q in qs:
            if q["id"] in ask and any(d not in ask for d in q["depends_on"]):
                raise SchemaError(
                    f"schema: ask names {q['id']!r} but not everything it depends on"
                )
    chunk_rows = value.get("chunk_rows")
    if chunk_rows is not None and (not isinstance(chunk_rows, int) or chunk_rows < 8):
        raise SchemaError("schema: chunk_rows must be an integer of at least 8")
    chunk_prompt = value.get("chunk_prompt", "own")
    if chunk_prompt not in ("shared", "own"):
        raise SchemaError('schema: chunk_prompt must be "shared" or "own"')
    sequential = bool(value.get("sequential", False))
    think = value.get("think", 0)
    if isinstance(think, bool) or not isinstance(think, int) or not 0 <= think <= 4096:
        raise SchemaError("schema: think must be a thought budget in tokens, 0 to 4096")
    return {
        "questions": qs,
        "instructions": value.get("instructions"),
        "policy": policy,
        "steps": max(1, min(int(value.get("steps", 1)), 8)),
        "think": think,
        "ask": ask,
        "chunk_rows": chunk_rows,
        "chunk_prompt": chunk_prompt,
        "sequential": sequential,
        "format": "lines" if len(qs) <= 10 else "indexed",
    }


# Answer template shape: (join between questions, what precedes the label,
# reply instruction). A small schema gets "lines", which is readable.
# "indexed" ("0yes 1no") costs three tokens a question against four or five
# and agreed with "lines" on every set tried: 42 booleans, ten 26-way
# choices, twenty 5-level scores. Past ten questions the saved rows keep a
# schema in one read. Two tokens a question, or no id at all, loses
# alignment beyond about twenty questions, because the id ties a label to
# its question.
FORMATS = {
    "lines": (
        "\n",
        "{id}: ",
        'Reply with one line per question, in this order, formatted as "id: label".',
    ),
    "indexed": (
        " ",
        "{id}",
        "Reply on one line with each question's id immediately followed by its "
        "label, separated by single spaces.",
    ),
}


def system_text(schema, chunked=False):
    s = (
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n"
    )
    if schema.get("instructions"):
        s += "\n" + str(schema["instructions"]).strip() + "\n"
    for q in schema["questions"]:
        s += f"\nQuestion {q['id']}: {q['instructions'].strip()}\n"
        for (name, desc), label in zip(q["choices"], q["labels"]):
            if q["type"] == "noul":
                s += f"  {label}: {str(desc).strip()}\n" if desc else f"  {label}\n"
            elif desc:
                s += f"  {label}: {name} ({str(desc).strip()})\n"
            else:
                s += f"  {label}: {name}\n"
    s += "\n" + FORMATS[schema.get("format", "lines")][2]
    if chunked:
        s += (
            " A reply may cover only some of the questions; answer every line "
            "that is present."
        )
    return s


def answer_text(qs, labels, fmt="lines"):
    join, lead, _ = FORMATS[fmt]
    return join.join(
        lead.format(id=q["id"]) + q["labels"][i] for q, i in zip(qs, labels)
    )


def enc(text):
    return TOK.encode(text, add_special_tokens=False)


def init_tokenizer(tok):
    global TOK, SCAFFOLD, THOUGHT_OPEN, THOUGHT_CLOSE
    TOK = tok
    THOUGHT_OPEN = enc("<|channel>thought\n")
    THOUGHT_CLOSE = enc("<channel|>")
    SCAFFOLD = enc(SCAFFOLD_TEXT)
    assert THOUGHT_OPEN + THOUGHT_CLOSE == SCAFFOLD, (
        "the thought tags must tokenize apart"
    )


def resolve_template(qs, head, lead, fmt):
    """Tokenize the answer template and find each question's slot. Every label
    must change exactly one token, at the same position for all of a question's
    labels, or this raises SchemaError. ``head`` is the token run the canvas
    starts with: the empty thought block for a plain read, and empty when the
    prompt already ends the thought channel. ``lead`` is the text before the
    first answer: the join when earlier answers are in the prompt, so the
    tokens match one joint template."""
    base_labels = [0] * len(qs)
    base = head + enc(lead + answer_text(qs, base_labels, fmt))
    if len(base) + 1 > CANVAS_LEN:
        raise SchemaError(
            f"answer template is {len(base)} tokens; the canvas holds {CANVAS_LEN - 1}"
        )
    if len(qs) == 1 and len(base) + 1 > CANVAS_LEN:
        raise SchemaError(
            f"question {qs[0]['id']!r} alone needs {len(base) + 1} canvas rows"
        )
    slots = []
    for qi, q in enumerate(qs):
        pos = None
        ids = [0] * len(q["labels"])
        for li in range(1, len(q["labels"])):
            labels = list(base_labels)
            labels[qi] = li
            e = head + enc(lead + answer_text(qs, labels, fmt))
            if len(e) != len(base):
                raise SchemaError(
                    f"question {q['id']!r}: label {q['labels'][li]!r} is not a "
                    "single token"
                )
            diffs = [i for i in range(len(e)) if e[i] != base[i]]
            if len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                raise SchemaError(
                    f"question {q['id']!r}: labels do not share one template slot"
                )
            pos = diffs[0]
            ids[li] = e[pos]
        ids[0] = base[pos]
        if len(set(ids)) != len(ids):
            raise SchemaError(
                f"question {q['id']!r}: two labels tokenize to the same id"
            )
        slots.append({"pos": pos, "label_ids": ids})
    return base, slots


_template_cache = {}


def template_for(schema, head, lead):
    fmt = schema.get("format", "lines")
    key = json.dumps(
        [head, lead, fmt] + [(q["id"], q["labels"]) for q in schema["questions"]]
    )
    if key not in _template_cache:
        _template_cache[key] = resolve_template(schema["questions"], head, lead, fmt)
    return _template_cache[key]


# ----------------------------------------------------------------------------
# Reads
# ----------------------------------------------------------------------------


def canvas_width(template):
    """Smallest multiple of CANVAS_STEP that holds the template and the turn close."""
    need = len(template) + 1
    return min(CANVAS_LEN, -(-need // CANVAS_STEP) * CANVAS_STEP)


def pin_xargs(template, slots, steps):
    """Past one denoise step the template must be held, or accept/renoise
    rewrites it: pin every canvas position that is not an answer slot."""
    if steps <= 1:
        return {}
    free = {s["pos"] for s in slots}
    return {
        "diffusion_pinned": [p for p in range(canvas_width(template)) if p not in free]
    }


def build_canvas(template, slots, seed):
    rng = random.Random(seed)
    canvas = list(template) + [TURN_CLOSE]
    canvas += [PAD] * (canvas_width(template) - len(canvas))
    for s in slots:
        canvas[s["pos"]] = rng.randrange(VOCAB)
    return canvas


def label_id_union(slots):
    ids = sorted({i for s in slots for i in s["label_ids"]})
    return ids[:128]  # vLLM's cap per request. A schema needs far fewer.


def upstream_chat(body, timeout=600):
    req = urllib.request.Request(
        ARGS.upstream.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def upstream_completions(body, timeout=600):
    req = urllib.request.Request(
        ARGS.upstream.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chat_prompt_ids(sys_text, state_text, thinking=False):
    """The prompt the chat endpoint would build, as token ids, ending after
    the model turn marker. Text states only. ``thinking`` turns the chat
    template's thinking marker on."""
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": state_text},
    ]
    out = TOK.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )
    ids = (
        out["input_ids"] if hasattr(out, "keys") else out
    )  # newer transformers return a dict
    return [int(t) for t in ids]


def think(sys_text, state_text, budget):
    """A read prefix that ends with a thought the model wrote: the chat prompt
    with thinking on, the open tag, up to ``budget`` generated tokens, the
    close tag. Returns the prefix and a diagnostics dict for the thought."""
    prompt = chat_prompt_ids(sys_text, state_text, thinking=True) + THOUGHT_OPEN
    started = time.time()
    d = upstream_completions(
        {
            "model": ARGS.model,
            "prompt": prompt,
            "max_tokens": budget,
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
            "stop_token_ids": THOUGHT_CLOSE,
        }
    )
    ids = [int(t.split(":")[1]) for t in d["choices"][0]["logprobs"]["tokens"]]
    closed = THOUGHT_CLOSE[0] in ids
    if closed:
        ids = ids[: ids.index(THOUGHT_CLOSE[0])]
    info = {
        "tokens": len(ids),
        "closed": closed,
        "ms": (time.time() - started) * 1e3,
        "text": TOK.decode(ids),
    }
    return prompt + ids + THOUGHT_CLOSE, info


def think_chat(sys_text, state_content, budget):
    """A thought written with the image in view: the chat endpoint with
    thinking on, capped at ``budget`` tokens and cut at the close tag.
    Returns the thought's token ids and a diagnostics dict."""
    body = {
        "model": ARGS.model,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": state_content},
        ],
        "max_tokens": budget,
        "logprobs": True,
        "top_logprobs": 0,
        "return_tokens_as_token_ids": True,
        "stop_token_ids": THOUGHT_CLOSE,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    started = time.time()
    d = upstream_chat(body)
    choice = d["choices"][0]
    ids = [
        int(t["token"].split(":")[1])
        for t in (choice.get("logprobs") or {}).get("content") or []
    ]
    if ids[: len(THOUGHT_OPEN)] == THOUGHT_OPEN:  # the model opens the channel itself
        ids = ids[len(THOUGHT_OPEN) :]
    closed = THOUGHT_CLOSE[0] in ids
    if closed:
        ids = ids[: ids.index(THOUGHT_CLOSE[0])]
    return ids, {
        "tokens": len(ids),
        "closed": closed,
        "ms": (time.time() - started) * 1e3,
        "text": TOK.decode(ids),
    }


def one_read(
    schema, template, slots, sys_text, state_content, seed, prefix=None, thinking=False
):
    if prefix is not None:
        return one_read_continuation(schema, template, slots, prefix, seed)
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": state_content},
    ]
    body = {
        "model": ARGS.model,
        "messages": messages,
        "max_tokens": len(template) + 1,
        "logprobs": True,
        "top_logprobs": TOPK,
        # Exact logprobs for every label at every position. With a long
        # option list most labels never rank in the top-k, and the model's
        # mass sits on tokens that spell the option name instead.
        "logprob_token_ids": label_id_union(slots),
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "vllm_xargs": {
            "diffusion_seed_canvas": build_canvas(template, slots, seed),
            "diffusion_canvas_length": canvas_width(template),
            "diffusion_max_steps": schema["steps"],
            "diffusion_read_only": True,
            **pin_xargs(template, slots, schema["steps"]),
        },
    }
    d = upstream_chat(body)
    content = d["choices"][0]["logprobs"]["content"]
    out = []
    for q, s in zip(schema["questions"], slots):
        top = {
            int(t["token"].split(":")[1]): t["logprob"]
            for t in content[s["pos"]]["top_logprobs"]
        }
        out.append(slot_distribution(top, s["label_ids"]))
    return out, d.get("usage", {})


def slot_distribution(top, label_ids):
    """Label probabilities at one slot from the returned logprobs: every
    label's own value plus the argmax token. Read-only logprobs are at
    temperature 1, so the label softmax uses them directly. The entropy is
    over that returned set."""
    floor = min(top.values()) - 5.0
    lp_t = [top.get(i, floor) for i in label_ids]
    mx = max(lp_t)
    ex = [math.exp(x - mx) for x in lp_t]
    probs = [e / sum(ex) for e in ex]
    top_p = [math.exp(v) for v in top.values()]
    return {
        "probs": probs,
        "label_mass": sum(math.exp(x) for x in lp_t),
        "entropy": -sum(p * math.log(p) for p in top_p if p > 0),
        "argmax_is_label": max(top, key=top.get) in label_ids,
    }


def one_read_continuation(schema, template, slots, prompt_ids, seed):
    """A read whose prompt already holds the thought scaffold and earlier
    answer lines, sent as token ids so the chat template cannot alter it."""
    body = {
        "model": ARGS.model,
        "prompt": prompt_ids,
        "max_tokens": len(template) + 1,
        "logprobs": TOPK,
        "logprob_token_ids": label_id_union(slots),
        "return_tokens_as_token_ids": True,
        "vllm_xargs": {
            "diffusion_seed_canvas": build_canvas(template, slots, seed),
            "diffusion_canvas_length": canvas_width(template),
            "diffusion_max_steps": schema["steps"],
            "diffusion_read_only": True,
            **pin_xargs(template, slots, schema["steps"]),
        },
    }
    d = upstream_completions(body)
    rows = d["choices"][0]["logprobs"]["top_logprobs"]
    out = []
    for q, sl in zip(schema["questions"], slots):
        top = {int(k.split(":")[1]): v for k, v in rows[sl["pos"]].items()}
        out.append(slot_distribution(top, sl["label_ids"]))
    return out, d.get("usage", {})


def read_many(
    schema,
    template,
    slots,
    sys_text,
    state_content,
    seed,
    n,
    prefix=None,
    thinking=False,
):
    results = [None] * n
    errors = [None] * n
    usages = [None] * n

    def run(k):
        try:
            results[k], usages[k] = one_read(
                schema,
                template,
                slots,
                sys_text,
                state_content,
                seed + k * 7919,
                prefix,
                thinking,
            )
        except Exception as e:  # raised again below
            errors[k] = e

    threads = [threading.Thread(target=run, args=(k,)) for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errors:
        if e is not None:
            raise e
    return results, usages


def schedule(qs):
    """Questions in stages: a question's stage comes after the stages of
    everything it depends on. Declaration order is kept within a stage."""
    ids = {q["id"] for q in qs}
    pending = list(qs)
    done: set = set()
    levels = []
    while pending:
        level = [
            q
            for q in pending
            if all(d in done or d not in ids for d in q["depends_on"])
        ]
        if not level:
            raise SchemaError(
                "schema: dependency cycle among " + ", ".join(q["id"] for q in pending)
            )
        levels.append(level)
        done |= {q["id"] for q in level}
        pending = [q for q in pending if q["id"] not in done]
    return levels


def chunk_groups(schema, qs):
    """``qs`` split, in order, into the fewest groups whose answer templates
    fit ``chunk_rows`` (the canvas by default). A question marked alone gets
    its own group."""
    limit = schema.get("chunk_rows") or CANVAS_LEN
    groups, group = [], []
    for q in qs:
        if q["alone"]:
            if group:
                groups.append(group)
                group = []
            groups.append([q])
            continue
        trial = group + [q]
        rows = (
            len(SCAFFOLD)
            + len(
                enc(answer_text(trial, [0] * len(trial), schema.get("format", "lines")))
            )
            + 1
        )
        if rows > limit and group:
            groups.append(group)
            group = [q]
        else:
            group = trial
    if group:
        groups.append(group)
    return groups


def answer_name(q, a):
    """The answer as the name ask_if compares against: yes or no, an option
    name, or a level name."""
    if a is None:
        return None
    return a["label"] if q["type"] == "noul" else a.get("choice", a.get("level"))


def decide(schema, state_content, seed):
    """One decision. Questions run in stages by their dependencies. A stage
    is one joint read, chunked by the canvas, with a question marked alone
    in its own read. Later stages condition on every earlier answer: a
    prefilled continuation for a text state, or the answers restated in the
    state for an image. A question whose ask_if condition failed is skipped
    and its answer is null."""
    started = time.time()
    qs = [
        q
        for q in schema["questions"]
        if not schema.get("ask") or q["id"] in schema["ask"]
    ]
    levels = schedule(qs)
    text_state = isinstance(state_content, str)
    fmt = schema["format"]
    join = FORMATS[fmt][0]
    # More than one read in sequence needs the full question list in every
    # prompt, so that later reads continue one answer.
    chained = len(levels) > 1 or schema["sequential"]
    sys_full = system_text(schema, chunked=not text_state and chained)
    base_ids, thought = None, None
    if chained and text_state:
        if schema["think"]:
            base_ids, thought = think(sys_full, state_content, schema["think"])
        else:
            base_ids = chat_prompt_ids(sys_full, state_content) + SCAFFOLD
    shared = schema["chunk_prompt"] == "shared"
    answered, lines, earlier = {}, [], []
    parts, stages, chunks, skipped = [], [], [], {}
    by_id = {q["id"]: q for q in schema["questions"]}

    def run(group, k, conditioned):
        sub = dict(schema, questions=group)
        # The thought is written once: in the prefix of a chained text decision,
        # or in the first read of anything else.
        sub["think"] = 0 if (chained and text_state) or conditioned else schema["think"]
        if text_state:
            if conditioned:
                prefix, lead, sys_text = (
                    base_ids + enc(join.join(lines)),
                    join,
                    sys_full,
                )
            elif chained:
                prefix, lead, sys_text = (base_ids if thought else None), "", sys_full
            else:
                prefix, lead, sys_text = (
                    None,
                    "",
                    (system_text(schema, chunked=True) if shared else system_text(sub)),
                )
            return decide_group(
                sub, sys_text, state_content, seed + 104729 * k, prefix, lead
            )
        state = state_content
        if conditioned:
            text = next(
                (p["text"] for p in state_content if p.get("type") == "text"), ""
            )
            text += "\n\nAnswers so far:\n" + "\n".join(earlier)
            state = [p for p in state_content if p.get("type") != "text"] + [
                {"type": "text", "text": text}
            ]
            sys_text = sys_full
        else:
            sys_text = sys_full if (chained or shared) else system_text(sub)
        return decide_group(sub, sys_text, state, seed + 104729 * k)

    def absorb(group, body, rows):
        parts.append((body, rows))
        chunks.append([q["id"] for q in group])
        answered.update(body["answers"])
        lines.append(
            answer_text(
                group,
                [q["labels"].index(body["answers"][q["id"]]["label"]) for q in group],
                fmt,
            )
        )
        earlier.extend(
            f"{q['id']}: {answer_name(q, body['answers'][q['id']])}" for q in group
        )

    k = 0
    for level in levels:
        asked = []
        for q in level:
            failed = next(
                (
                    (dep, vals)
                    for dep, vals in q["ask_if"].items()
                    if answer_name(by_id[dep], answered.get(dep)) not in vals
                ),
                None,
            )
            if failed:
                answered[q["id"]] = None
                skipped[q["id"]] = {
                    "because": failed[0],
                    "was": answer_name(by_id[failed[0]], answered.get(failed[0])),
                    "wanted": failed[1],
                }
                continue
            asked.append(q)
        if not asked:
            continue
        stages.append([q["id"] for q in asked])
        groups = chunk_groups(schema, asked)
        conditioned = bool(lines)
        if schema["sequential"] or len(groups) == 1:
            for group in groups:
                body, rows = run(
                    group, k, conditioned or (schema["sequential"] and bool(lines))
                )
                absorb(group, body, rows)
                k += 1
        else:
            with ThreadPoolExecutor(max_workers=min(len(groups), MAX_PARALLEL)) as ex:
                results = list(
                    ex.map(
                        lambda gk, conditioned=conditioned: run(
                            gk[1], gk[0], conditioned
                        ),
                        [(k + i, g) for i, g in enumerate(groups)],
                    )
                )
            for group, (body, rows) in zip(groups, results):
                absorb(group, body, rows)
            k += len(groups)

    answers = {q["id"]: answered.get(q["id"]) for q in qs}
    diag_q = {}
    for body, _ in parts:
        diag_q.update(body["diagnostics"]["questions"])
    # A thought written here (a chained text decision) is outside every
    # group's row count. One written inside a group is already counted there.
    extra_rows = thought["tokens"] if thought else 0
    if thought is None:
        thoughts = [b["diagnostics"].get("thought") for b, _ in parts]
        thought = (
            thoughts[0] if len(parts) == 1 else ([t for t in thoughts if t] or None)
        )
    one = len(parts) == 1 and not skipped
    diagnostics = {
        "steps": schema["steps"],
        "stages": stages,
        "skipped": skipped,
        "chunks": chunks,
        "chunk_prompt": "full" if chained else schema["chunk_prompt"],
        "sequential": schema["sequential"],
        "conditioning": (
            None
            if len(stages) <= 1 and not schema["sequential"]
            else ("prefill" if text_state else "restated")
        ),
        "thought": thought,
        "samples": (
            parts[0][0]["diagnostics"]["samples"]
            if one
            else {
                "n": [b["diagnostics"]["samples"]["n"] for b, _ in parts],
                "tops": [b["diagnostics"]["samples"]["tops"] for b, _ in parts],
                "policy": [b["diagnostics"]["samples"]["policy"] for b, _ in parts],
            }
        ),
        "timing": {
            "total_ms": (time.time() - started) * 1e3,
            "reads": sum(b["diagnostics"]["timing"]["reads"] for b, _ in parts),
        },
        "prompt_tokens": max(
            (b["diagnostics"].get("prompt_tokens") or 0) for b, _ in parts
        )
        or None,
        "questions": diag_q,
        "engine": "vllm",
    }
    return {"answers": answers, "diagnostics": diagnostics}, sum(
        rows for _, rows in parts
    ) + extra_rows


def decide_group(schema, sys_text, state_content, seed, prefix=None, lead=""):
    started = time.time()
    thought = None
    head = SCAFFOLD if prefix is None else []
    thinking = False
    if prefix is None and schema["think"]:
        if isinstance(state_content, str):
            prefix, thought = think(sys_text, state_content, schema["think"])
            head = []
        else:
            # With images the model writes the thought with the image in view, and
            # the server seeds it into the canvas ahead of the answer, so the read
            # keeps the image. The canvas bounds the thought.
            answer = enc(
                answer_text(
                    schema["questions"],
                    [0] * len(schema["questions"]),
                    schema.get("format", "lines"),
                )
            )
            fits = CANVAS_LEN - 1 - len(THOUGHT_OPEN) - len(THOUGHT_CLOSE) - len(answer)
            if fits < 8:
                raise SchemaError(
                    f"think: the canvas leaves {fits} rows for a thought "
                    "beside this template"
                )
            ids, thought = think_chat(
                sys_text, state_content, min(schema["think"], fits)
            )
            thought["budget"] = min(schema["think"], fits)
            head = THOUGHT_OPEN + ids + THOUGHT_CLOSE
            thinking = True
    template, slots = template_for(schema, head, lead)
    policy = schema["policy"]
    if policy["mode"] == "fixed":
        reads, usages = read_many(
            schema,
            template,
            slots,
            sys_text,
            state_content,
            seed,
            policy["n"],
            prefix,
            thinking,
        )
        extended = None
        first_entropy = None
    else:
        reads, usages = read_many(
            schema, template, slots, sys_text, state_content, seed, 1, prefix, thinking
        )
        first_entropy = {
            q["id"]: r["entropy"] for q, r in zip(schema["questions"], reads[0])
        }
        extended = (
            max(first_entropy.values()) > policy["threshold"] and policy["max"] > 1
        )
        if extended:
            more, more_usages = read_many(
                schema,
                template,
                slots,
                sys_text,
                state_content,
                seed + 1,
                policy["max"] - 1,
                prefix,
                thinking,
            )
            reads += more
            usages += more_usages
    prompt_tokens = next(
        (u["prompt_tokens"] for u in usages if u and u.get("prompt_tokens")), None
    )
    elapsed_ms = (time.time() - started) * 1e3

    answers = {}
    diag_q = {}
    n = len(reads)
    for qi, q in enumerate(schema["questions"]):
        per = [r[qi]["probs"] for r in reads]
        mean = [sum(p[i] for p in per) / n for i in range(len(q["labels"]))]
        top = max(range(len(mean)), key=lambda i: mean[i])
        a = {
            "type": q["type"],
            "label": q["labels"][top],
            "confidence": mean[top],
            "probabilities": {c[0]: m for c, m in zip(q["choices"], mean)},
        }
        if q["type"] == "noul":
            a["noul"] = mean[0]
        elif q["type"] == "choice":
            a["choice"] = q["choices"][top][0]
        else:
            a["score"] = sum((i + 1) * m for i, m in enumerate(mean))
            a["level"] = q["choices"][top][0]
        if n > 1:
            var = sum((p[top] - mean[top]) ** 2 for p in per) / (n - 1)
            a["stderr"] = (var / n) ** 0.5
            a["agreement"] = (
                sum(1 for p in per if max(range(len(p)), key=lambda i: p[i]) == top) / n
            )
        answers[q["id"]] = a
        diag_q[q["id"]] = {
            "pos": slots[qi]["pos"],
            "entropy": [r[qi]["entropy"] for r in reads],
            "label_mass": reads[0][qi]["label_mass"],
            "argmax_is_label": reads[0][qi]["argmax_is_label"],
        }
    tops = [
        {
            q["id"]: [
                q["labels"][
                    max(range(len(r[qi]["probs"])), key=lambda i: r[qi]["probs"][i])
                ],
                max(r[qi]["probs"]),
                r[qi]["entropy"],
            ]
            for qi, q in enumerate(schema["questions"])
        }
        for r in reads
    ]
    return {
        "answers": answers,
        "diagnostics": {
            "steps": schema["steps"],
            "samples": {
                "n": n,
                "tops": tops,
                "policy": dict(
                    policy, extended=extended, first_read_entropy=first_entropy
                ),
            },
            "timing": {"total_ms": elapsed_ms, "reads": n},
            "thought": thought,
            "prompt_tokens": prompt_tokens,
            "questions": diag_q,
            "engine": "vllm",
        },
    }, len(template) + 1 + (thought["tokens"] if thought else 0)


# ----------------------------------------------------------------------------
# Jev's contract
# ----------------------------------------------------------------------------

JEV_EXTENSIONS = (
    "instructions",
    "samples",
    "auto_max",
    "auto_threshold",
    "steps",
    "think",
    "ask",
    "chunk_rows",
    "chunk_prompt",
    "sequential",
)


def jev_schema(body):
    """This server's schema from a Jev request body."""
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise SchemaError("questions: needs a non-empty map of id -> question")
    out = []
    for qid, q in qs.items():
        if not isinstance(q, dict):
            raise SchemaError(f"question {qid!r}: must be an object")
        kind, crit, ins = q.get("type"), q.get("criteria"), q.get("instructions", "")
        item = {
            "id": qid,
            "type": kind,
            "instructions": ins if isinstance(ins, str) else json.dumps(ins),
        }
        if kind == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise SchemaError(
                    f"question {qid!r}: noul criteria must be an object with "
                    "true and false"
                )
            item["criteria"] = crit
        elif kind == "choice":
            if not isinstance(crit, dict) or not crit:
                raise SchemaError(
                    f"question {qid!r}: choice criteria must map option names "
                    "to descriptions"
                )
            item["options"] = [
                {"name": str(n), "description": d} for n, d in crit.items()
            ]
        elif kind == "score":
            if not isinstance(crit, list):
                raise SchemaError(
                    f"question {qid!r}: score criteria must be an ordered list "
                    "of levels"
                )
            item["levels"] = crit
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        for key in ("depends_on", "ask_if", "alone"):
            if key in q:
                item[key] = q[key]
        out.append(item)
    schema = {k: body[k] for k in JEV_EXTENSIONS if k in body}
    schema["questions"] = out
    return parse_schema(schema)


def image_part(content_type, data):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{content_type};base64," + base64.b64encode(data).decode()
        },
    }


def jev_images(value):
    """Image parts from the body's "images": data URLs, or objects with
    content_type and base64."""
    parts = []
    for i, im in enumerate(value or []):
        if isinstance(im, str) and im.startswith("data:image/"):
            parts.append({"type": "image_url", "image_url": {"url": im}})
        elif (
            isinstance(im, dict)
            and str(im.get("content_type", "")).startswith("image/")
            and isinstance(im.get("base64"), str)
        ):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{im['content_type']};base64,{im['base64']}"
                    },
                }
            )
        else:
            raise SchemaError(
                f"images[{i}]: a data:image/... URL or an object with "
                "content_type and base64"
            )
    return parts


def jev_state(body, image_parts=()):
    """The user message: the state as text (as given, or as JSON), with any
    images ahead of it."""
    state = body.get("state")
    if state is None:
        raise SchemaError("state: required")
    text = state if isinstance(state, str) else json.dumps(state)
    if not image_parts:
        return text
    return list(image_parts) + [{"type": "text", "text": text}]


def jev_answer(q, a):
    if a is None:
        return None
    if q["type"] == "noul":
        return {"type": "noul", "noul": a["noul"]}
    if q["type"] == "choice":
        return {
            "type": "choice",
            "choice": a["choice"],
            "probabilities": a["probabilities"],
            "confidence": a["confidence"],
        }
    names = [c[0] for c in q["choices"]]
    probs = {str(i): a["probabilities"][n] for i, n in enumerate(names)}
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs.values())),
        "legend": {str(i): n for i, n in enumerate(names)},
        "probabilities": probs,
        "confidence": a["confidence"],
    }


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


def message_text(m):
    c = m.get("content", "")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        return self._json(404, {"error": {"message": "unknown route"}})

    def _read_request(self):
        """-> (body, image parts): a JSON body, or multipart/form-data with the
        JSON in a part named request and each image as a file part, in order."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        ctype = self.headers.get("content-type", "")
        if not ctype.lower().startswith("multipart/form-data"):
            return json.loads(raw), []
        from email.parser import BytesParser
        from email.policy import HTTP

        msg = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw
        )
        body, images = None, []
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            data = part.get_payload(decode=True)
            if name == "request":
                body = json.loads(data)
            elif part.get_content_type().startswith("image/"):
                images.append(image_part(part.get_content_type(), data))
            else:
                raise ValueError(
                    f"part {name!r}: neither the request JSON nor an image"
                )
        if body is None:
            raise ValueError(
                "multipart needs a part named request holding the JSON body"
            )
        return body, images

    def do_POST(self):
        if API_KEY and self.headers.get("authorization", "") != f"Bearer {API_KEY}":
            return self._json(
                401,
                {
                    "error": {
                        "message": "missing or wrong bearer token",
                        "type": "authentication_error",
                    }
                },
            )
        if self.path == "/v1/raw/chat/completions":
            return self._raw_chat()
        try:
            req, images = self._read_request()
        except Exception as e:
            return self._json(
                400,
                {
                    "error": {
                        "message": f"invalid body: {e}",
                        "type": "invalid_request_error",
                    }
                },
            )
        if self.path == "/v1/systemone":
            return self._systemone(req, images)
        if self.path == "/v1/chat/completions":
            return self._chat(req)
        return self._json(404, {"error": {"message": "unknown route"}})

    def _raw_chat(self):
        """Pass the body and status through to vLLM's chat completions."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        req = urllib.request.Request(
            ARGS.upstream.rstrip("/") + "/v1/chat/completions",
            data=raw,
            headers={
                "content-type": self.headers.get("content-type", "application/json")
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                code, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, body = e.code, e.read()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _decide(self, schema, state, seed):
        """-> (status, body) with the error body already shaped."""
        try:
            return 200, decide(schema, state, seed)
        except SchemaError as e:
            return 422, {"error": {"message": str(e), "type": "validation_error"}}
        except urllib.error.HTTPError as e:
            return 502, {
                "error": {
                    "message": (
                        f"upstream {e.code}: {e.read()[:300].decode(errors='replace')}"
                    ),
                    "type": "server_error",
                }
            }
        except Exception as e:
            return 500, {"error": {"message": repr(e), "type": "server_error"}}

    def _systemone(self, req, images):
        try:
            schema = jev_schema(req)
            state = jev_state(req, images + jev_images(req.get("images")))
        except SchemaError as e:
            return self._json(
                422, {"error": {"message": str(e), "type": "validation_error"}}
            )
        code, result = self._decide(schema, state, int(req.get("seed", 42)))
        if code != 200:
            return self._json(code, result)
        body, completion_tokens = result
        answers = {
            q["id"]: jev_answer(q, body["answers"][q["id"]])
            for q in schema["questions"]
        }
        labels = " ".join(
            f"{k}={v['label'] if v else 'skipped'}" for k, v in body["answers"].items()
        )
        print(
            f"systemone: {labels} "
            f"reads={body['diagnostics']['timing']['reads']} "
            f"{body['diagnostics']['timing']['total_ms']:.0f}ms",
            flush=True,
        )
        self._json(
            200,
            {
                "model": ARGS.model,
                "answers": answers,
                # vLLM's prompt count when the reads reported one, since it covers
                # images. Otherwise the tokenizer's count of the text prompt.
                "usage": {
                    "input_tokens": body["diagnostics"].get("prompt_tokens")
                    or (
                        len(chat_prompt_ids(system_text(schema), state))
                        if isinstance(state, str)
                        else 0
                    ),
                    "output_tokens": completion_tokens,
                },
                "diagnostics": body["diagnostics"],
            },
        )

    def _chat(self, req):
        msgs = req.get("messages") or []
        if (
            len(msgs) != 2
            or msgs[0].get("role") not in ("system", "developer")
            or msgs[1].get("role") != "user"
        ):
            return self._json(
                400,
                {
                    "error": {
                        "message": (
                            "a structured request is exactly two messages: the "
                            "schema (system) and the state JSON (user)"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )
        try:
            schema_value = json.loads(message_text(msgs[0]))
            schema = parse_schema(schema_value)
            content = msgs[1].get("content", "")
            has_image = isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in ("image_url", "image")
                for p in content
            )
            if has_image:
                # image parts pass through to vLLM unchanged, with text parts as context
                state = content
            else:
                state = message_text(msgs[1]).strip()
                json.loads(state)
        except SchemaError as e:
            return self._json(
                400, {"error": {"message": str(e), "type": "invalid_request_error"}}
            )
        except Exception as e:
            return self._json(
                400,
                {
                    "error": {
                        "message": (
                            "system must be a JSON question schema and user "
                            f"must be JSON state or image parts: {e}"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )
        code, result = self._decide(schema, state, int(req.get("seed", 42)))
        if code != 200:
            if code == 422:
                result["error"]["type"] = "invalid_request_error"
                code = 400
            return self._json(code, result)
        body, completion_tokens = result
        content = json.dumps(body, indent=2)
        labels = " ".join(
            f"{k}={v['label'] if v else 'skipped'}" for k, v in body["answers"].items()
        )
        print(
            f"structured: {labels} "
            f"reads={body['diagnostics']['timing']['reads']} "
            f"{body['diagnostics']['timing']['total_ms']:.0f}ms",
            flush=True,
        )
        self._json(
            200,
            {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", "dgemma-structured"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "total_tokens": completion_tokens,
                },
            },
        )


def self_signed(cert_dir):
    """Paths of a self-signed certificate and key in cert_dir, made with
    openssl on first use."""
    os.makedirs(cert_dir, exist_ok=True)
    cert, key = os.path.join(cert_dir, "djev.crt"), os.path.join(cert_dir, "djev.key")
    if not (os.path.exists(cert) and os.path.exists(key)):
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "3650",
                "-subj",
                "/CN=djev",
                "-keyout",
                key,
                "-out",
                cert,
            ],
            check=True,
            capture_output=True,
        )
    return cert, key


def serve_tls(host, port, cert_dir):
    cert, key = self_signed(cert_dir)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    global ARGS, CANVAS_LEN, CANVAS_STEP
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default="http://127.0.0.1:8010")
    p.add_argument("--model", default="dgemma")
    p.add_argument("--tokenizer", default="/models/dgemma", help="HF id or local path")
    p.add_argument("--canvas", type=int, default=64, help="the served canvas length")
    p.add_argument(
        "--canvas-step",
        type=int,
        default=16,
        help="request widths round up to a multiple of this",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8011)
    p.add_argument(
        "--tls-port", type=int, default=0, help="also listen with HTTPS here (0 = off)"
    )
    p.add_argument(
        "--cert-dir",
        default=os.path.expanduser("~/.cache/djev"),
        help="directory for the self-signed certificate",
    )
    ARGS = p.parse_args()
    CANVAS_LEN = ARGS.canvas
    CANVAS_STEP = ARGS.canvas_step
    init_tokenizer(AutoTokenizer.from_pretrained(ARGS.tokenizer))
    if ARGS.tls_port:
        serve_tls(ARGS.host, ARGS.tls_port, ARGS.cert_dir)
        print(
            f"structured server https on {ARGS.host}:{ARGS.tls_port} (self-signed)",
            flush=True,
        )
    print(
        f"structured server on {ARGS.host}:{ARGS.port} -> {ARGS.upstream} "
        f"(canvas {CANVAS_LEN})",
        flush=True,
    )
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
