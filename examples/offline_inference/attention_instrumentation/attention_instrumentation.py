#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import gzip
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class TokenRange:
    start: int
    end: int

    def __contains__(self, i):
        return self.start <= i < self.end


class MultimodalTokenizer:
    def __init__(self, vision_ranges, lang_ranges, prompt_len, total_len):
        self.vision_ranges = [TokenRange(**r) for r in vision_ranges]
        self.lang_ranges = [TokenRange(**r) for r in lang_ranges]
        self.prompt_len = prompt_len
        self.total_len = total_len

    def token_type(self, i):
        if i >= self.prompt_len:
            return "generated"
        if any(i in r for r in self.vision_ranges):
            return "vision"
        if any(i in r for r in self.lang_ranges):
            return "language"
        return "unknown"

    def get_vision_tokens(self):
        return [i for r in self.vision_ranges for i in range(r.start, r.end)]

    def get_lang_tokens(self):
        return [i for r in self.lang_ranges for i in range(r.start, r.end)]


class AttentionAnalyzer:
    """Wraps attention scores [T, H, T] with token-type context."""

    def __init__(self, attn_scores, token_meta: dict):
        self.s = attn_scores  # [T, H, T]
        self.tokenizer = MultimodalTokenizer(
            token_meta.get("vision_ranges", []),
            token_meta.get("lang_ranges", []),
            token_meta.get("prompt_len", 0),
            token_meta.get("total_len", 0),
        )

    def _w(self, q, h, avg):
        if avg:
            return self.s[q].mean(0)
        if h is not None:
            return self.s[q, h]
        return self.s[q]

    def attention_for_token(self, q, head_idx=None, avg_heads=False):
        """[H,T] or [T] attention weights for query token q."""
        return self._w(q, head_idx, avg_heads)

    def top_attended_tokens(self, q, head_idx=None, top_k=5, avg_heads=False):
        """List of (key_idx, weight, token_type) sorted by weight desc."""
        if not avg_heads and head_idx is None:
            raise ValueError("Provide head_idx or avg_heads=True")
        w = self._w(q, head_idx, avg_heads)
        top = np.argsort(w)[::-1][:top_k]
        return [(int(i), float(w[i]), self.tokenizer.token_type(i)) for i in top]

    def cross_modal_attention(self, head_idx=None, avg_heads=False) -> float:
        """Fraction of vision-token attention on language tokens."""
        vis, lng = self.tokenizer.get_vision_tokens(), self.tokenizer.get_lang_tokens()
        if not vis or not lng:
            return 0.0
        if not avg_heads and head_idx is None:
            raise ValueError("Provide head_idx or avg_heads=True")
        sv = self.s[vis].mean(1) if avg_heads else self.s[vis, head_idx]
        t = float(sv.sum())
        return float(sv[:, lng].sum()) / t if t else 0.0


def extract_attention_from_response(response: dict) -> dict | None:
    data = response.get("attn_capture_data", [])
    if not data:
        return None
    out = {}
    for item in data:
        b64, shape = item.get("data"), item.get("shape")
        if not (b64 and shape):
            continue
        scores = (
            np.frombuffer(gzip.decompress(base64.b64decode(b64)), dtype=np.float16)
            .astype(np.float32)
            .reshape(shape)
        )
        li = item.get("layer_idx")
        out[li] = {
            "scores": scores,
            "token_meta": item.get("token_meta", {}),
            "model": response.get("model"),
            "layer": li,
        }
    return out or None


def build_token_map(model_id, prompt, response):
    """idx→text for prompt tokens (chat-template) + generated tokens (logprobs)."""
    m, n = {}, 0
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        full = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok.encode(full, add_special_tokens=False)
        n = len(ids)
        m = {i: tok.decode([t]).replace("\n", "↵") for i, t in enumerate(ids)}
    except Exception:
        pass
    try:
        for off, lp in enumerate(response["choices"][0]["logprobs"]["content"]):
            m[n + off] = lp["token"].replace("\n", "↵")
    except Exception:
        pass
    return m, n


def call_api(client, model_id, prompt, layers, max_tokens=None):
    raw = client.chat.completions.with_raw_response.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"attn_capture": 1, "attn_capture_layers": layers},
        **({} if max_tokens is None else {"max_tokens": max_tokens}),
        temperature=0,
        logprobs=True,
    )
    return json.loads(raw.content)


def fmt(k, m):
    """'text(idx)' label."""
    raw = m.get(k)
    t = raw.strip() if raw is not None else None
    return f"{t or repr(raw) or '?'}({k})"


def row(w, indices, m, mark=None):
    """'tok(i)=0.23◀  tok(j)=0.11  ...' for given indices."""
    return "  ".join(
        f"{fmt(k, m)}={w[k]:.2f}{'◀' if mark and k in mark else ''}" for k in indices
    )


def main(
    api_base="http://127.0.0.1:8000/v1", api_key="EMPTY", layers="2,18,27"
) -> None:
    from openai import OpenAI

    client, model_id = OpenAI(api_key=api_key, base_url=api_base), None
    model_id = client.models.list().data[0].id

    pairs = [
        ("alpha", "1729"),
        ("beta", "2048"),
        ("gamma", "3141"),
        ("delta", "4096"),
        ("epsilon", "5678"),
        ("zeta", "6174"),
        ("eta", "7777"),
        ("theta", "8008"),
        ("iota", "9999"),
        ("kappa", "1234"),
        ("lambda", "5555"),
        ("mu", "3721"),
        ("nu", "8642"),
        ("xi", "2468"),
        ("omicron", "1357"),
        ("alpha", None),
    ]
    prompt, expected = (
        "\n".join(f"{k}: {v}" if v else f"{k}:" for k, v in pairs),
        "1729",
    )

    print(f"{'=' * 60}\nNeedle-in-a-haystack  |  model={model_id}  layers={layers}")
    print(f"Prompt:\n{prompt}\nExpected: {expected!r}")

    resp = call_api(client, model_id, prompt, layers)
    generated = resp["choices"][0]["message"]["content"].strip()
    print(f"Output: {generated!r}  {'✓' if expected in generated else '✗'}")

    attn = extract_attention_from_response(resp)
    if not attn:
        print(
            "No attention data — start server with --enable-attention-instrumentation"
        )
        return

    m, _ = build_token_map(model_id, prompt, resp)
    needle = {k for k, t in m.items() if t.strip() == "alpha"}

    for li in sorted(attn):
        s, meta = attn[li]["scores"], attn[li]["token_meta"]
        T, H, _ = s.shape
        pl = meta["prompt_len"]
        q = T - 1
        print(f"\n── L{li} (T={T} H={H} prompt={pl}) ──")
        avg = s[q].mean(0)
        print(f"  avg  {row(avg, np.argsort(avg)[::-1][:5], m, needle)}")
        for h in range(H):
            w = s[q, h]
            top3 = np.argsort(w)[::-1][:3]
            hit = any(k in needle and w[k] > 0.15 for k in top3)
            print(f"  h{h:<2} {row(w, top3, m, needle if hit else None)}")


def example_codename_retrieval(
    api_base="http://127.0.0.1:8000/v1", api_key="EMPTY", layers="0,9,12,18,24,27"
) -> None:
    from openai import OpenAI

    client, model_id = OpenAI(api_key=api_key, base_url=api_base), None
    model_id = client.models.list().data[0].id

    prompt = (
        "Security Log: Agent Codename Retrieval\n\n"
        "[Entry 01] Agent: James,  Codename: 'Falcon'\n"
        "[Entry 02] Agent: Sarah,  Codename: 'Whisper'\n"
        "[Entry 03] Agent: Mike,   Codename: 'Hammer'\n"
        "[Entry 04] Agent: Omega,  Codename: 'Phantom'\n"
        "[Entry 05] Agent: Linda,  Codename: 'Spark'\n"
        "[Entry 06] Agent: Robert, Codename: 'Echo'\n\n"
        "Request: Retrieve the codename for Agent Omega.\n"
        "Agent: Omega, Codename:"
    )
    expected = "Phantom"

    print(f"{'=' * 60}\nCodename retrieval  |  model={model_id}  layers={layers}")
    print(f"Prompt:\n{prompt}\nExpected: {expected!r}")

    resp = call_api(client, model_id, prompt, layers)
    generated = resp["choices"][0]["message"]["content"].strip()
    ok = "✓" if expected.lower() in generated.lower() else "✗"
    print(f"Output: {generated!r}  {ok}")

    attn = extract_attention_from_response(resp)
    if not attn:
        print(
            "No attention data — start server with --enable-attention-instrumentation"
        )
        return

    m, _ = build_token_map(model_id, prompt, resp)

    for li in sorted(attn):
        s, meta = attn[li]["scores"], attn[li]["token_meta"]
        T, H, _ = s.shape
        pl = meta["prompt_len"]
        total_len = meta.get("total_len", T)
        offset = total_len - T
        print(f"\n── L{li} (T={T} H={H} prompt={pl}) ──")
        for qi in range(T):
            abs_qi = qi + offset
            if abs_qi < pl:
                continue  # skip prompt rows, show only generated tokens
            tok_label = m.get(abs_qi, "?")
            # avg across heads — full key axis (all T key positions)
            avg = s[qi].mean(0)  # [T]
            top5_rel = np.argsort(avg)[::-1][:5]
            top5_abs = top5_rel + offset
            w_g = {int(a): float(avg[r]) for r, a in zip(top5_rel, top5_abs)}
            print(f"  {tok_label!r:<12} [avg] → {row(w_g, top5_abs, m)}")
            # per-head
            for h in range(H):
                w_h = s[qi, h]  # [T]
                top3_rel = np.argsort(w_h)[::-1][:3]
                top3_abs = top3_rel + offset
                w_hg = {int(a): float(w_h[r]) for r, a in zip(top3_rel, top3_abs)}
                print(f"  {'':<12} [h{h:<2}] → {row(w_hg, top3_abs, m)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--example", choices=["needle", "codename"], default="needle")
    p.add_argument("--layers", default=None)
    args = p.parse_args()
    fn = main if args.example == "needle" else example_codename_retrieval
    fn(
        api_base=args.api_base,
        api_key=args.api_key,
        **({"layers": args.layers} if args.layers else {}),
    )
