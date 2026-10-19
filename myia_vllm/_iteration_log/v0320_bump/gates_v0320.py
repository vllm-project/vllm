"""v0.32.0 bump gate battery — reproduit la batterie 13 gates des bumps v0.29/v0.30.

Usage: python gates_v0320.py
Sortie: PASS/FAIL par gate, récapitulatif texte + JSON (gates_v0320_<epoch>.json).

Gates: smoke · vision (PNG 4 quadrants généré en stdlib) · tool-calling ·
thinking · preserve_thinking · prefill 30k/95k/235k (+ survie) · concurrent N=16 ×2
(prefix cache) · FORCING (6× ~4.6k tok concurrents → forwards dans (4096, 8192]).
"""
import base64
import concurrent.futures
import json
import os
import struct
import time
import urllib.request
import zlib

API = "http://localhost:5002/v1/chat/completions"
MODEL = "qwen3.6-35b-a3b"
KEY = None
for line in open("d:/vllm/myia_vllm/.env"):
    if line.startswith("VLLM_API_KEY_MEDIUM="):
        KEY = line.split("=", 1)[1].strip()
assert KEY, "VLLM_API_KEY_MEDIUM not found in .env"


def post(payload: dict, timeout: int = 600) -> tuple[dict, float]:
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, time.time() - t0


def text_of(body: dict) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def png_4quadrants(w: int = 64, h: int = 64) -> bytes:
    """Minimal PNG: 4 solid-color quadrants (TL red, TR green, BL blue, BR yellow)."""
    rows = b""
    for y in range(h):
        row = b"\x00"
        for x in range(w):
            tl, tr = x < w // 2, y < h // 2
            px = (255, 0, 0) if (tl and tr) else (0, 255, 0) if tr else (0, 0, 255) if tl else (255, 255, 0)
            row += bytes(px)
        rows += row

    def chunk(typ: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


RESULTS: list[dict] = []


def record(gate: str, ok: bool, detail: str) -> None:
    RESULTS.append({"gate": gate, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {gate}: {detail}", flush=True)


def fill_text(target_tokens: int) -> str:
    # ~4 chars/token English filler; final prompt_tokens read from the response usage.
    return ("The quick brown fox jumps over the lazy dog near the riverbank. " * (target_tokens // 11 + 1))[: target_tokens * 4]


def main() -> None:
    # 1. smoke
    try:
        body, dt = post({"model": MODEL, "messages": [{"role": "user", "content": "Reply with exactly: PONG"}], "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        ok = "PONG" in text_of(body).upper()
        record("smoke", ok, f"{dt:.2f}s → {text_of(body).strip()[:40]!r}")
    except Exception as e:
        record("smoke", False, f"EXC {e}")
        return  # no point continuing if the engine is dead

    # 2. vision — 4-quadrant PNG
    try:
        b64 = base64.b64encode(png_4quadrants()).decode()
        body, dt = post(
            {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": "This image shows 4 solid color quadrants. List the 4 colors in order top-left, top-right, bottom-left, bottom-right. One word each."},
                        ],
                    }
                ],
                "max_tokens": 200,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        t = text_of(body).lower()
        # French or English color names both accepted
        hits = sum(any(w in t for w in words) for words in [("red", "rouge"), ("green", "vert"), ("blue", "bleu"), ("yellow", "jaune")])
        record("vision", hits == 4, f"{hits}/4 colors in {dt:.1f}s → {text_of(body).strip()[:80]!r}")
    except Exception as e:
        record("vision", False, f"EXC {e}")

    # 3. tool-calling
    try:
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        body, dt = post({"model": MODEL, "messages": [{"role": "user", "content": "What is the weather in Lyon right now? Use the tool."}], "tools": tools, "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        tc = body["choices"][0]["message"].get("tool_calls") or []
        ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "lyon" in tc[0]["function"]["arguments"].lower()
        record("tool-calling", ok, f"{dt:.1f}s → {tc[0]['function']['arguments'][:60] if tc else 'no tool_calls'}")
    except Exception as e:
        record("tool-calling", False, f"EXC {e}")

    # 4. thinking
    try:
        body, dt = post({"model": MODEL, "messages": [{"role": "user", "content": "A bat and a ball cost 1.10 together. The bat costs 1.00 more than the ball. How much is the ball? Answer briefly."}], "max_tokens": 2000, "temperature": 0})
        reasoning = body["choices"][0]["message"].get("reasoning") or ""
        ok = len(reasoning) > 50 and body["choices"][0]["finish_reason"] == "stop"
        record("thinking", ok, f"{len(reasoning)} chars reasoning, finish={body['choices'][0]['finish_reason']}")
    except Exception as e:
        record("thinking", False, f"EXC {e}")

    # 5. preserve_thinking — assistant history keeps its thinking block
    try:
        msgs = [
            {"role": "user", "content": "Remember: my favorite number is 17."},
            {"role": "assistant", "content": "Sure, I will remember that your favorite number is 17. Is there anything else?"},
            {"role": "user", "content": "What is my favorite number? One word."},
        ]
        body, dt = post({"model": MODEL, "messages": msgs, "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        ok = "17" in text_of(body)
        record("preserve_thinking", ok, f"{dt:.1f}s → {text_of(body).strip()[:50]!r}")
    except Exception as e:
        record("preserve_thinking", False, f"EXC {e}")

    # 6-8. prefill ladders + survival
    for name, target in [("prefill-30k", 31_500), ("prefill-95k", 102_000), ("prefill-235k", 253_500)]:
        try:
            prompt = fill_text(target) + "\n\nReply with exactly: OK"
            body, dt = post({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 24, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=900)
            pt = body.get("usage", {}).get("prompt_tokens", 0)
            ok = "OK" in text_of(body).upper()
            record(name, ok, f"{pt} tok in {dt:.1f}s → {pt / dt:.0f} tok/s; survival={'OK' in text_of(body).upper()}")
        except Exception as e:
            record(name, False, f"EXC {e}")

    # 9. concurrent N=16 ×2 (distinct prompts; repeat shows prefix-cache warmth)
    prompts16 = [f"Explain in 3 sentences why {topic} matters for LLM inference. Be concise." for topic in ["paged attention", "KV cache quantization", "tensor parallelism", "chunked prefill", "speculative decoding", "prefix caching", "CUDA graphs", "continuous batching", "flash attention", "expert parallelism", "MoE routing", "quantization error", "context length", "prefill/decode split", "sampling temperature", "beam search"]]

    def one(p: str) -> float:
        body, _ = post({"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 150, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
        return body.get("usage", {}).get("completion_tokens", 0)

    for round_name in ["concurrent-N16", "concurrent-N16-repeat"]:
        try:
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(16) as ex:
                comps = list(ex.map(one, prompts16))
            dt = time.time() - t0
            record(round_name, all(c > 0 for c in comps), f"{sum(comps)} tokens in {dt:.1f}s → {sum(comps) / dt:.0f} tok/s aggregate")
        except Exception as e:
            record(round_name, False, f"EXC {e}")

    # 10. FORCING — 6× ~4618-token prompts concurrently → forwards in (4096, 8192]
    try:
        fp = fill_text(4_618) + "\n\nReply with exactly: OK"
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(6) as ex:
            outs = list(ex.map(lambda _: post({"model": MODEL, "messages": [{"role": "user", "content": fp}], "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=600), range(6)))
        dt = time.time() - t0
        oks = ["OK" in text_of(b).upper() for b, _ in outs]
        tot = sum(b.get("usage", {}).get("prompt_tokens", 0) for b, _ in outs)
        record("FORCING", all(oks), f"6×(≈4618 tok) = {tot} tok in {dt:.1f}s, all-200={all(oks)}")
    except Exception as e:
        record("FORCING", False, f"EXC {e}")

    # summary
    n_pass = sum(1 for r in RESULTS if r["ok"])
    print(f"\nBATTERY: {n_pass}/{len(RESULTS)} PASS", flush=True)
    out = f"d:/vllm/myia_vllm/_iteration_log/v0320_bump/gates_v0320_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump({"ts": int(time.time()), "pass": n_pass, "total": len(RESULTS), "results": RESULTS}, f, indent=2)
    print(f"JSON: {out}", flush=True)


if __name__ == "__main__":
    main()