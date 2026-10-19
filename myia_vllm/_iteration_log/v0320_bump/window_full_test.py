"""#57035 behaviour test: prompt + max_tokens reservation exceeding max_model_len.

v0.31 (measured 2026-10-09, PR #110): prompt+max_tokens > 262 144 → HTTP 400
"maximum context length" WITHOUT trimming. v0.32 changelog: "max_tokens is an
upper bound instead of a reservation, so long prompts are no longer rejected"
(#57035).

Target case: prompt ≈ 260K tokens (under the window) + max_tokens=4096 →
reservation sum ≈ 264K > 262 144. v0.32 expected: HTTP 200; v0.31 behaviour
would be HTTP 400. Prompt size is calibrated via /tokenize before sending.
"""
import json
import time
import urllib.error
import urllib.request

API = "http://localhost:5002/v1/chat/completions"
TOKENIZE = "http://localhost:5002/tokenize"
MODEL = "qwen3.6-35b-a3b"
KEY = None
for line in open("d:/vllm/myia_vllm/.env"):
    if line.startswith("VLLM_API_KEY_MEDIUM="):
        KEY = line.split("=", 1)[1].strip()

PHRASE = "The quick brown fox jumps over the lazy dog near the riverbank. "  # 15 tok
MAX_TOKENS = 4096
TARGET = 260_000  # prompt tokens target (under 262 144; sum with MAX_TOKENS exceeds it)


def count_tokens(prompt: str) -> int:
    req = urllib.request.Request(
        TOKENIZE,
        data=json.dumps({"model": MODEL, "prompt": prompt}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["count"]


def build(n_phrases: int, suffix: str) -> str:
    return PHRASE * n_phrases + suffix


suffix = "\n\nReply with exactly: OK"
n = TARGET // 14  # ~14.0 tok/phrase in-sequence (measured 2026-10-19)
prompt = build(n, suffix)
count = count_tokens(prompt)
print(f"calibration: {n} phrases -> {count} tokens (target ~{TARGET})", flush=True)
for _ in range(4):  # proportional refinement loop
    if 259_000 <= count <= 261_500:
        break
    n = max(1, int(n * TARGET / count))
    prompt = build(n, suffix)
    count = count_tokens(prompt)
    print(f"refined: {n} phrases -> {count} tokens", flush=True)

print(f"RESERVATION TEST: prompt={count} tok + max_tokens={MAX_TOKENS} = {count + MAX_TOKENS} > 262144 ? {count + MAX_TOKENS > 262144}", flush=True)

req = urllib.request.Request(
    API,
    data=json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
)
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        body = json.loads(r.read().decode())
    usage = body.get("usage", {})
    print(f"HTTP {r.status} in {time.time() - t0:.1f}s")
    print(f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')} finish={body['choices'][0].get('finish_reason')}")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code} in {time.time() - t0:.1f}s")
    print(f"body: {e.read().decode()[:600]}")