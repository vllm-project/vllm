#!/usr/bin/env python3
"""
Debug client for a local vLLM OpenAI-compatible server: one chat-completions
call and one text completions call. Uses greedy decoding (temperature=0) and
100 completion tokens for each.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "http://127.0.0.1:9527"
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = str(SCRIPT_DIR / "single_request_test.log")
MAX_TOKENS = 30
TEMPERATURE = 0.0
TOP_P = 1.0
STREAM = False
REQUEST_TIMEOUT_S = 600.0

# Deliberately non-trivial: multi-constraint scheduling + arithmetic.
CHALLENGING_USER_PROMPT = """You are given:
- Task A takes 7 hours and must finish before 18:00.
- Task B takes 5 hours and cannot overlap A; B needs 1 hour cool-down after any work before C.
- Task C takes 4 hours and must start after 12:00.
Work day is 09:00–18:00 (no overnight). If all three must complete today, is it feasible?
Answer with a tight schedule (start/end times) or explain the contradiction. Be concise."""

COMPLETION_PROMPT = "深圳是个很好的"


def _json_request(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick_first_model_id(base_url: str) -> str:
    r = _json_request("GET", f"{base_url.rstrip('/')}/v1/models")
    models = r.get("data") or []
    if not models:
        raise RuntimeError("No models returned from /v1/models")
    mid = models[0].get("id")
    if not mid:
        raise RuntimeError(f"Unexpected /v1/models payload: {r!r}")
    return str(mid)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("single_request_test")

    base = BASE_URL.rstrip("/")

    try:
        model_id = pick_first_model_id(base)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.error("Failed to query /v1/models: %s", e)
        return 1
    except Exception as e:
        log.error("Failed to parse /v1/models: %s", e)
        return 1

    log.info("Resolved model id: %s", model_id)

    # --- /v1/chat/completions ---
    # chat_url = f"{base}/v1/chat/completions"
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a careful analytical assistant. Follow instructions precisely.",
    #     },
    #     {"role": "user", "content": CHALLENGING_USER_PROMPT},
    # ]
    
    # chat_payload: dict[str, Any] = {
    #     "model": model_id,
    #     "messages": messages,
    #     "max_tokens": MAX_TOKENS,
    #     "temperature": TEMPERATURE,
    #     "top_p": TOP_P,
    #     "stream": STREAM,
    # }
    
    # log.info("=== POST /v1/chat/completions ===")
    # log.info("Request URL: %s", chat_url)
    # log.info("Request payload (JSON):\n%s", json.dumps(chat_payload, ensure_ascii=False, indent=2))
    
    # try:
    #     chat_resp = _json_request("POST", chat_url, body=chat_payload, timeout_s=REQUEST_TIMEOUT_S)
    # except urllib.error.HTTPError as e:
    #     body = e.read().decode("utf-8", errors="replace")
    #     log.error("HTTP %s: %s", e.code, body)
    #     return 1
    # except (urllib.error.URLError, TimeoutError, OSError) as e:
    #     log.error("Request failed: %s", e)
    #     return 1
    
    # log.info("Raw response (JSON):\n%s", json.dumps(chat_resp, ensure_ascii=False, indent=2))
    
    # try:
    #     ch0 = chat_resp["choices"][0]
    #     chat_text = ch0.get("message", {}).get("content")
    #     chat_finish = ch0.get("finish_reason")
    # except Exception:
    #     chat_text, chat_finish = None, None
    
    # log.info("Extracted assistant content:\n%s", chat_text if chat_text is not None else "<missing>")
    # log.info("Finish reason: %s", chat_finish)

    # --- /v1/completions ---
    comp_url = f"{base}/v1/completions"
    comp_payload: dict[str, Any] = {
        "model": model_id,
        "prompt": COMPLETION_PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": STREAM,
    }

    log.info("=== POST /v1/completions ===")
    log.info("Request URL: %s", comp_url)
    log.info("Request payload (JSON):\n%s", json.dumps(comp_payload, ensure_ascii=False, indent=2))

    try:
        comp_resp = _json_request("POST", comp_url, body=comp_payload, timeout_s=REQUEST_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        log.error("HTTP %s: %s", e.code, body)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.error("Request failed: %s", e)
        return 1

    log.info("Raw response (JSON):\n%s", json.dumps(comp_resp, ensure_ascii=False, indent=2))

    try:
        co0 = comp_resp["choices"][0]
        comp_text = co0.get("text")
        comp_finish = co0.get("finish_reason")
    except Exception:
        comp_text, comp_finish = None, None

    log.info("Extracted completion text:\n%s", comp_text if comp_text is not None else "<missing>")
    log.info("Finish reason: %s", comp_finish)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
