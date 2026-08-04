# SPDX-License-Identifier: Apache-2.0
"""HTTP request for VLLM Serving Engine."""

# Standard
from typing import Any, Optional
import json
import sys
import time
import urllib.error
import urllib.request

_MAX_ERR = 65536
MetricValue = tuple[str, Any]
MetricMap = dict[str, MetricValue]
_METRIC_NAMES = {
    "prompt_tokens": "Input tokens",
    "output_tokens": "Output tokens",
    "ttft_ms": "TTFT (ms)",
    "tpot_ms_per_token": "TPOT (ms/token)",
    "total_latency_ms": "Total latency (ms)",
    "throughput_tokens_per_s": "Throughput (tokens/s)",
    "model": "Model",
}


def _clip(text: str, limit: int = _MAX_ERR) -> str:
    return (
        text
        if len(text) <= limit
        else text[: max(0, limit - 24)] + "\n...(message truncated)..."
    )


def _info(msg: str) -> None:
    print(f"lmcache query: {msg}", file=sys.stderr)


def _openai_error(obj: dict[str, Any]) -> Optional[str]:
    err = obj.get("error")
    if err is None:
        return None
    if isinstance(err, str):
        return err.strip() or None
    if not isinstance(err, dict):
        return _clip(str(err))
    for key in ("message", "detail"):
        val = err.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        typ = err.get("type") or err.get("code")
        if key == "message" and isinstance(typ, str) and typ.strip():
            return f"{typ.strip()}: {val.strip()}"
        return val.strip()
    try:
        return _clip(json.dumps(err, ensure_ascii=False))
    except Exception:
        return _clip(str(err))


def _raise_openai_error(obj: dict[str, Any]) -> None:
    msg = _openai_error(obj)
    if msg:
        raise RuntimeError(_clip(msg))


def _raise_json_blob_error(blob: str) -> None:
    s = blob.strip()
    if not s.startswith("{"):
        return
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return
    if isinstance(obj, dict):
        _raise_openai_error(obj)


def _api_url(base: str, path: str) -> str:
    base = base.strip()
    if "://" not in base:
        base = f"http://{base}"
    base = base.rstrip("/")
    return f"{base if base.endswith('/v1') else base + '/v1'}/{path}"


def _read_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="GET"), timeout=max(timeout + 2.0, 5.0)
        ) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:512]
        raise RuntimeError(
            f"GET {url} failed (HTTP {e.code}): {body or 'no body'}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"GET {url} failed: {getattr(e, 'reason', e)}") from e
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from GET {url}: {e}") from e
    if not isinstance(obj, dict):
        raise RuntimeError(f"GET {url}: expected a JSON object")
    return obj


def _sse_piece(obj: dict[str, Any], chat: bool) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    c0 = choices[0]
    return (
        str((c0.get("delta") or {}).get("content") or "")
        if chat
        else str(c0.get("text") or "")
    )


def _trim_misc_buffer(misc: list[str], limit: int = _MAX_ERR) -> None:
    while misc and sum(map(len, misc)) > limit:
        misc.pop(0)


def _stream(
    url: str,
    body: dict[str, Any],
    timeout: float,
    *,
    chat: bool,
    max_tokens: int,
) -> dict[str, Any]:
    """POST with ``stream: true``; parse SSE; return the completion text plus
    TTFT/TPOT and token metrics."""
    payload = {
        **body,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0, first_token_t, pieces, usage, misc = time.time(), None, [], None, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    misc.append(line)
                    _trim_misc_buffer(misc)
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    misc.append(chunk)
                    _trim_misc_buffer(misc)
                    continue
                if not isinstance(obj, dict):
                    continue
                _raise_openai_error(obj)
                piece = _sse_piece(obj, chat)
                if piece:
                    first_token_t = first_token_t or time.time()
                    pieces.append(piece)

                u_chunk = obj.get("usage")
                if u_chunk is not None:
                    usage = u_chunk
            t1 = time.time()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        _raise_json_blob_error(err_body)
        raise RuntimeError(
            _clip(f"POST {url} failed (HTTP {e.code}):\n{_clip(err_body)}")
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"POST {url} failed: {getattr(e, 'reason', e)}") from e

    misc_text = "\n".join(misc).strip()
    _raise_json_blob_error(misc_text)
    joined = "".join(pieces)
    if not joined and usage is None:
        raise RuntimeError(
            _clip(f"No completion output from engine. Captured response:\n{misc_text}")
            if misc_text
            else "Empty response from engine (no SSE chunks parsed)."
        )

    u = usage or {}
    prompt_tokens = int(u.get("prompt_tokens") or 0)
    num_completion = int(u.get("completion_tokens") or 0)
    # Match V2RequestSender: server count if present, else max_tokens cap.
    num_generated = num_completion if num_completion > 0 else max_tokens
    if first_token_t is None:
        # Use total round-trip as a conservative TTFT approximation.
        ttft_s = t1 - t0
        decode_time = 0.0
    else:
        ttft_s = first_token_t - t0
        decode_time = t1 - first_token_t
    dt = t1 - t0
    decoding_speed = (num_generated / decode_time) if decode_time > 0 else 0.0
    tpot_s = (
        (decode_time / num_generated) if num_generated > 0 and decode_time > 0 else 0.0
    )
    return {
        "answer": joined,
        "prompt_tokens": prompt_tokens,
        "output_tokens": num_generated,
        "ttft_ms": ttft_s * 1000.0,
        "tpot_ms_per_token": tpot_s * 1000.0,
        "total_latency_ms": dt * 1000.0,
        "throughput_tokens_per_s": decoding_speed,
    }


def _missing_chat_template(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "chat template",
            "chat_template",
            "chattemplate",
            "template resolution",
            "must provide a chat template",
            "default chat template is no longer allowed",
        )
    )


def _weak_completions_error(msg: str) -> bool:
    msg = msg.lower()
    return any(
        s in msg
        for s in (
            "empty response from engine",
            "no completion output from engine",
            "no sse chunks parsed",
        )
    )


class Request:
    """Build and send one query request against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base: str,
        model: Optional[str],
        max_tokens: int,
        timeout: float,
        *,
        completions_only: bool = False,
        chat_first: bool = False,
    ) -> None:
        self._base = base
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._completions_only = completions_only
        self._chat_first = chat_first

    def build_request(self, prompt: str) -> dict[str, Any]:
        """Build request payload and metadata for the provided prompt."""
        model = self._model or self._first_model_id()
        return {
            "base": self._base,
            "model": model,
            "prompt": prompt,
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
            "completions_only": self._completions_only,
            "chat_first": self._chat_first,
        }

    def send_request(self, prompt: str) -> tuple[str, MetricMap]:
        """Send request and return the completion text and its metrics.

        Args:
            prompt: The fully expanded prompt to send to the engine.

        Returns:
            A ``(answer, metrics)`` tuple where ``answer`` is the model's
            completion text and ``metrics`` is a :data:`MetricMap` of token
            and latency stats keyed by metric id.
        """
        request_data = self.build_request(prompt)
        result = self._query_with_fallback(request_data)
        answer = str(result.pop("answer", ""))
        stats = {"model": request_data["model"], **result}
        metrics = {key: (_METRIC_NAMES.get(key, key), stats[key]) for key in stats}
        return answer, metrics

    def _first_model_id(self) -> str:
        """Return the first model ID from ``GET /v1/models``."""
        obj = _read_json(_api_url(self._base, "models"), self._timeout)
        data = obj.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(
                "GET /v1/models returned no models; pass --model explicitly."
            )
        first = data[0]
        if not isinstance(first, dict) or "id" not in first:
            raise RuntimeError("GET /v1/models: first entry missing 'id'.")
        return str(first["id"])

    def _query_with_fallback(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Send one query and fallback between completions/chat endpoints."""
        if request_data["completions_only"]:
            return self._query(request_data, chat=False)
        try:
            return self._query(request_data, chat=request_data["chat_first"])
        except RuntimeError as first_err:
            if request_data["chat_first"]:
                if not _missing_chat_template(first_err):
                    raise
                _info(
                    "chat API failed (no chat template); retrying with /v1/completions"
                )
                return self._query(request_data, chat=False)
            _info("/v1/completions failed; retrying with /v1/chat/completions")
            try:
                return self._query(request_data, chat=True)
            except RuntimeError as second_err:
                if _weak_completions_error(str(first_err)) and _missing_chat_template(
                    second_err
                ):
                    _info(
                        "base / completion-only models: try `--completions` or "
                        "an instruct model with a chat template."
                    )
                    raise second_err
                raise RuntimeError(f"{first_err}; then {second_err}") from second_err

    def _query(self, request_data: dict[str, Any], *, chat: bool) -> dict[str, Any]:
        path = "chat/completions" if chat else "completions"
        model = request_data["model"]
        prompt = request_data["prompt"]
        max_tokens = request_data["max_tokens"]
        timeout = request_data["timeout"]
        body = (
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            if chat
            else {"model": model, "prompt": prompt, "max_tokens": max_tokens}
        )
        return _stream(
            _api_url(request_data["base"], path),
            body,
            timeout,
            chat=chat,
            max_tokens=max_tokens,
        )
