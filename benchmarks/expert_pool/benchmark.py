# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send one frozen warmup request, then one measured request to a fresh server.

This is issue-text generation timing, not the SWE-Lancer correctness evaluation.
See README.md for the server configuration and the historical script provenance.
"""

import argparse
import datetime
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--pair", type=Path, default=Path(__file__).with_name("pair.json"))
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="flashnext")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=int, default=1200)
    a = p.parse_args()
    source = Path(a.pair).read_bytes()
    pair = json.loads(source)
    tasks = pair["tasks"]
    if len(tasks) != 2 or [t["role"] for t in tasks] != ["warmup", "measure"]:
        raise ValueError(
            "Expected exactly one warmup task followed by one measure task"
        )
    sequence = [(tasks[0], "warmup"), (tasks[1], "measure")]
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite earlier measurements.
    with output.open("x") as out:
        for index, (task, role) in enumerate(sequence):
            prompt = task["prompt_token_ids"]
            request_body = dict(
                model=a.model,
                prompt=prompt,
                max_tokens=a.max_tokens,
                temperature=0,
                top_p=1,
                stream=True,
                stream_options={"include_usage": True},
                seed=0,
            )
            request = urllib.request.Request(
                a.base_url.rstrip("/") + "/v1/completions",
                data=json.dumps(request_body).encode(),
                headers={"Content-Type": "application/json"},
            )
            result = dict(
                schema_version=1,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                label=a.label,
                sequence_index=index,
                role=role,
                task_id=task["id"],
                pair_sha256=hashlib.sha256(source).hexdigest(),
                prompt_sha256=task["prompt_sha256"],
                expected_prompt_token_ids=task["prompt_token_ids"],
                request=request_body,
                content="",
                usage=None,
                finish_reason=None,
                first_token_s=None,
                last_token_s=None,
                stream_events=0,
            )
            print(
                f"Starting {a.label}: {role} {task['id']} "
                f"({len(task['prompt_token_ids'])} input tokens)",
                flush=True,
            )
            result["raw_sse_events"] = []
            result["sse_done"] = False
            start = time.perf_counter()  # Includes HTTP request and server scheduling.
            try:
                with urllib.request.urlopen(request, timeout=a.timeout) as response:
                    for raw in response:
                        line = raw.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            result["sse_done"] = True
                            break
                        result["raw_sse_events"].append(data)
                        event = json.loads(data)
                        if event.get("error"):
                            raise RuntimeError(event["error"])
                        if event.get("usage"):
                            result["usage"] = event["usage"]
                        for choice in event.get("choices", []):
                            content = choice.get("text") or ""
                            if content:
                                elapsed = time.perf_counter() - start
                                if result["first_token_s"] is None:
                                    result["first_token_s"] = elapsed
                                result["last_token_s"] = elapsed
                                result["content"] += content
                                result["stream_events"] += 1
                            if choice.get("finish_reason"):
                                result["finish_reason"] = choice["finish_reason"]
                result["elapsed_s"] = time.perf_counter() - start
                if not result["sse_done"]:
                    raise RuntimeError("SSE ended without DONE")
                completion_tokens = (result["usage"] or {}).get("completion_tokens")
                if (
                    type(completion_tokens) is not int
                    or completion_tokens < 1
                    or not result["content"]
                ):
                    raise RuntimeError(
                        "Missing completion token usage or nonempty output"
                    )
                if result["finish_reason"] not in ("stop", "length"):
                    raise RuntimeError(
                        "Incomplete or unexpected finish reason: "
                        f"{result['finish_reason']}"
                    )
                if (result["usage"] or {}).get("prompt_tokens") != len(
                    task["prompt_token_ids"]
                ):
                    raise RuntimeError(
                        "Server input token count does not match frozen prompt"
                    )
                result["completion_tokens"] = completion_tokens
                result["e2e_tok_s"] = completion_tokens / result["elapsed_s"]
                decode_duration = result["last_token_s"] - result["first_token_s"]
                result["decode_tok_s"] = (
                    (completion_tokens - 1) / decode_duration
                    if completion_tokens > 1 and decode_duration > 0
                    else None
                )
                for metric in ("e2e_tok_s", "decode_tok_s"):
                    value = result[metric]
                    if value is not None and not math.isfinite(value):
                        raise RuntimeError(f"Non-finite {metric}")
                result["decode_metric_note"] = (
                    "Approximate client-observed "
                    "(completion_tokens-1)/(last-text-event-first-text-event)"
                )
                result["output_sha256"] = hashlib.sha256(
                    result["content"].encode()
                ).hexdigest()
                result["truncated"] = result["finish_reason"] == "length"
                result["correctness"] = "Not graded: issue-text inference workload only"
            except Exception as exc:
                result["elapsed_s"] = time.perf_counter() - start
                result["error"] = repr(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    result["error_body"] = exc.read().decode(errors="replace")
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(
                json.dumps(
                    {
                        k: result.get(k)
                        for k in (
                            "label",
                            "role",
                            "task_id",
                            "completion_tokens",
                            "first_token_s",
                            "elapsed_s",
                            "decode_tok_s",
                            "e2e_tok_s",
                            "finish_reason",
                            "error",
                        )
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if result.get("error"):
                raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
