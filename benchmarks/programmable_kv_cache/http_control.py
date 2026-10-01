# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate HTTP control against a real server and observable GPU cache reuse."""

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests

from vllm import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    get_none_hash_seed,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-request-associations", action="store_true")
    args = parser.parse_args()
    session = requests.Session()
    run_id = uuid.uuid4().hex
    report: dict[str, Any] = {"base_url": args.base_url, "actions": {}, "inference": []}

    def post(path, body):
        response = session.post(args.base_url + path, json=body, timeout=180)
        response.raise_for_status()
        return response.json()

    caps = session.get(args.base_url + "/v1/kv/capabilities", timeout=30).json()[
        "engines"
    ]
    assert len(caps) == 1
    caps = caps[0]
    report["capabilities"] = caps
    invocation = {key: caps[key] for key in ("engine_id", "engine_epoch", "namespace")}

    def envelope(name, kind, payload, version="1"):
        return {
            "protocol_version": "0.1",
            "message_id": run_id + "-" + name,
            "actions": [
                {
                    "action_id": "action",
                    "action_type": kind,
                    "action_version": version,
                    "payload": payload,
                }
            ],
        }

    def control(method, payload):
        return post(
            "/v1/kv/control",
            {"method": method, "invocation": invocation, "payload": payload},
        )

    def submit(name, kind, payload, version="1"):
        result = control("submit", envelope(name, kind, payload, version))
        report["actions"][name] = result
        queried = operation(name)
        assert queried == result["results"][0]
        return result

    def operation(name):
        return control(
            "operation", {"message_id": run_id + "-" + name, "action_id": "action"}
        )

    model = session.get(args.base_url + "/v1/models", timeout=30).json()["data"][0][
        "id"
    ]
    train = pq.read_table(
        args.dataset / "main/train-00000-of-00001.parquet"
    ).to_pylist()
    text = "".join(
        f"Question: {x['question']}\nAnswer: {x['answer']}\n\n" for x in train[:8]
    )
    text += "Question: What is 1 + 1?\nAnswer:"
    tokens = post("/tokenize", {"model": model, "prompt": text})["tokens"]

    def infer(hints=None):
        body = {
            "model": model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": 16,
            "return_token_ids": True,
        }
        if hints is not None:
            body["kv_hints"] = hints
        response = post("/v1/completions", body)
        assert response["choices"][0]["prompt_token_ids"] == tokens
        report["inference"].append(response)
        return response

    profile = caps["cache_profile"]
    groups = [g for g in profile["groups"] if g["prefix_cacheable"]]
    assert groups
    boundary = math.lcm(*(g["block_size"] for g in groups))
    end = len(tokens) // boundary * boundary
    assert end > 0
    hash_fn = get_hash_fn_by_name(profile["hash_algorithm"])
    init_none_hash(hash_fn)
    assert get_none_hash_seed() == profile["hash_seed"], (
        "Client and server hash seeds differ"
    )
    request = Request(
        "hash",
        tokens,
        SamplingParams(max_tokens=16),
        None,
        block_hasher=get_request_block_hasher(profile["hash_block_size"], hash_fn),
    )
    block_hash = request.block_hashes[end // profile["hash_block_size"] - 1].hex()
    target = {
        "kind": "hashes",
        "namespace": caps["namespace"],
        "blocks": [
            {
                "block_hash": block_hash,
                "group_id": group["group_id"],
                "start_token": (end - 1) // group["block_size"] * group["block_size"],
                "end_token": end,
            }
            for group in groups
        ],
    }
    cleaned = submit("initial-evict", "evict", {"target": target})["results"][0]
    assert cleaned["status"] in ("completed", "missing"), cleaned
    cold, warm = infer(), infer()

    def cached(response):
        return response["usage"]["prompt_tokens_details"]["cached_tokens"]

    assert cached(cold) == 0
    assert cached(warm) > 0
    retained_payload = {
        "target": target,
        "owner_id": "validation",
        "declaration_id": "d",
        "priority": 80,
        "ttl_seconds": 300,
    }
    retained = submit("retain", "retain", retained_payload)
    assert retained["results"][0]["status"] == "completed"
    assert control("submit", envelope("retain", "retain", retained_payload))["replayed"]
    release = submit(
        "release", "release", {"owner_id": "validation", "declaration_id": "d"}
    )
    assert release["results"][0]["status"] == "completed"
    for continuation in ("main", "branch"):
        associated = submit(
            "associate-" + continuation,
            "associate",
            {
                "session_id": "validation",
                "continuation_id": continuation,
                "expected_revision": 0,
                "target": target,
            },
        )
        assert associated["results"][0]["status"] == "completed"
    state = control("associations", {"session_id": "validation"})
    assert len(state["items"]) == 2
    selected = next(s for s in state["items"] if s["continuation_id"] == "main")
    rewrite = submit(
        "rewrite",
        "associate",
        {
            "session_id": "validation",
            "continuation_id": "main",
            "expected_revision": selected["revision"],
            "target": {**target, "blocks": []},
        },
    )
    assert rewrite["results"][0]["status"] == "failed"
    assert "append-only" in rewrite["results"][0]["reason"]
    assert control("associations", {"session_id": "validation"}) == state
    eviction = submit("evict", "evict", {"target": target})["results"][0]
    assert eviction["status"] == "completed", eviction
    assert eviction["affected_blocks"] >= len(groups)
    signal_payload = {
        "target": {"kind": "session", "session_id": "validation"},
        "signal": "tool_wait",
    }
    after = infer(envelope("attached", "signal", signal_payload))
    assert cached(after) < cached(warm)
    assert (
        cold["choices"][0]["token_ids"]
        == warm["choices"][0]["token_ids"]
        == after["choices"][0]["token_ids"]
    )
    attached = operation("attached")
    assert attached["status"] == "unconsumed"
    assert attached["binding_id"]
    report["actions"]["attached"] = attached
    chat_hint = envelope("chat-attached", "signal", signal_payload)
    chat = post(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0,
            "max_tokens": 16,
            "kv_hints": chat_hint,
        },
    )
    assert chat["choices"]
    chat_result = operation("chat-attached")
    assert chat_result["status"] == "unconsumed" and chat_result["binding_id"]
    report["chat"] = chat
    report["actions"]["chat-attached"] = chat_result
    streaming = session.post(
        args.base_url + "/v1/completions",
        json={
            "model": model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": 8,
            "stream": True,
            "kv_hints": envelope("stream-attached", "signal", signal_payload),
        },
        timeout=180,
        stream=True,
    )
    streaming.raise_for_status()
    chunks = list(streaming.iter_lines(decode_unicode=True))
    assert "data: [DONE]" in chunks
    stream_result = operation("stream-attached")
    assert stream_result["status"] == "unconsumed" and stream_result["binding_id"]
    report["stream_chunks"] = chunks
    report["actions"]["stream-attached"] = stream_result
    for options in ({"n": 2}, {"use_beam_search": True}, {"prompt": [tokens, tokens]}):
        response = session.post(
            args.base_url + "/v1/completions",
            json={
                "model": model,
                "prompt": tokens,
                "max_tokens": 1,
                "kv_hints": chat_hint,
                **options,
            },
            timeout=30,
        )
        assert response.status_code == 400, response.text
    for name, kind, version in (("pin", "pin", "1"), ("version", "signal", "future")):
        result = submit(name, kind, signal_payload, version)
        assert result["results"][0]["status"] == "unsupported"
    movement = submit("movement", "prefetch", {"target": target, "source": "cpu"})
    assert movement["results"][0]["status"] == "unsupported"
    for changes, status in (
        ({"engine_id": "wrong"}, 404),
        ({"engine_epoch": "old"}, 409),
        ({"namespace": "wrong"}, 400),
    ):
        response = session.post(
            args.base_url + "/v1/kv/control",
            json={
                "method": "associations",
                "invocation": {**invocation, **changes},
                "payload": {},
            },
            timeout=30,
        )
        assert response.status_code == status, response.text
    finished = submit(
        "finish",
        "finish_task",
        {"session_id": "validation", "expected_revision": state["revision"]},
    )
    assert finished["results"][0]["status"] == "completed"
    assert control("associations", {})["items"] == []
    if args.validate_request_associations:
        tracked = "requests-" + run_id

        def associated_inference(
            name, continuation, input_tokens, revision=0, growth=False
        ):
            hints = envelope(
                name,
                "associate",
                {
                    "session_id": tracked,
                    "continuation_id": continuation,
                    "expected_revision": revision,
                    "target": {"kind": "request", "request_id": "$request"},
                    "observe_growth": growth,
                },
            )
            response = post(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": input_tokens,
                    "max_tokens": 16,
                    "temperature": 0,
                    "return_token_ids": True,
                    "kv_hints": hints,
                },
            )
            result = operation(name)
            assert result["status"] in ("completed", "partial"), result
            report["actions"][name] = result
            report.setdefault("request_inference", []).append(response)
            rows = control("associations", {"session_id": tracked})["items"]
            state = next(x for x in rows if x["continuation_id"] == continuation)
            return response, state

        snapshot_response, snapshot = associated_inference(
            "snapshot-request", "snapshot", tokens
        )
        assert snapshot["checkpoint"]["token_count"] == len(tokens)
        live_response, live = associated_inference(
            "live-request", "live", tokens, growth=True
        )
        history = tokens + live_response["choices"][0]["token_ids"]
        assert live["checkpoint"]["token_count"] == len(history)
        followed = submit(
            "follow-request",
            "retain",
            {
                "owner_id": tracked,
                "declaration_id": "following",
                "priority": 90,
                "ttl_seconds": 300,
                "scope": "follow_growth",
                "target": {
                    "kind": "session",
                    "session_id": tracked,
                    "continuation_id": "live",
                },
            },
        )
        assert followed["results"][0]["status"] in ("completed", "partial")
        tool_tokens = post(
            "/tokenize", {"model": model, "prompt": "\nTool result: 2. Continue.\n"}
        )["tokens"]
        resumed_input = history + tool_tokens
        resumed_response, resumed = associated_inference(
            "resume-request",
            "live",
            resumed_input,
            revision=live["revision"],
            growth=True,
        )
        assert resumed["checkpoint"]["token_count"] == len(resumed_input) + len(
            resumed_response["choices"][0]["token_ids"]
        )
        _, hit = associated_inference("cache-hit-request", "hit", tokens)
        assert {x["block_hash"] for x in hit["blocks"]} == {
            x["block_hash"] for x in snapshot["blocks"]
        }
        bad_tokens = tokens.copy()
        bad_tokens[0] = (bad_tokens[0] + 1) % 1000
        invalid_hint = envelope(
            "request-rewrite",
            "associate",
            {
                "session_id": tracked,
                "continuation_id": "live",
                "expected_revision": resumed["revision"],
                "target": {"kind": "request", "request_id": "$request"},
            },
        )
        post(
            "/v1/completions",
            {
                "model": model,
                "prompt": bad_tokens,
                "max_tokens": 1,
                "temperature": 0,
                "kv_hints": invalid_hint,
            },
        )
        rejected = operation("request-rewrite")
        assert (
            rejected["status"] == "failed"
            and "new continuation ID" in rejected["reason"]
        )
        report["actions"]["request-rewrite"] = rejected
        final = control("associations", {"session_id": tracked})
        assert (
            next(x for x in final["items"] if x["continuation_id"] == "live") == resumed
        )
        report["request_associations"] = final
        request_finished = submit(
            "finish-requests",
            "finish_task",
            {
                "session_id": tracked,
                "expected_revision": final["revision"],
            },
        )
        assert request_finished["results"][0]["status"] == "completed"
        assert control("associations", {})["items"] == []
    report["operation_events"] = control("events", {"stream": "operation"})
    report["association_events"] = control("events", {"stream": "association"})
    report["cached_tokens"] = [cached(x) for x in report["inference"]]
    report["cache_transfer_tested"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cached_tokens": report["cached_tokens"],
                "eviction": eviction,
                "passed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
