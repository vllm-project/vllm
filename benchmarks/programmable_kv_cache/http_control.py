# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate HTTP control against a real server and observable GPU cache reuse."""

import argparse
import json
import math
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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

if __package__:
    from .client import ACTIVE, wait_operation
    from .client import envelope as make_envelope
    from .evaluate import answer_number
else:
    from client import ACTIVE, wait_operation
    from client import envelope as make_envelope
    from evaluate import answer_number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-request-associations", action="store_true")
    parser.add_argument("--validate-movement", choices=("cpu", "mooncake"))
    parser.add_argument("--evaluation-samples", type=int, default=0)
    parser.add_argument("--destination-url")
    parser.add_argument("--primitive-actions", action="store_true")
    args = parser.parse_args()
    if args.evaluation_samples and not args.validate_movement:
        parser.error("Movement evaluation requires --validate-movement")
    if args.destination_url and args.validate_movement != "mooncake":
        parser.error("Cross-engine preparation requires Mooncake movement")
    session = requests.Session()
    run_id = uuid.uuid4().hex
    report: dict[str, Any] = {"base_url": args.base_url, "actions": {}, "inference": []}

    def post(path, body, *, base_url=None):
        response = session.post(
            (base_url or args.base_url) + path, json=body, timeout=180
        )
        response.raise_for_status()
        return response.json()

    caps = session.get(args.base_url + "/v1/kv/capabilities", timeout=30).json()[
        "engines"
    ]
    assert len(caps) == 1
    caps = caps[0]
    report["capabilities"] = caps
    invocation = {key: caps[key] for key in ("engine_id", "engine_epoch", "namespace")}

    priority_targets = {}
    report["action_profile"] = "rfc59708" if args.primitive_actions else "legacy"

    def envelope(name, kind, payload, version="1"):
        if args.primitive_actions and version == "1":
            if kind == "retain":
                key = payload["owner_id"], payload["declaration_id"]
                priority_targets[key] = payload["target"]
                kind, version, payload = (
                    "vllm.set_priority",
                    "1.0",
                    {
                        "target": payload["target"],
                        "location": "local_g1",
                        "claim_id": "/".join(key),
                        "revision": 1,
                        "value": payload["priority"],
                        "ttl_seconds": payload["ttl_seconds"],
                    },
                )
            elif kind == "release":
                key = payload["owner_id"], payload["declaration_id"]
                kind, version, payload = (
                    "vllm.set_priority",
                    "1.0",
                    {
                        "target": priority_targets[key],
                        "location": "local_g1",
                        "claim_id": "/".join(key),
                        "revision": 2,
                        "value": None,
                    },
                )
            elif kind in ("demote", "prefetch", "share"):
                primitive = "vllm.move_to" if kind == "demote" else "vllm.copy_to"
                source = payload["source"] if kind == "prefetch" else "gpu"
                destination = (
                    "gpu"
                    if kind == "prefetch"
                    else payload.get("destination", "mooncake")
                )
                kind, version, payload = (
                    primitive,
                    "1.0",
                    {
                        "target": payload["target"],
                        "src": {"tier": source},
                        "dst": {"tier": destination},
                    },
                )
        return make_envelope(run_id + "-" + name, kind, payload, version)

    def control(method, payload):
        return post(
            "/v1/kv/control",
            {"method": method, "invocation": invocation, "payload": payload},
        )

    def submit(name, kind, payload, version="1", *, call=control):
        result = call("submit", envelope(name, kind, payload, version))
        report["actions"][name] = result
        queried = operation(name, call=call)
        initial = result["results"][0]
        assert queried["operation_id"] == initial["operation_id"]
        assert queried["sequence"] >= initial["sequence"]
        if initial["status"] not in ACTIVE:
            assert queried == initial
        return result

    def operation(name, *, call=control):
        return call(
            "operation", {"message_id": run_id + "-" + name, "action_id": "action"}
        )

    def wait_for_operation(name, *, call=control):
        result = wait_operation(call, run_id + "-" + name)
        report["actions"][name + "-terminal"] = result
        return result

    def reclaim_after_demote(name, target, result):
        assert result["status"] in ("completed", "partial"), result
        assert "cold_copy_stored" in result["completed_steps"], result
        if not result["blocked_targets"]:
            return
        deadline = time.monotonic() + 120
        attempt = 0
        while True:
            evicted = submit(f"{name}-reclaim-{attempt}", "evict", {"target": target})[
                "results"
            ][0]
            if evicted["status"] in ("completed", "missing"):
                return
            assert time.monotonic() < deadline, evicted
            attempt += 1
            time.sleep(0.05)

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

    def infer(hints=None, *, base_url=None, model_id=None):
        body = {
            "model": model_id or model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": 16,
            "return_token_ids": True,
        }
        if hints is not None:
            body["kv_hints"] = hints
        if args.validate_movement == "cpu":
            body["kv_transfer_params"] = {"max_load_tokens": 0}
        response = post("/v1/completions", body, base_url=base_url)
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
        cache_salt=json.dumps(
            [profile["request_cache_salt_prefix"], None],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if profile.get("request_cache_salt_prefix") is not None
        else None,
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
    if args.validate_movement:
        tier = args.validate_movement
        assert any(
            a["action_type"] == "prefetch" and tier in a["tiers"]
            for a in caps["actions"]
        ), caps
        before_movement = infer()
        assert cached(before_movement) == cached(warm)
        if tier == "mooncake":
            peer = None
            if args.destination_url:
                (peer,) = session.get(
                    args.destination_url + "/v1/kv/capabilities", timeout=30
                ).json()["engines"]
                assert peer["namespace"] == caps["namespace"], (caps, peer)
                assert peer["engine_id"] != caps["engine_id"]
                assert peer["engine_epoch"] != caps["engine_epoch"]
            submit(
                "share",
                "share",
                {
                    "target": target,
                    "destination_engine_id": peer["engine_id"]
                    if peer
                    else "validation-peer",
                },
            )
            shared = wait_for_operation("share")
            assert shared["status"] == "completed", shared
            assert shared["completed_steps"] == ["cold_copy_stored"]
            if peer:
                peer_invocation = {
                    key: peer[key] for key in ("engine_id", "engine_epoch", "namespace")
                }

                def peer_control(method, payload):
                    return post(
                        "/v1/kv/control",
                        {
                            "method": method,
                            "invocation": peer_invocation,
                            "payload": payload,
                        },
                        base_url=args.destination_url,
                    )

                submit("peer-evict", "evict", {"target": target}, call=peer_control)
                submit(
                    "peer-prefetch",
                    "prefetch",
                    {
                        "target": target,
                        "source": "mooncake",
                    },
                    call=peer_control,
                )
                ready = wait_for_operation("peer-prefetch", call=peer_control)
                assert ready["status"] == "completed", ready
                peer_model = session.get(
                    args.destination_url + "/v1/models", timeout=30
                ).json()["data"][0]["id"]
                peer_response = infer(
                    base_url=args.destination_url, model_id=peer_model
                )
                assert cached(peer_response) == cached(warm)
                assert (
                    peer_response["choices"][0]["token_ids"]
                    == cold["choices"][0]["token_ids"]
                )
                report["cross_engine"] = {"destination": peer, "readiness": ready}
            assert cached(infer()) == cached(warm)
        submit("demote", "demote", {"target": target, "destination": tier})
        demoted = wait_for_operation("demote")
        reclaim_after_demote("demote", target, demoted)
        # Only control submissions and queries occur between Demote and readiness.
        submit("prefetch", "prefetch", {"target": target, "source": tier})
        prepared = wait_for_operation("prefetch")
        assert prepared["status"] == "completed", prepared
        assert "gpu_content_registered" in prepared["completed_steps"], prepared
        resumed = infer()
        assert cached(resumed) == cached(warm)
        assert resumed["choices"][0]["token_ids"] == cold["choices"][0]["token_ids"]
        report["movement"] = {"tier": tier, "demote": demoted, "prefetch": prepared}
        missing_target = {
            **target,
            "blocks": [{**b, "block_hash": "00" * 32} for b in target["blocks"]],
        }
        submit(
            "missing-prefetch", "prefetch", {"target": missing_target, "source": tier}
        )
        missing = wait_for_operation("missing-prefetch")
        assert missing["status"] in ("failed", "missing"), missing
        if args.evaluation_samples:
            test = pq.read_table(
                args.dataset / "main/test-00000-of-00001.parquet"
            ).to_pylist()
            assert 0 < args.evaluation_samples <= len(test)
            indices = random.Random(0).sample(range(len(test)), args.evaluation_samples)
            shared_text = "".join(
                f"Question: {x['question']}\nAnswer: {x['answer']}\n\n"
                for x in train[:8]
            )

            def evaluate_question(index, mode):
                body = {
                    "model": model,
                    "prompt": shared_text
                    + f"Question: {test[index]['question']}\nAnswer:",
                    "temperature": 0,
                    "max_tokens": 1024,
                    "stop": ["Question:", "Assistant:", "<|separator|>"],
                    "return_token_ids": True,
                }
                if mode == "recompute":
                    body["cache_salt"] = f"{run_id}-question-{index}"
                if tier == "cpu":
                    body["kv_transfer_params"] = {"max_load_tokens": 0}
                response = requests.post(
                    args.base_url + "/v1/completions",
                    json=body,
                    timeout=600,
                )
                response.raise_for_status()
                output = response.json()
                choice = output["choices"][0]
                expected = answer_number(test[index]["answer"])
                predicted = answer_number(choice["text"])
                return {
                    "index": index,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": expected is not None and predicted == expected,
                    "text": choice["text"],
                    "token_ids": choice["token_ids"],
                    "finish_reason": choice["finish_reason"],
                    "cached_tokens": cached(output),
                }

            evaluation = {"indices": indices, "seed": 0, "num_fewshot": 8, "runs": {}}
            report["evaluation"] = evaluation
            for mode in (
                "recompute",
                "mooncake_prefetch" if tier == "mooncake" else "cpu_prefetch",
            ):
                rows = []
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=8) as workers:
                    for offset in range(0, len(indices), 8):
                        if mode != "recompute":
                            infer()
                            name = f"evaluation-demote-{offset}"
                            submit(
                                name, "demote", {"target": target, "destination": tier}
                            )
                            result = wait_for_operation(name)
                            reclaim_after_demote(name, target, result)
                            name = f"evaluation-prefetch-{offset}"
                            submit(name, "prefetch", {"target": target, "source": tier})
                            result = wait_for_operation(name)
                            assert result["status"] == "completed", result
                        rows.extend(
                            workers.map(
                                lambda index, mode=mode: evaluate_question(index, mode),
                                indices[offset : offset + 8],
                            )
                        )
                result = {
                    "seconds": time.perf_counter() - started,
                    "accuracy": sum(r["correct"] for r in rows) / len(rows),
                    "length_limited": sum(r["finish_reason"] == "length" for r in rows),
                    "rows": rows,
                }
                evaluation["runs"][mode] = result
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "accuracy": result["accuracy"],
                            "samples": len(rows),
                        }
                    ),
                    flush=True,
                )
            baseline, moved = evaluation["runs"].values()
            assert all(row["cached_tokens"] == 0 for row in baseline["rows"])
            assert all(row["cached_tokens"] >= end for row in moved["rows"])
            assert baseline["length_limited"] <= 0.01 * args.evaluation_samples
            assert moved["length_limited"] <= 0.01 * args.evaluation_samples
            assert min(baseline["accuracy"], moved["accuracy"]) >= 0.8
            assert baseline["accuracy"] - moved["accuracy"] <= 0.01
            evaluation["answer_mismatches"] = sum(
                a["predicted"] != b["predicted"]
                for a, b in zip(baseline["rows"], moved["rows"], strict=True)
            )
    elif not any(a["action_type"] == "prefetch" for a in caps["actions"]):
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
    report["cache_transfer_tested"] = bool(args.validate_movement)
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
