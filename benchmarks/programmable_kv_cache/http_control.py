# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate attached and independent native KV movement against real servers."""

import argparse
import json
import math
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests

from benchmarks.programmable_kv_cache.client import (
    answer_number,
    envelope,
    wait_operation,
)
from benchmarks.programmable_kv_cache.run_rollout_matrix import (
    dataset_identity,
    source_hashes,
)


def cached(response):
    return response["usage"]["prompt_tokens_details"]["cached_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine-id")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tier", choices=("mooncake", "cpu"), default="mooncake")
    parser.add_argument("--policy", action="store_true")
    parser.add_argument("--destination-url")
    parser.add_argument("--evaluation-samples", type=int, default=0)
    args = parser.parse_args()
    if args.destination_url and args.tier != "mooncake":
        parser.error("Cross-engine preparation requires Mooncake")
    if args.evaluation_samples < 0:
        parser.error("--evaluation-samples must be nonnegative")
    session = requests.Session()
    base = args.base_url.rstrip("/")
    run = uuid.uuid4().hex
    report = {
        "profile": "native_primitives",
        "actions": {},
        "inference": [],
        "source_hashes": source_hashes(),
        "dataset_identity": dataset_identity(args.dataset),
    }

    def get(path, base_url=base):
        response = session.get(base_url + path, timeout=30)
        response.raise_for_status()
        return response.json()

    def post(path, payload, base_url=base):
        response = session.post(base_url + path, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()

    engines = [
        e
        for e in get("/v1/kv/capabilities")["engines"]
        if args.engine_id is None or e["engine_id"] == args.engine_id
    ]
    assert len(engines) == 1, engines
    caps = engines[0]
    report["capabilities"] = caps
    invocation = {k: caps[k] for k in ("engine_id", "engine_epoch", "namespace")}

    def control(method, payload):
        return post(
            "/v1/kv/control",
            {"method": method, "invocation": invocation, "payload": payload},
        )

    def wait(name, *, acknowledge=True):
        result = wait_operation(control, run + "-" + name, acknowledge=acknowledge)
        report["actions"][name] = result
        return result

    model = get("/v1/models")["data"][0]["id"]
    train = pq.read_table(
        args.dataset / "main/train-00000-of-00001.parquet"
    ).to_pylist()
    text = "".join(
        f"Question: {x['question']}\nAnswer: {x['answer']}\n\n" for x in train[:8]
    )
    text += "Question: What is 1 + 1?\nAnswer:"
    tokens = post("/tokenize", {"model": model, "prompt": text})["tokens"]
    boundary = math.lcm(
        *(
            g["block_size"]
            for g in caps["cache_profile"]["groups"]
            if g["prefix_cacheable"]
        )
    )
    end = len(tokens) // boundary * boundary
    assert end > 0
    selector = {"kind": "request", "request_id": "$request", "end_token": end}

    def infer(hints=None, *, base_url=base, model_id=None):
        body = {
            "model": model_id or model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": 16,
            "return_token_ids": True,
            "cache_salt": run,
        }
        if hints is not None:
            body["kv_hints"] = hints
        if args.tier == "cpu":
            body["kv_transfer_params"] = {"max_load_tokens": 0}
        response = post("/v1/completions", body, base_url)
        report["inference"].append(response)
        return response

    report["passed"] = False
    try:
        first = infer(
            envelope(
                run + "-move",
                "vllm.move_to",
                {
                    "target": selector,
                    "src": {"tier": "gpu"},
                    "dst": {"tier": args.tier},
                },
            )
        )
        moved = wait("move")
        assert moved["status"] == "completed", moved
        assert moved["target_outcomes"] and all(
            "cold_copy_stored" in o["completed_steps"]
            and "gpu_copies_reclaimed" in o["completed_steps"]
            for o in moved["target_outcomes"]
        ), moved
        resumed = infer(
            envelope(
                run + "-copy",
                "vllm.copy_to",
                {
                    "target": selector,
                    "src": {"tier": args.tier},
                    "dst": {"tier": "gpu"},
                },
            )
        )
        copied = wait("copy")
        assert copied["status"] == "completed" and copied["target_outcomes"], copied
        assert all(
            o["transfer_id"] is not None
            and "gpu_content_registered" in o["completed_steps"]
            for o in copied["target_outcomes"]
        ), copied
        assert first["choices"][0]["token_ids"] == resumed["choices"][0]["token_ids"]
        assert first["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert resumed["usage"]["prompt_tokens_details"]["cached_tokens"] >= end
        target = {
            "kind": "hashes",
            "namespace": caps["namespace"],
            "blocks": [
                {
                    k: outcome[k]
                    for k in ("block_hash", "group_id", "start_token", "end_token")
                }
                for outcome in copied["target_outcomes"]
            ],
        }

        def action(name, kind, payload, *, acknowledge=True):
            message = envelope(run + "-" + name, kind, payload)
            control("submit", message)
            result = wait(name, acknowledge=acknowledge)
            assert result["status"] == "completed", result
            return result

        def movement(name, kind, source, destination):
            return action(
                name,
                kind,
                {
                    "target": target,
                    "src": {"tier": source},
                    "dst": {"tier": destination},
                },
            )

        priority = {
            "target": target,
            "location": "local_g1",
            "claim_id": run,
            "revision": 1,
            "value": 80,
            "ttl_seconds": 300,
        }
        action("priority", "vllm.set_priority", priority, acknowledge=False)
        assert control(
            "submit", envelope(run + "-priority", "vllm.set_priority", priority)
        )["replayed"]
        control("acknowledge", {"message_id": run + "-priority"})
        replay = session.post(
            base + "/v1/kv/control",
            json={
                "method": "submit",
                "invocation": invocation,
                "payload": envelope(run + "-priority", "vllm.set_priority", priority),
            },
            timeout=30,
        )
        assert replay.status_code == 409 and replay.json()["error"]["code"] == (
            "acknowledged"
        )
        report["acknowledged_replay"] = replay.json()
        action(
            "release",
            "vllm.set_priority",
            {
                **priority,
                "revision": 2,
                "value": None,
                "ttl_seconds": None,
            },
        )
        movement("independent-copy", "vllm.copy_to", "gpu", args.tier)
        assert cached(infer()) >= end
        movement("independent-move", "vllm.move_to", "gpu", args.tier)
        movement("independent-prepare", "vllm.copy_to", args.tier, "gpu")
        again = infer()
        assert cached(again) >= end
        assert again["choices"][0]["token_ids"] == first["choices"][0]["token_ids"]

        if args.destination_url:
            (peer,) = get("/v1/kv/capabilities", args.destination_url)["engines"]
            assert peer["namespace"] == caps["namespace"], (caps, peer)
            assert peer["engine_id"] != caps["engine_id"]
            assert peer["engine_epoch"] != caps["engine_epoch"]

            def peer_control(method, payload):
                return post(
                    "/v1/kv/control",
                    {
                        "method": method,
                        "invocation": {k: peer[k] for k in invocation},
                        "payload": payload,
                    },
                    args.destination_url,
                )

            message = run + "-peer-copy"
            peer_control(
                "submit",
                envelope(
                    message,
                    "vllm.copy_to",
                    {
                        "target": target,
                        "src": {"tier": "mooncake"},
                        "dst": {"tier": "gpu"},
                    },
                ),
            )
            ready = wait_operation(peer_control, message, acknowledge=True)
            assert ready["status"] == "completed", ready
            peer_model = get("/v1/models", args.destination_url)["data"][0]["id"]
            response = infer(base_url=args.destination_url, model_id=peer_model)
            assert cached(response) >= end
            assert (
                response["choices"][0]["token_ids"] == first["choices"][0]["token_ids"]
            )
            report["cross_engine"] = {"destination": peer, "readiness": ready}

        if args.policy:
            for name, signal in (
                ("policy-move", "tool_wait"),
                ("policy-copy", "resume"),
            ):
                receipt = control(
                    "submit",
                    envelope(
                        run + "-" + name,
                        "signal",
                        {"target": target, "signal": signal},
                        "1",
                    ),
                )
                parent = receipt["results"][0]
                assert (
                    parent["status"] == "completed" and parent["derived_operations"]
                ), parent
                children = [
                    wait_operation(
                        control,
                        child["message_id"],
                        child["action_id"],
                        acknowledge=True,
                    )
                    for child in parent["derived_operations"]
                ]
                assert all(child["status"] == "completed" for child in children), (
                    children
                )
                report["actions"][name] = {"parent": parent, "children": children}
                control("acknowledge", {"message_id": parent["message_id"]})
            again = infer()
            assert again["usage"]["prompt_tokens_details"]["cached_tokens"] >= end
            assert again["choices"][0]["token_ids"] == first["choices"][0]["token_ids"]
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
                body["cache_salt"] = (
                    f"{run}-question-{index}" if mode == "recompute" else run
                )
                if args.tier == "cpu":
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
                "prepared",
            ):
                rows = []
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=8) as workers:
                    for offset in range(0, len(indices), 8):
                        if mode == "prepared":
                            infer()
                            name = f"evaluation-demote-{offset}"
                            movement(name, "vllm.move_to", "gpu", args.tier)
                            movement(
                                f"evaluation-copy-{offset}",
                                "vllm.copy_to",
                                args.tier,
                                "gpu",
                            )
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
            prefix_tokens = post("/tokenize", {"model": model, "prompt": shared_text})[
                "tokens"
            ]
            reusable = len(prefix_tokens) // boundary * boundary
            assert reusable > 0
            assert all(row["cached_tokens"] >= reusable for row in moved["rows"])
            assert baseline["length_limited"] <= 0.01 * args.evaluation_samples
            assert moved["length_limited"] <= 0.01 * args.evaluation_samples
            assert min(baseline["accuracy"], moved["accuracy"]) >= 0.8
            assert baseline["accuracy"] - moved["accuracy"] <= 0.01
            evaluation["answer_mismatches"] = sum(
                a["predicted"] != b["predicted"]
                for a, b in zip(baseline["rows"], moved["rows"], strict=True)
            )

        report["final_capacity"] = control("capacity", {})
        assert report["final_capacity"]["records"] == 0
        assert not report["final_capacity"].get("pending_transfers")
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        session.close()
    print(
        json.dumps(
            {
                "passed": True,
                "cached_tokens": resumed["usage"]["prompt_tokens_details"][
                    "cached_tokens"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
