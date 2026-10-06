# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""R1 MOVE at completion, R2 COPY at admission, using real Mooncake and inference."""

import argparse
import json
import math
import uuid
from pathlib import Path

import pyarrow.parquet as pq
import requests

if __package__:
    from .client import envelope, wait_operation
else:
    from client import envelope, wait_operation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tier", choices=("mooncake", "cpu"), default="mooncake")
    parser.add_argument("--policy", action="store_true")
    args = parser.parse_args()
    session = requests.Session()
    base = args.base_url.rstrip("/")
    run = uuid.uuid4().hex
    report = {"profile": "request_attached_primitives", "actions": {}, "inference": []}

    def get(path):
        response = session.get(base + path, timeout=30)
        response.raise_for_status()
        return response.json()

    def post(path, payload):
        response = session.post(base + path, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()

    engines = [
        e
        for e in get("/v1/kv/capabilities")["engines"]
        if e["engine_id"] == args.engine_id
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

    def wait(name):
        result = wait_operation(control, run + "-" + name)
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

    def infer(hints=None):
        body = {
            "model": model,
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
        response = post("/v1/completions", body)
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
        if args.policy:
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
                    wait_operation(control, child["message_id"], child["action_id"])
                    for child in parent["derived_operations"]
                ]
                assert all(child["status"] == "completed" for child in children), (
                    children
                )
                report["actions"][name] = {"parent": parent, "children": children}
            again = infer()
            assert again["usage"]["prompt_tokens_details"]["cached_tokens"] >= end
            assert again["choices"][0]["token_ids"] == first["choices"][0]["token_ids"]
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
