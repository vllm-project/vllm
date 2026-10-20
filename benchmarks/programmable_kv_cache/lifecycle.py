# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real shared branches, independent TTLs, resume races and cancellation."""

import argparse
import asyncio
import gzip
import json
import uuid
from pathlib import Path

from benchmarks.programmable_kv_cache.client import envelope
from benchmarks.programmable_kv_cache.rollout import RolloutClient
from benchmarks.programmable_kv_cache.run_rollout_matrix import source_hashes


async def run(args):
    client = RolloutClient(args.base_url, "priority")
    report = {
        "passed": False,
        "requests": client.rows,
        "controls": client.controls,
        "source_hashes": source_hashes(),
    }
    try:
        await client.initialize()
        with gzip.open(args.trace, "rt") as file:
            prompt = json.load(file)["requests"][0]["prompt_ids"]
        prompt = prompt[: client.boundary + 128]
        root = await client.generate(
            "root", prompt, {"temperature": 0, "max_tokens": 16}
        )
        root_target = client.targets["root"]
        parent = {
            "session_id": root_target["session_id"],
            "continuation_id": "turn-0",
            "end_token": client.boundary,
        }
        for name, tail in (("b", "Branch B"), ("c", "Branch C")):
            response = await client.http.post(
                "/tokenize", json={"model": client.model, "prompt": tail}
            )
            response.raise_for_status()
            branch = await client.generate(
                name,
                prompt + root["token_ids"] + response.json()["tokens"],
                {"temperature": 0, "max_tokens": 16},
                task_parent_id=root_target["session_id"],
                parent_prefix=parent,
            )
            assert (
                branch["usage"]["prompt_tokens_details"]["cached_tokens"]
                >= client.boundary
            )
        shared = await client.generate(
            "external", prompt, {"temperature": 0, "max_tokens": 16}
        )
        assert (
            shared["usage"]["prompt_tokens_details"]["cached_tokens"] >= client.boundary
        )
        external = client.targets["external"]
        residency = await client.call("residency", {"target": external})
        blocks = [
            {
                key: row[key]
                for key in ("block_hash", "group_id", "start_token", "end_token")
            }
            for row in residency["items"]
            if row["resident_copies"]
        ]
        assert blocks
        target = {
            "kind": "hashes",
            "namespace": client.invocation["namespace"],
            "blocks": blocks,
            "start_token": 0,
            "end_token": client.boundary,
        }
        for owner, priority, ttl in (
            (root_target["session_id"], 90, 60),
            (external["session_id"], 20, 2),
        ):
            result = await client.action(
                "retain",
                {
                    "target": target,
                    "owner_id": owner,
                    "declaration_id": "shared",
                    "priority": priority,
                    "ttl_seconds": ttl,
                },
                version="1",
            )
            assert result["status"] == "completed", result

        async def priorities():
            current = await client.call("residency", {"target": target})
            report.setdefault("residency", []).append(current)
            return {p for row in current["items"] for p in row["eviction_priorities"]}

        assert await priorities() == {90}
        await client.finish("root")
        snapshot = await client.call("associations", {})
        report["after_task_tree_cleanup"] = snapshot
        names = {row["session_id"] for row in snapshot["items"]}
        assert external["session_id"] in names
        assert root_target["session_id"] not in names
        assert client.targets["b"]["session_id"] not in names
        assert client.targets["c"]["session_id"] not in names
        assert await priorities() == {20}
        await asyncio.sleep(2.1)
        assert await priorities() == {0}

        stored = await client.action(
            "vllm.copy_to",
            {
                "target": target,
                "src": {"tier": "gpu"},
                "dst": {"tier": "mooncake"},
            },
        )
        assert stored["status"] == "completed", stored
        evicted = await client.action("evict", {"target": target}, version="1")
        assert evicted["status"] == "completed", evicted
        if args.capacity_deferral:
            payload = {
                "target": target,
                "src": {"tier": "mooncake"},
                "dst": {"tier": "gpu"},
            }
            attempts = []
            for _ in range(len(blocks) + 1):
                result = await client.action("vllm.copy_to", payload)
                attempts.append(result)
                if result["status"] == "completed":
                    break
                assert result["status"] == "partial" and result["blocked_targets"] > 0
            assert attempts[0]["status"] == "partial"
            assert attempts[-1]["status"] == "completed"
            report["capacity_retry"] = attempts
            resumed = await client.generate(
                "capacity-resumed", prompt, {"temperature": 0, "max_tokens": 16}
            )
            assert (
                resumed["usage"]["prompt_tokens_details"]["cached_tokens"]
                >= client.boundary
            )
            assert resumed["token_ids"] == root["token_ids"]
            await client.action("evict", {"target": target}, version="1")
            partial = await client.action("vllm.copy_to", payload)
            assert partial["status"] == "partial" and partial["blocked_targets"] > 0
            fallback = await client.generate(
                "capacity-fallback", prompt, {"temperature": 0, "max_tokens": 16}
            )
            assert fallback["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
            assert fallback["token_ids"] == root["token_ids"]
            report["capacity_fallback"] = partial
            for identifier in tuple(client.targets):
                await client.finish(identifier)
            report["final_capacity"] = await client.wait_control_idle()
            report["final_associations"] = await client.call("associations", {})
            assert report["final_capacity"]["records"] == 0
            assert not report["final_associations"]["items"]
            report["passed"] = True
            return
        message = uuid.uuid4().hex
        command = envelope(
            message,
            "vllm.copy_to",
            {
                "target": target,
                "src": {"tier": "mooncake"},
                "dst": {"tier": "gpu"},
            },
        )
        submitted = await client.call("submit", command)
        assert (await client.call("submit", command))["replayed"]
        resumed = await client.generate(
            "raced", prompt, {"temperature": 0, "max_tokens": 16}
        )
        prepared = await client.wait(message)
        assert prepared["status"] == "completed"
        assert (
            resumed["usage"]["prompt_tokens_details"]["cached_tokens"]
            >= client.boundary
        )
        assert resumed["token_ids"] == root["token_ids"]
        report["resume_race"] = {"submission": submitted, "completion": prepared}

        await client.action("evict", {"target": target}, version="1")
        message = uuid.uuid4().hex
        receipt = await client.call(
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
        operation = receipt["results"][0]
        cancelled = await client.call(
            "cancel",
            {
                key: operation[key]
                for key in ("message_id", "action_id", "operation_id")
            },
        )
        assert cancelled["status"] in ("cancelled", "completed"), cancelled
        report["cancellation"] = cancelled
        await client.wait(message)
        fallback = await client.generate(
            "fallback", prompt, {"temperature": 0, "max_tokens": 16}
        )
        assert fallback["token_ids"] == root["token_ids"]
        absent = dict(target, blocks=[dict(b, block_hash="ff" * 32) for b in blocks])
        missing = await client.action(
            "vllm.copy_to",
            {
                "target": absent,
                "src": {"tier": "mooncake"},
                "dst": {"tier": "gpu"},
            },
        )
        assert missing["status"] == "missing", missing
        report["missing_content"] = missing
        for identifier in tuple(client.targets):
            await client.finish(identifier)
        report["final_capacity"] = await client.wait_control_idle()
        report["final_associations"] = await client.call("associations", {})
        assert report["final_capacity"]["records"] == 0
        assert not report["final_associations"]["items"]
        report["passed"] = True
        print(
            json.dumps({"passed": True, "branches": 3, "shared_priority": [90, 20, 0]})
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            await client.close()
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(args.output, "wt") as file:
                json.dump(report, file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capacity-deferral",
        action="store_true",
        help="Check partial loads and retry on a server with a one-block load budget",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
