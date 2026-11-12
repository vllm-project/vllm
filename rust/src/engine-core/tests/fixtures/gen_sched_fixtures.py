# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate differential fixtures for the Rust scheduler leaf components.

Runs random operation sequences against the Python reference
(`vllm.v1.core.sched.request_queue`, `vllm.v1.core.encoder_cache_manager`,
and CPython's `set` of ints) and records every observable result, including
the internal heap / deque order and set iteration order.

Usage (from the repo root):
    .venv/bin/python rust/src/engine-core/tests/fixtures/gen_sched_fixtures.py
"""

import json
import random
from pathlib import Path

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
)
from vllm.v1.request import Request

OUT_DIR = Path(__file__).parent
ID_CHARS = ["a", "b", "z", "A", "0", "_", "é", "中", "😀"]


def _random_request_ids(rng: random.Random, n: int) -> list[str]:
    ids: set[str] = set()
    while len(ids) < n:
        ids.add("".join(rng.choices(ID_CHARS, k=rng.randint(0, 3))))
    out = sorted(ids)
    rng.shuffle(out)
    return out


def gen_queue_case(rng: random.Random, policy: str, num_ops: int) -> dict:
    n = rng.randint(1, 24)
    ids = _random_request_ids(rng, n)
    arrivals = [0.0, 1.0, 1.5, 2.0, -0.5]
    reqs = [
        Request(
            request_id=ids[i],
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            arrival_time=rng.choice(arrivals),
            priority=rng.randint(-2, 2),
        )
        for i in range(n)
    ]
    initial = [
        {"id": r.request_id, "priority": r.priority, "arrival_time": r.arrival_time}
        for r in reqs
    ]
    index = {id(r): i for i, r in enumerate(reqs)}
    queue = PriorityRequestQueue() if policy == "priority" else FCFSRequestQueue()
    present: list[int] = []

    def internal() -> list[int]:
        items = queue._heap if policy == "priority" else list(queue)
        return [index[id(r)] for r in items]

    ops = []
    for _ in range(num_ops):
        absent = [i for i in range(n) if i not in present]
        choices = ["iter", "set_arrival", "set_priority"]
        if absent:
            choices += ["add"] * 4 + ["prepend"] * 2 + ["prepend_many"]
        if present:
            choices += ["pop"] * 2 + ["peek", "remove", "remove_many"]
        op = rng.choice(choices)
        result = None
        if op == "add":
            i = rng.choice(absent)
            queue.add_request(reqs[i])
            present.append(i)
            args = [i]
        elif op == "prepend":
            i = rng.choice(absent)
            queue.prepend_request(reqs[i])
            present.append(i)
            args = [i]
        elif op == "prepend_many":
            many = rng.sample(absent, rng.randint(1, min(4, len(absent))))
            queue.prepend_requests([reqs[i] for i in many])
            present.extend(many)
            args = many
        elif op == "pop":
            result = index[id(queue.pop_request())]
            present.remove(result)
            args = []
        elif op == "peek":
            result = index[id(queue.peek_request())]
            args = []
        elif op == "remove":
            i = rng.choice(present)
            queue.remove_request(reqs[i])
            present.remove(i)
            args = [i]
        elif op == "remove_many":
            many = rng.sample(present, rng.randint(1, len(present)))
            # Also pass requests that are not queued, which must be ignored.
            extra = rng.sample(absent, min(len(absent), rng.randint(0, 2)))
            queue.remove_requests([reqs[i] for i in many + extra])
            for i in many:
                present.remove(i)
            args = many + extra
        elif op == "iter":
            result = [index[id(r)] for r in queue]
            args = []
        elif op == "set_arrival":
            # Mutating a queued request breaks the heap invariant; the queue
            # must keep comparing live values like Python does.
            i = rng.randrange(n)
            reqs[i].arrival_time = rng.choice(arrivals)
            args = [i, reqs[i].arrival_time]
        elif op == "set_priority":
            i = rng.randrange(n)
            reqs[i].priority = rng.randint(-2, 2)
            args = [i, reqs[i].priority]
        assert len(queue) == len(present)
        ops.append({"op": op, "args": args, "result": result, "state": internal()})
    return {
        "policy": policy,
        "requests": initial,
        "ops": ops,
    }


def gen_int_set_case(rng: random.Random, num_ops: int) -> dict:
    hi = rng.choice([8, 16, 40, 200])
    s: set[int] = set()
    ops = []
    for _ in range(num_ops):
        if s and rng.random() < 0.45:
            x = rng.choice(list(s)) if rng.random() < 0.8 else rng.randrange(hi)
            s.discard(x)
            op = "discard"
        else:
            x = rng.randrange(hi) if rng.random() < 0.7 else rng.randrange(8)
            s.add(x)
            op = "add"
        ops.append({"op": op, "value": x, "state": list(s)})
    return {"ops": ops}


class FixtureRequest:
    """The request surface the encoder cache manager reads."""

    def __init__(self, request_id: str, hashes: list[str], counts: list[int]):
        self.request_id = request_id
        self.mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=h,
                mm_position=PlaceholderRange(offset=0, length=c),
            )
            for h, c in zip(hashes, counts)
        ]

    def get_num_encoder_embeds(self, input_id: int) -> int:
        return self.mm_features[input_id].mm_position.get_num_embeds()


def gen_encoder_cache_case(rng: random.Random, num_ops: int) -> dict:
    num_hashes = rng.randint(2, 30)
    hash_sizes = {f"h{i}": rng.randint(1, 6) for i in range(num_hashes)}
    cache_size = rng.randint(6, 40)
    reqs = []
    for r in range(rng.randint(1, 5)):
        num_items = rng.randint(1, 24)
        hashes = [rng.choice(list(hash_sizes)) for _ in range(num_items)]
        reqs.append(FixtureRequest(f"r{r}", hashes, [hash_sizes[h] for h in hashes]))
    manager = EncoderCacheManager(cache_size)
    ops = []
    for _ in range(num_ops):
        r = rng.randrange(len(reqs))
        req = reqs[r]
        input_id = rng.randrange(len(req.mm_features))
        op = rng.choice(
            ["schedule"] * 5 + ["free_input"] * 3 + ["free", "freed", "cached_ids"]
        )
        if op == "schedule":
            budget = rng.randint(0, 10)
            to_schedule = rng.randint(0, 4)
            if manager.check_and_update_cache(req, input_id):
                result = "hit"
            elif manager.can_allocate(req, input_id, budget, to_schedule):
                manager.allocate(req, input_id)
                result = "allocated"
            else:
                result = "rejected"
            args = [r, input_id, budget, to_schedule]
        elif op == "free_input":
            manager.free_encoder_input(req, input_id)
            result = None
            args = [r, input_id]
        elif op == "free":
            manager.free(req)
            result = None
            args = [r]
        elif op == "freed":
            result = manager.get_freed_mm_hashes()
            args = []
        else:
            result = list(manager.get_cached_input_ids(req))
            args = [r]
        ops.append(
            {
                "op": op,
                "args": args,
                "result": result,
                "num_free_slots": manager.num_free_slots,
                "num_freeable_slots": manager.num_freeable_slots,
                "freeable": [[h, n] for h, n in manager.freeable.items()],
                "freed": list(manager.freed),
            }
        )
    return {
        "cache_size": cache_size,
        "requests": [
            {
                "id": req.request_id,
                "hashes": [f.identifier for f in req.mm_features],
                "counts": [f.mm_position.length for f in req.mm_features],
            }
            for req in reqs
        ],
        "ops": ops,
    }


def main() -> None:
    rng = random.Random(0)
    queue_cases = [
        gen_queue_case(rng, policy, rng.randint(5, 60))
        for policy in ("priority", "fcfs")
        for _ in range(60)
    ]
    (OUT_DIR / "request_queue_cases.json").write_text(
        json.dumps(queue_cases, separators=(",", ":"), ensure_ascii=False)
    )
    int_set_cases = [gen_int_set_case(rng, rng.randint(5, 100)) for _ in range(80)]
    (OUT_DIR / "py_int_set_cases.json").write_text(
        json.dumps(int_set_cases, separators=(",", ":"))
    )
    ecm_cases = [gen_encoder_cache_case(rng, rng.randint(10, 80)) for _ in range(60)]
    (OUT_DIR / "encoder_cache_cases.json").write_text(
        json.dumps(ecm_cases, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
