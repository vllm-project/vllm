# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow scheduler for validating the Rust scheduler inside a real engine.

The Python scheduler drives the engine. A Rust scheduler receives identical
calls: the same requests, the same real model outputs and KV connector
outputs, and replayed Mooncake store lookups. Every output is compared step
by step, so GPU nondeterminism and transfer timing cannot mask or fake a
difference. The result is written to `$SHADOW_OUT` as JSON.

Usage (in-process engine, repo root on `PYTHONPATH`):

    VLLM_ENABLE_V1_MULTIPROCESSING=0 SHADOW_OUT=shadow.json python -c "..."
    LLM(..., scheduler_cls="tests.v1.core.rust_shadow_scheduler.ShadowScheduler")
"""

import copy
import json
import os
from array import array

import tests.v1.core.rust_core_diff as _rcd
from tests.v1.core.rust_core_diff import first_diff
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import worker as mc
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.rust_scheduler import RustAsyncScheduler, RustScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.structured_output import StructuredOutputManager

_orig_normalize = _rcd.normalize


def normalize(obj):
    # A deep-copied TokenBuffer becomes a plain array('i'); compare as a list.
    if isinstance(obj, array):
        return [int(x) for x in obj]
    return _orig_normalize(obj)


_rcd.normalize = normalize  # recursion inside rust_core_diff goes through here

# Mooncake lookups: the primary's client does the real RPC and records the
# result; any other client (the shadow's) replays it.
_LOOKUPS: dict[str, list] = {}
_orig_lookup = mc.LookupKeyClient.lookup
_orig_prefetch = mc.LookupKeyClient.prefetch
_orig_discard = mc.LookupKeyClient.discard


def _lookup(self, req_id, num_tokens, block_hashes, non_block=False):
    if getattr(self, "_shadow_primary", False):
        res = _orig_lookup(self, req_id, num_tokens, block_hashes, non_block)
        _LOOKUPS.setdefault(req_id, []).append(res)
        return res
    return _LOOKUPS[req_id].pop(0)


def _prefetch(self, *args, **kwargs):
    if getattr(self, "_shadow_primary", False):
        return _orig_prefetch(self, *args, **kwargs)


def _discard(self, req_id):
    if getattr(self, "_shadow_primary", False):
        return _orig_discard(self, req_id)


mc.LookupKeyClient.lookup = _lookup
mc.LookupKeyClient.prefetch = _prefetch
mc.LookupKeyClient.discard = _discard


class ShadowScheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config,
        kv_cache_config,
        structured_output_manager,
        block_size,
        **kwargs,
    ):
        is_async = vllm_config.scheduler_config.async_scheduling
        py_cls = AsyncScheduler if is_async else Scheduler
        rs_cls = RustAsyncScheduler if is_async else RustScheduler
        self.primary = py_cls(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            **kwargs,
        )
        conn = self.primary.connector
        if conn is not None and getattr(conn, "connector_scheduler", None):
            conn.connector_scheduler.client._shadow_primary = True
        self.shadow = rs_cls(
            vllm_config,
            copy.deepcopy(kv_cache_config),
            StructuredOutputManager(vllm_config),
            block_size,
            **kwargs,
        )
        self.out = os.environ.get("SHADOW_OUT", "shadow_result.json")
        self.steps = 0
        self.compared = 0
        self.diff = None
        self._so_map = {}

    def _check(self, what, a, b):
        self.compared += 1
        if self.diff is None:
            d = first_diff(normalize(a), normalize(b), what)
            if d:
                self.diff = f"step {self.steps}: {d}"
        self._dump()

    def _dump(self):
        if self.compared % 50 and self.diff is None:
            return
        with open(self.out, "w") as f:
            json.dump(dict(steps=self.steps, compared=self.compared, diff=self.diff), f)

    def add_request(self, request):
        twin = copy.deepcopy(request)
        self.primary.add_request(request)
        self.shadow.add_request(twin)

    def schedule(self, throttle_prefills=False):
        self.steps += 1
        a = self.primary.schedule(throttle_prefills)
        b = self.shadow.schedule(throttle_prefills)
        self._so_map[id(a)] = b
        self._check("SchedulerOutput", a, b)
        return a

    def update_from_output(self, scheduler_output, model_runner_output):
        b_so = self._so_map.pop(id(scheduler_output))
        a = self.primary.update_from_output(scheduler_output, model_runner_output)
        b = self.shadow.update_from_output(b_so, model_runner_output)
        self._check("EngineCoreOutputs", a, b)
        self._check(
            "counts",
            (self.primary.get_request_counts(), self.primary.get_kv_cache_usage()),
            (self.shadow.get_request_counts(), self.shadow.get_kv_cache_usage()),
        )
        return a

    def finish_requests(self, request_ids, finished_status):
        ids = list(request_ids) if request_ids is not None else None
        a = self.primary.finish_requests(ids, finished_status)
        b = self.shadow.finish_requests(ids, finished_status)
        self._check("finish_requests", a, b)
        return a

    def reset_prefix_cache(self, *args, **kwargs):
        a = self.primary.reset_prefix_cache(*args, **kwargs)
        b = self.shadow.reset_prefix_cache(*args, **kwargs)
        self._check("reset_prefix_cache", a, b)
        return a

    def shutdown(self):
        with open(self.out, "w") as f:
            json.dump(
                dict(
                    steps=self.steps, compared=self.compared, diff=self.diff, final=True
                ),
                f,
            )
        self.shadow.shutdown()
        self.primary.shutdown()

    # Everything else is answered by the primary.
    def get_grammar_bitmask(self, scheduler_output):
        return self.primary.get_grammar_bitmask(scheduler_output)

    def update_draft_token_ids(self, draft_token_ids):
        self.primary.update_draft_token_ids(draft_token_ids)
        self.shadow.update_draft_token_ids(draft_token_ids)

    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
        raise NotImplementedError("shadow: spec decode not wired")

    def has_requests(self):
        return self.primary.has_requests()

    def has_unfinished_requests(self):
        return self.primary.has_unfinished_requests()

    def has_request(self, request_id):
        return self.primary.has_request(request_id)

    def get_num_unfinished_requests(self):
        return self.primary.get_num_unfinished_requests()

    def has_finished_requests(self):
        return self.primary.has_finished_requests()

    @property
    def pause_state(self):
        return self.primary.pause_state

    def set_pause_state(self, pause_state):
        self.primary.set_pause_state(pause_state)
        self.shadow.set_pause_state(pause_state)

    def reset_encoder_cache(self):
        self.primary.reset_encoder_cache()
        self.shadow.reset_encoder_cache()

    def get_request_counts(self):
        return self.primary.get_request_counts()

    def get_kv_cache_usage(self):
        return self.primary.get_kv_cache_usage()

    def make_stats(self):
        return self.primary.make_stats()

    def get_kv_connector(self):
        return self.primary.get_kv_connector()

    def get_ec_connector(self):
        return self.primary.get_ec_connector()

    def get_kv_event_publisher_config(self):
        return self.primary.get_kv_event_publisher_config()

    def __getattr__(self, name):
        return getattr(self.__dict__["primary"], name)
