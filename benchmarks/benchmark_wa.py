# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the performance of workload-aware (WA) cache policy.

paper: https://arxiv.org/abs/2506.02634

trace: https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon

This file allows you to benchmark the effect of 
workload-aware kv cache cache policy via the
2-hours open source trace.

Profiler will first profile hyperparameters for each workload
based the first hour and dump to a file,
then vLLM will use these hyperparameters to serve the second hour.

This file includes a RequestIssuer to simulate the multi-turn dialog,
a RequestMapper to reconstruct the token_ids based on hashed block,
they will preserve the KVCache use pattern in multi-turn dialog.

This file will calucate the metrics and report the QTTFTs, TPOTs.

Example usage:
    python benchmark_wa.py \
        --model Qwen/Qwen2-7B-Instruct/ \
        --enable-prefix-caching \
        --block-size 128 \
        --num_gpu_blocks_override 5120 \
        --qwen-trace-file qwen_traceA_blksz_16.jsonl \
        --max-num-batched-tokens 16384    
        --trace-block-size 16 \
        --workloads '[("qwen",-1)]' \
        --enable-wa-policy \
        --prefill-mode

"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import random
import threading
import time
import traceback
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime
from itertools import combinations
from typing import Optional

import numpy as np
from colorama import Fore, Style
from profiler_utils import BlockManager, ExtraInfo
from sortedcontainers import SortedDict
from tqdm import tqdm
from transformers import AutoTokenizer

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter, FlexibleArgumentParser


def color_print(*args, **kwargs):
    print(Fore.GREEN)
    print(*args, **kwargs)
    print(Style.RESET_ALL)


PrefixHash = int

HYPERPARAMETERS_FILE = f"params-{datetime.now().strftime(r'%m%d-%H%M%S')}.json"
PREFILL_MODE = False
USE_RAW_TOKEN_IDS = False


class Request:
    def __init__(
        self,
        hash_ids: list[int],
        output_len: int,
        specific_interval: Optional[float] = None,
        type_id: str = "default",
        turn: int = 1,
        prev_request: Request = None,
        fixed_prompt_len: Optional[int] = None,
        chat_id: str = None,
        parent_chat_id: str = None,
        origin_ts: float = None,
        temperature: float = None,
        request_mapper: RequestMapper = None,
        relative_ts: Optional[float] = None,
    ):
        self.id = None
        self.hash_ids = hash_ids
        if turn == 1:  # first turn
            assert prev_request is None, (
                f"turn == 1 but have prev request, id:{prev_request.chat_id}"
            )
            # first turn, no matter P mode / P+D mode, we can generate token_ids now
            self.prompt = request_mapper.generate_token_list(
                chat_id, hash_ids, fixed_prompt_len, turn, prev_request
            )
        else:
            if PREFILL_MODE:  # P mode, just generate token_ids via the block info
                self.prompt = request_mapper.generate_token_list(
                    chat_id, hash_ids, fixed_prompt_len, turn, prev_request
                )
            else:  # P+D mode, token_ids depends on last turn's real output
                self.prompt = None  # init later
        self.output_len = output_len
        self.specific_interval = specific_interval
        self.type_id = type_id
        self.turn = turn
        self.prev_request = prev_request
        self.next_request = None
        self.next_request_idx = None
        self.next_req_distance = None
        self.fixed_prompt_len = fixed_prompt_len
        self.chat_id = chat_id
        self.parent_chat_id = parent_chat_id
        self.origin_ts = origin_ts
        self.relative_ts = relative_ts  # optional for USE_RAW_TOKEN_IDS
        self.output: Optional[list[int]] = None
        self.initialized = self.prev_request is None
        self.pinned = False
        self.request_issue_time = 0
        self.pre_tokens = 0
        self.temperature = temperature
        self.request_mapper = request_mapper
        self.metrics = None

    @property
    def params(self) -> SamplingParams:
        return SamplingParams(
            ignore_eos=True,
            max_tokens=self.output_len,
            min_tokens=self.output_len,
        )

    @property
    def metainfo(self) -> dict:
        assert self.init()
        return str(self.type_id) + "_" + str(self.turn)

    def generate_token_now(self):
        self.prompt = self.request_mapper.generate_token_list(
            self.chat_id,
            self.hash_ids,
            self.fixed_prompt_len,
            self.turn,
            self.prev_request,
            force_generate=True,
        )

    def init(self) -> bool:
        if self.initialized:  # first turn
            return True
        if self.prev_request.output is None:  # prev request's output is not ready
            return False
        assert isinstance(self.prev_request.output, list)

        # construct multi turn contents when prev's output is ready
        self.prompt = self.request_mapper.generate_token_list(
            self.chat_id,
            self.hash_ids,
            self.fixed_prompt_len,
            self.turn,
            self.prev_request,
        )
        self.initialized = True
        return True

    def to_dict(self):
        return {
            "id": self.id,
            "output_len": self.output_len,
            "specific_interval": self.specific_interval,
            "type_id": self.type_id,
            "fixed_prompt_len": self.fixed_prompt_len,
            "chat_id": self.chat_id,
            "parent_chat_id": self.parent_chat_id,
            "ts": self.origin_ts,
            "metrics": self.metrics,
        }

    def get_hashs(self, block_size: int) -> list[PrefixHash]:
        assert 0, "get hashes is not supported now.."
        assert isinstance(self.prompt, list), (
            "prompt should be tokenized when call get_hashs"
        )
        block_cnt = len(self.prompt) // block_size
        is_first_block = True
        block_hashes = []
        prev_block_hash: Optional[int] = None
        for i in range(block_cnt):
            cur_token_ids = self.prompt[i * block_size : i * block_size + block_size]
            prev_block_hash = hash((is_first_block, prev_block_hash, *cur_token_ids))
            block_hashes.append(prev_block_hash)
            is_first_block = False
        return block_hashes


class EventType(enum.Enum):
    added = enum.auto()
    prefilled = enum.auto()


class Event:
    def __init__(self, ty, data):
        self.ty = ty
        self.data = data


class Profiler:
    def __init__(
        self, data, block_size: int, fitting_version: str = "", overwrite: bool = False
    ):
        self.data = data
        self.block_size = block_size
        self.event_queue: SortedDict[Event] = SortedDict()

        self.param_file_name = HYPERPARAMETERS_FILE
        self.overwrite = overwrite
        self.ttft_mean = 2000

        estimate_block_number = 0
        for request in data:
            estimate_block_number += math.ceil(request.fixed_prompt_len / block_size)
        estimate_block_number = int(estimate_block_number * 1.1)
        print(f"profiler, gpu {estimate_block_number=}")

        self.block_manager: BlockManager = BlockManager(
            block_size=block_size, num_block=estimate_block_number
        )

        self.added: set[str] = set()
        self.metrics: dict = {}

        self._init_event_queue(data)

    def _get_extras(self, request: Request) -> ExtraInfo:
        turn = request.turn
        workload = f"{request.type_id}_{turn}"
        reach_ts = request.relative_ts
        prefilled_ts = reach_ts + self.ttft_mean / 1000.0
        return ExtraInfo(workload, reach_ts, prefilled_ts, reach_ts)

    def add_request(self, request: Request) -> tuple[int, int]:
        assert request.prompt is not None, (
            f"{request.chat_id} requests for profiling should have token_ids now!"
        )
        hit_cnt = self.block_manager.alloc(request.prompt, self._get_extras(request))
        return hit_cnt

    def free_request(self, request):
        self.block_manager.free(
            request.prompt,
            self._get_extras(request),
        )

    def process(self):
        illegal_cnt = 0
        while self.event_queue:
            ts, e = self.event_queue.popitem(0)
            # print(e.ty, ts, e.data["chat_id"])
            if e.ty == EventType.added:
                cid = e.data.chat_id
                hit_cnt = self.add_request(e.data)
                self.added.add(cid)
                self.metrics[cid] = {
                    "extra": vars(self._get_extras(request=e.data)),
                    "num_tokens": e.data.fixed_prompt_len,
                    "hit": hit_cnt,
                }
            elif e.ty == EventType.prefilled:
                assert e.data.chat_id in self.added, f"{e.data.chat_id=}"
                if e.data.chat_id not in self.added:
                    illegal_cnt += 1
                    continue
                self.free_request(e.data)
        print(f"{illegal_cnt=}")

        dump_hyperparameters = {
            "a": self.block_manager.get_coefficients(),
            "p": self.block_manager.get_next_turn_p(),
        }

        # dump the hyperparameter file
        if not os.path.exists(self.param_file_name) or self.overwrite:
            with open(self.param_file_name, "w") as f:
                json.dump(dump_hyperparameters, f, indent=4)

    def _init_event_queue(self, data):
        prev_reqs = set([-1, "-1"])
        for request in data:
            cid = request.chat_id
            pid = request.parent_chat_id
            if pid not in prev_reqs:
                continue
            ts: float = request.relative_ts
            ttft: float = self.ttft_mean
            assert ts < ts + ttft, f"illegal {ts=}, {ttft=}"
            self.event_queue[ts] = Event(EventType.added, request)
            self.event_queue[ts + ttft] = Event(EventType.prefilled, request)
            prev_reqs.add(cid)
        print(f"Total event number {len(self.event_queue) // 2}")


class RequestMapper:
    def __init__(self, block_size: int = 16):
        # map from hash id to [token_id,]
        self.hash_id_to_token_list = {}
        self.block_size = block_size
        self.token_id_min = 10
        self.token_id_max = 20000

    def _validate_token_list(self, token_list: list[int]):
        for token in token_list:
            assert token >= self.token_id_min and token <= self.token_id_max

    def generate_token_list(
        self,
        chat_id: str,
        hash_ids: list[int],
        input_len: int,
        turn: int,
        prev_request: Request,
        force_generate: Optional[bool] = False,
    ) -> list[int]:
        if USE_RAW_TOKEN_IDS:
            if not PREFILL_MODE:
                raise RuntimeError("unimplemented for P+D with raw token_ids")
            return hash_ids

        assert math.ceil(input_len / self.block_size) == len(hash_ids)
        res = []
        if turn == 1:  # first turn
            for i, hash_id in enumerate(hash_ids):
                if i < len(hash_ids) - 1:
                    cur_length = self.block_size
                else:
                    cur_length = input_len % self.block_size
                if cur_length == 0:
                    cur_length = self.block_size
                if hash_id in self.hash_id_to_token_list:  # exist
                    tokens = self.hash_id_to_token_list[hash_id]
                else:  # generate a new random
                    tokens = [
                        random.randint(self.token_id_min, self.token_id_max)
                        for _ in range(cur_length)
                    ]
                    self.hash_id_to_token_list[hash_id] = tokens
                res.extend(tokens)
            # for turn=1, no matter p-only or p/d, this assert must be true
            assert len(res) == input_len, (
                f"{chat_id} input len {input_len}, actual len {len(res)} mismatch"
            )
        else:  # multi turn
            parent_blocks = []
            has_incomplete_block = False
            # last mutable blocks content
            if prev_request.fixed_prompt_len % self.block_size > 0:
                assert prev_request.hash_ids[-1] in self.hash_id_to_token_list, (
                    f"{chat_id} {prev_request.hash_ids[-1]} not in map, may truncate?"
                )
                parent_blocks.extend(
                    self.hash_id_to_token_list[prev_request.hash_ids[-1]]
                )
                has_incomplete_block = True
            if has_incomplete_block:
                last_incomplete_block_index = (
                    prev_request.fixed_prompt_len // self.block_size
                )
            else:
                last_incomplete_block_index = -1

            if (
                PREFILL_MODE or force_generate
            ):  # P-mode, just use token_ids, not contact prev outputs
                for i, hash_id in enumerate(hash_ids):
                    if i < len(hash_ids) - 1:
                        cur_length = self.block_size
                    else:
                        cur_length = input_len % self.block_size
                    if cur_length == 0:
                        cur_length = self.block_size

                    if (
                        hash_id in self.hash_id_to_token_list
                    ):  # exist last immutable blocks
                        tokens = self.hash_id_to_token_list[hash_id]
                        res.extend(tokens)
                    else:  # may belongs to parent requests
                        assert i >= last_incomplete_block_index, (
                            f"{chat_id} P not map index {i}"
                            f"must >= {last_incomplete_block_index}"
                        )
                        if (
                            i == last_incomplete_block_index
                        ):  # special case, use partial old blocks
                            tokens = parent_blocks
                            remain_length = cur_length - len(parent_blocks)
                            remain_tokens = [
                                random.randint(self.token_id_min, self.token_id_max)
                                for _ in range(remain_length)
                            ]
                            tokens.extend(remain_tokens)
                        else:
                            tokens = [
                                random.randint(self.token_id_min, self.token_id_max)
                                for _ in range(cur_length)
                            ]
                        assert len(tokens) == cur_length, (
                            f"{chat_id} P, token length {len(tokens)} != {cur_length}"
                        )
                        res.extend(tokens)
                        self.hash_id_to_token_list[hash_id] = tokens
            else:  # P+D, use last output to replace token ids
                assert prev_request.output_len == len(prev_request.output), (
                    f"{chat_id} P+D prev_request not finished"
                )
                parent_blocks.extend(prev_request.output)
                new_input_token_length = (
                    input_len - prev_request.fixed_prompt_len - prev_request.output_len
                )
                assert new_input_token_length >= 0, (
                    f"{chat_id} P+D new input token length {new_input_token_length}"
                    f"<0, input len {input_len},"
                    f"prev_input{prev_request.fixed_prompt_len}"
                    f"prev output len {prev_request.output_len} may truncate here?"
                )
                new_input_tokens = [
                    random.randint(self.token_id_min, self.token_id_max)
                    for _ in range(new_input_token_length)
                ]
                concated_blocks = parent_blocks + new_input_tokens
                block_number = math.ceil(len(concated_blocks) / self.block_size)
                prev_input_block_number = (
                    prev_request.fixed_prompt_len // self.block_size
                )
                assert prev_input_block_number + block_number == len(hash_ids), (
                    f"{chat_id} P+D last immutable input block number "
                    f"{prev_input_block_number}, "
                    f"input+output+input2 block number {block_number}"
                    f" cur_hash_lengths: {len(hash_ids)}"
                )
                for i, hash_id in enumerate(hash_ids):
                    if i < len(hash_ids) - 1:
                        cur_length = self.block_size
                    else:
                        cur_length = input_len % self.block_size
                    if cur_length == 0:
                        cur_length = self.block_size

                    if (
                        hash_id in self.hash_id_to_token_list
                    ):  # exist last immutable blocks
                        tokens = self.hash_id_to_token_list[hash_id]
                        res.extend(tokens)
                    else:  # may belongs to parent requests
                        assert i >= prev_input_block_number, (
                            f"{chat_id} P+D not map index {i}"
                            f"must >= {prev_input_block_number}"
                        )
                        diff = i - prev_input_block_number  # start from 0
                        assert diff < block_number, (
                            f"{chat_id} P+D not map index diff {diff}"
                            f"must < block number {block_number}"
                        )
                        mx = min(self.block_size * (diff + 1), len(concated_blocks))
                        tokens = concated_blocks[self.block_size * diff : mx]
                        res.extend(tokens)
                        self.hash_id_to_token_list[hash_id] = tokens

            assert len(res) == input_len, (
                f"{chat_id} input len {input_len}, actual len {len(res)} mismatch"
            )

        # self._validate_token_list(res)
        return res


class RequestsBuilder:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        args=None,
    ):
        self.dataset_path = ""
        self.tokenizer = tokenizer
        self.block_size = args.trace_block_size
        self.token_id_min = 100
        self.token_id_max = 20000
        self.args = args
        self.block_generator = combinations(
            range(self.token_id_min, self.token_id_max + 1), self.block_size
        )
        self._build_funcs = {"qwen": self._build_qwen_requests}
        self.request_mapper = RequestMapper(block_size=self.block_size)

    def build(
        self,
        workloads: list[tuple[str, int]],
        temperature: Optional[float] = None,
        model_ty: Optional[str] = None,
    ) -> list[Request]:
        requests_list = []
        workload_cnt = {w[0]: 0 for w in workloads if isinstance(w, tuple)}
        for i, workload in enumerate(workloads):
            # Duplicate a requests set.
            if isinstance(workload, int):
                same_as = workload
                workload_id = workloads[same_as][0]
                workload_cnt[workload_id] += 1
                requests: list[Request] = deepcopy(requests_list[same_as])
                for req in requests:
                    req.type_id = f"{workload_id}_{workload_cnt[workload_id]}"
                requests_list.append(requests)
                continue

            workload_id = workload[0]
            workload_cnt[workload_id] += 1
            assert workload_id in self._build_funcs
            build_func = self._build_funcs[workload_id]
            num_requests = workload[1]
            type_id = workload_id
            if workload_cnt[workload_id] > 1:
                type_id += f"_{workload_cnt[workload_id]}"
            build_args = {
                "num_requests": num_requests,
                "type_id": type_id,
                "temperature": temperature,
                "model_ty": model_ty,
            }
            if build_func == self._build_qwen_requests:
                build_args["qwen_trace_file"] = self.args.qwen_trace_file
            requests_list.append(build_func(**build_args))

        return self._flatten(requests_list)

    def _flatten(self, requests_list: list[list[Request]]) -> list[Request]:
        res = []
        for requests in requests_list:
            res.extend(requests)
        return res

    def _build_request(
        self,
        hash_ids: list[int],
        output_len: Optional[int] = None,
        specific_interval: Optional[float] = None,
        type_id: str = "default",
        turn: int = 1,
        prev_request: Optional[Request] = None,
        fixed_prompt_len: Optional[int] = None,
        chat_id: Optional[str] = None,
        parent_chat_id: Optional[str] = None,
        origin_ts: Optional[float] = None,
        temperature: Optional[float] = None,
        relative_ts: Optional[float] = None,
    ) -> Request:
        assert output_len is not None
        return Request(
            hash_ids=hash_ids,
            output_len=output_len,
            specific_interval=specific_interval,
            type_id=type_id,
            turn=turn,
            prev_request=prev_request,
            fixed_prompt_len=fixed_prompt_len,
            chat_id=chat_id,
            parent_chat_id=parent_chat_id,
            origin_ts=origin_ts,
            temperature=temperature,
            request_mapper=self.request_mapper,
            relative_ts=relative_ts,
        )

    def _get_prompt(self, prompt_len):
        prompt = []
        for _ in range(prompt_len // self.block_size):
            prompt.extend(next(self.block_generator))
        for _ in range(prompt_len % self.block_size):
            prompt.append(random.randint(self.token_id_min, self.token_id_max))
        return prompt

    def load_json_or_jsonl(self, file_path):
        assert os.path.exists(file_path), f"{file_path=} does not exist"
        with open(file_path, encoding="utf-8") as f:
            if file_path.endswith(".json"):
                # Try to load as JSON
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
            elif file_path.endswith(".jsonl"):
                # Read as jsonl
                f.seek(0)
                result = []
                for line_number, line in enumerate(f, 1):
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    try:
                        result.append(json.loads(stripped_line))
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_number}: {e}")
            else:
                raise ValueError(f"Invalid trace path {file_path}")
            return result

    def _build_qwen_requests(
        self,
        num_requests: int,
        *args,
        max_prompt_len: int = 32768,
        **kwargs,
    ) -> list[Request]:
        requests = []
        qwen_trace_file = kwargs.get("qwen_trace_file")
        temp = kwargs.get("temperature")
        model_ty = kwargs.get("model_ty")

        def remap_token_ids(token_ids):
            mx_id = 32000
            token_ids = [t % mx_id for t in token_ids]
            return token_ids

        dataset = self.load_json_or_jsonl(qwen_trace_file)
        if num_requests == -1:
            num_requests = len(dataset)
        assert len(dataset) >= num_requests
        print(f"total lengths before process {num_requests}")

        def pts(ts):
            if isinstance(ts, float):
                return ts
            dt = datetime.strptime(ts, r"%Y-%m-%d %H:%M:%S.%f")
            timestamp = dt.timestamp()
            return timestamp

        prev_requests = {-1: None, 0: None, "-1": None}
        skip_cnt = 0
        for i, req in enumerate(dataset[:num_requests]):
            hash_ids = req["no_sp_sw_token_ids" if USE_RAW_TOKEN_IDS else "hash_ids"]
            assert isinstance(hash_ids, list)

            if model_ty == "llama":
                # remap the token ids
                hash_ids = remap_token_ids(hash_ids)

            if USE_RAW_TOKEN_IDS:
                ts = float(req["relative_ts"])
                nxt_ts = (
                    float(dataset[i + 1]["relative_ts"])
                    if i + 1 < num_requests
                    else None
                )
            else:
                ts = pts(req["timestamp"])
                nxt_ts = (
                    pts(dataset[i + 1]["timestamp"]) if i + 1 < num_requests else None
                )
            input_len: int = int(
                req["input_token_length" if USE_RAW_TOKEN_IDS else "input_length"]
            )
            output_len: int = int(
                req["output_token_length" if USE_RAW_TOKEN_IDS else "output_length"]
            )
            type_id: str = req["type"]
            chat_id: str = req["chat_id"]
            turn: int = req["turn"]
            parent_chat_id: str = req["parent_chat_id"]
            relative_ts = float(req["relative_ts"]) if USE_RAW_TOKEN_IDS else None
            # nxt_relative_ts = float(dataset[i + 1])
            if (
                len(hash_ids) == 0
                or (
                    not USE_RAW_TOKEN_IDS
                    and len(hash_ids) * self.block_size > max_prompt_len
                )
                or (USE_RAW_TOKEN_IDS and len(hash_ids) > max_prompt_len)
                or output_len <= 0
                or parent_chat_id not in prev_requests
            ):
                skip_cnt += 1
                continue
            prev_req = prev_requests[parent_chat_id]
            request = self._build_request(
                hash_ids=hash_ids,
                output_len=1 if PREFILL_MODE else output_len,
                specific_interval=nxt_ts - ts if nxt_ts else None,
                type_id=type_id,
                turn=turn,
                prev_request=prev_req,
                fixed_prompt_len=input_len,
                chat_id=chat_id,
                parent_chat_id=parent_chat_id,
                origin_ts=ts,
                temperature=temp,
                relative_ts=relative_ts,
            )
            if prev_req:
                prev_req.next_request = request
                prev_req.next_request_idx = i

            requests.append(request)
            prev_requests[chat_id] = request

        print(f"valid requests len {len(requests)}, skip cnt {skip_cnt}")
        return requests


class RequestIssuer:
    def __init__(self, llm: LLMEngine, block_size: int = 16):
        self.llm = llm
        self.requests: deque[Request] = deque()
        self.request_issue_time = {}
        self.thread = None
        self.counter = Counter()
        self.requests_dic = {}
        self.block_size = block_size

    def issue_time(self, request_id: str) -> float:
        return self.request_issue_time.get(request_id, None)

    def empty(self) -> bool:
        return len(self.requests) == 0

    def add_request(self, request: Request):
        assert request.id is None
        request.id = str(next(self.counter))
        self.requests.append(request)
        self.requests_dic[request.id] = request

    def get_request(self, request_id: str) -> Request:
        assert request_id in self.requests_dic
        return self.requests_dic[request_id]

    def start(self):
        self.thread = threading.Thread(target=self._start)
        self.thread.start()

    def stop(self):
        self.requests = deque()
        self.qps = math.inf

    def join(self):
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        self.thread = None

    def _get_ready_request(self) -> Optional[Request]:
        found = None
        for i in range(len(self.requests)):
            if self.requests[i].init():
                found = self.requests[i]
                break
            if self.requests[i].pinned:
                break
        if found:
            tmp = []
            while True:
                req = self.requests.popleft()
                if req == found:
                    break
                tmp.append(req)
            while tmp:
                self.requests.appendleft(tmp[-1])
                tmp.pop()
        return found

    def _start(self):
        while self.requests:
            request = self._get_ready_request()

            if request:
                assert request.init()
                request_id = request.id
                prompt = request.prompt
                params = request.params
                metainfo = request.metainfo

                self.llm.add_request(
                    request_id=request_id,
                    prompt=TokensPrompt(prompt_token_ids=prompt),
                    params=params,
                    type_info=metainfo,
                )
                self.request_issue_time[request_id] = time.perf_counter()
                request.request_issue_time = time.perf_counter()

            # TODO This may be later than actual time
            if request and request.specific_interval:
                time.sleep(request.specific_interval)


def run_vllm(requests: list[Request], args) -> float:
    t1 = time.perf_counter()

    engine_args = EngineArgs.from_cli_args(args)
    llm = LLMEngine.from_engine_args(engine_args)

    t2 = time.perf_counter()
    color_print(f"[benchmark-cache] run vllm. build llm engine time: {t2 - t1} s")

    request_issuer = RequestIssuer(llm=llm, block_size=args.block_size)

    for request in requests:
        request_issuer.add_request(request)

    num_steps: int = 0
    prompt_durations, decode_durations = [], []
    outputs: dict[str, list[RequestOutput]] = defaultdict(list)
    requests_each_step = []

    pbar: tqdm = tqdm(total=len(requests), desc="Processed", dynamic_ncols=True)

    request_issuer.start()
    time.sleep(1)

    # request_issuer.add_all()
    start = time.perf_counter()
    try:
        while llm.has_unfinished_requests() or (
            pbar.n < pbar.total and not request_issuer.empty()
        ):
            step_start = time.perf_counter()
            step_outputs = llm.step()
            step_finish = time.perf_counter()
            step_duration = step_finish - step_start

            requests_each_step.append([output.request_id for output in step_outputs])
            for output in step_outputs:
                output.token_length = len(output.outputs[0].token_ids)
                output.latency = step_duration
                output.timestamp = step_finish
                output.step_start = step_start
                if (
                    output.token_length != 0
                ):  # for speculative decoding, can simply ignore it now
                    while (
                        outputs[output.request_id]
                        and outputs[output.request_id][-1].token_length
                        >= output.token_length
                    ):
                        outputs[output.request_id].pop(-1)
                    outputs[output.request_id].append(output)
                if len(outputs[output.request_id]) == 1:
                    outputs[output.request_id][0].start_decode_ts = step_finish
                    outputs[output.request_id][0].ttft = step_duration + (
                        step_start - outputs[output.request_id][0].step_start
                    )
                    outputs[output.request_id][0].qlatency = (
                        step_finish - request_issuer.issue_time(output.request_id)
                    )
                if output.finished:
                    output_token_ids = list(output.outputs[0].token_ids)
                    request_issuer.get_request(
                        output.request_id
                    ).output = output_token_ids
                    pbar.update(1)

            prompt_run = all(
                len(output.outputs[0].token_ids) == 1 for output in step_outputs
            )
            if prompt_run:
                prompt_durations.append(step_duration)
            else:
                decode_durations.append(step_duration)
            num_steps += 1

            if step_outputs is None:  # sleep to stop busy-waiting
                time.sleep(0.01)
    except Exception:
        traceback.print_exc()
    finally:
        request_issuer.stop()
        request_issuer.join()
        pbar.close()
    end = time.perf_counter()
    color_print(f"[benchmark-cache] run vllm.all steps time: {end - start} s")

    return (
        outputs,
        prompt_durations,
        decode_durations,
        requests_each_step,
    )


def print_metrics(
    args: argparse.Namespace,
    requests: list[Request],
    outputs: dict[str, list[RequestOutput]],
    prompt_durations: list[float],
    decode_durations: list[float],
    requests_each_step: Optional[list[list[str]]],
    verbose: Optional[bool],
    filename: Optional[str],
):
    outputs: list[list[RequestOutput]] = [outputs[req.id] for req in requests]

    def group_metrics(type_id: Optional[str] = None):
        chosens = [
            (req, out)
            for req, out in zip(requests, outputs)
            if type_id is None or req.type_id == type_id
        ]
        input_lengths, output_lengths = [], []
        iterations, accept_rates = [], []
        qttfts, ttfts, tpots, tpors = [], [], [], []
        decoding_durations, query_durations, queueings = [], [], []
        hit_tokens = []
        for request, request_outputs in chosens:
            if not request_outputs:
                continue
            output = request_outputs[-1]
            output_len = len(output.outputs[0].token_ids)
            decode_duration = (
                request_outputs[-1].timestamp - request_outputs[0].start_decode_ts
            )
            query_duration = decode_duration + request_outputs[0].ttft
            decoding_durations.append(decode_duration)
            query_durations.append(query_duration)
            num_iterations = len(request_outputs)
            accept_rate = (output_len - num_iterations) / max(1, num_iterations)
            input_lengths.append(request.fixed_prompt_len)
            output_lengths.append(output_len)
            iterations.append(len(request_outputs))
            accept_rates.append(accept_rate)
            qttfts.append(request_outputs[0].qlatency)
            ttfts.append(request_outputs[0].ttft)
            queueings.append(request_outputs[0].qlatency - request_outputs[0].ttft)
            for prev, output in zip(request_outputs, request_outputs[1:]):
                num_tokens = output.token_length - prev.token_length
                tpots.extend(
                    output.latency / max(1, num_tokens) for _ in range(num_tokens)
                )
            tpors.extend(output.latency for output in request_outputs[1:])

            hit_tokens.append(request_outputs[0].num_cached_tokens)

            def get_last(lst):
                return lst[-1] if len(lst) > 0 else None

            request.metrics = {
                "qttft": get_last(qttfts),
                "ttft": get_last(ttfts),
                "tpot": get_last(tpots),
                "tpor": get_last(tpors),
                "decoding_dur": get_last(decoding_durations),
                "query_dur": get_last(query_durations),
                "queue_dur": get_last(queueings),
                "iterations": get_last(iterations),
                "accept_rate": get_last(accept_rates),
                "hit_tokens": get_last(hit_tokens),
            }

        qttfts = qttfts if qttfts else [0]
        ttfts = ttfts if ttfts else [0]
        tpots = tpots if tpots else [0]
        tpors = tpors if tpors else [0]

        selected_percentiles = [99, 90, 75]

        def _add(metrics, key: str, lst: list[float]):
            metrics[f"mean_{key}_ms"] = np.mean(lst or 0) * 1000
            metrics[f"std_{key}_ms"] = np.std(lst or 0) * 1000
            metrics[f"median_{key}_ms"] = np.median(lst or 0) * 1000
            for p in [50, 80, 90, 95, 99]:
                metrics[f"p{p}_{key}_ms"] = np.percentile(lst or 0, p) * 1000

            if verbose:
                metrics[f"percentiles_{key}_ms"] = [
                    (p, np.percentile(lst or 0, p) * 1000) for p in selected_percentiles
                ]
                metrics[f"all_{key}"] = lst

        metrics = {
            "total_input_tokens": sum(input_lengths),
            "total_output_tokens": sum(output_lengths),
            "total_requests": len(chosens),
        }

        _add(metrics, "qttft", qttfts)
        _add(metrics, "ttft", ttfts)
        _add(metrics, "tpot", tpots)
        _add(metrics, "tpors", tpors)
        _add(metrics, "decode", decoding_durations)
        _add(metrics, "query", query_durations)
        _add(metrics, "queue", queueings)

        metrics["extras"] = {
            "total_hit_tokens": sum(hit_tokens),
            "hit_rate": sum(hit_tokens) / sum(input_lengths)
            if sum(input_lengths) > 0
            else 0.0,
        }

        return metrics

    pre_filename = filename[: -len(".json")]

    metrics = {}
    metrics["args"] = args.__dict__
    # it's not json serializable
    metrics["args"]["compilation_config"] = None
    group_ids = [req.type_id for req in requests]
    vis_group_ids = set()
    for type_id in group_ids:
        if type_id in vis_group_ids:
            continue
        vis_group_ids.add(type_id)
        metrics[type_id] = group_metrics(type_id)
    metrics["all"] = group_metrics()

    if verbose:
        metrics["requests_each_step"] = requests_each_step

    def serialize_safe(obj):
        try:
            return str(obj)
        except Exception:
            return None

    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4, default=serialize_safe)

    all_requests_output = [req.to_dict() for req in requests]
    assert filename.endswith(".json")
    pre_filename = filename[: -len(".json")]
    with open(f"{pre_filename}-raw.json", "w") as f:
        json.dump(all_requests_output, f, indent=4)


def main(args: argparse.Namespace):
    # print(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )

    model_ty = "qwen" if "qwen" in args.model.lower() else "llama"

    # step1, read and constuct requests from trace
    t1 = time.perf_counter()
    requests = RequestsBuilder(
        tokenizer=tokenizer,
        args=args,
    ).build(
        workloads=args.workloads,
        model_ty=model_ty,
    )
    t2 = time.perf_counter()
    color_print(f"Total request {len(requests)}")
    color_print(f"[benchmark-cache] build data set time: {t2 - t1} s")

    # step2, use the first hours' request to profile
    color_print("Split the first hours' request to profile ")
    t1 = time.perf_counter()
    profile_requests = []
    test_request_map = {-1: None}
    test_requests = []
    throw_cnt = 0
    for request in requests:
        if not USE_RAW_TOKEN_IDS:
            request.relative_ts = request.origin_ts
        if request.relative_ts <= 3600:  # first hours request
            profile_requests.append(request)
        else:  # second hours request
            if request.turn == 1:
                test_requests.append(request)
                test_request_map[request.chat_id] = request
            else:
                if request.parent_chat_id in test_request_map:
                    test_requests.append(request)
                    test_request_map[request.chat_id] = request
                else:  # throw the request chain start in the first hour
                    throw_cnt += 1
    assert len(profile_requests) + len(test_requests) + throw_cnt == len(requests)
    color_print(
        f"Profile request number {len(profile_requests)}"
        f"test number {len(test_requests)} throw cnt {throw_cnt}"
    )
    # profile and dump the file
    if args.enable_wa_policy and args.wa_offline_param_path == "":
        color_print(
            f"wa_offline_param_path not specified,"
            f"profiling and dump to {HYPERPARAMETERS_FILE}"
        )
        for request in profile_requests:
            request.generate_token_now()
        profiler = Profiler(
            data=profile_requests,
            block_size=args.block_size,
            fitting_version="v1",
            overwrite=True,
        )
        profiler.process()

        gpu_hit = profiler.block_manager.gpu_hit
        total_alloc = profiler.block_manager.total_alloc
        hit_ratio = gpu_hit / total_alloc
        color_print(f"profiler {gpu_hit=}, {total_alloc=}, {hit_ratio=:.6f}")

        args.wa_offline_param_path = HYPERPARAMETERS_FILE
    t2 = time.perf_counter()
    color_print(f"Finish profile and dump the hyperparameter file, time {t2 - t1} s")
    requests = test_requests

    # step3, inference last-half hours' requests
    outputs, prompt_durations, decode_durations, requests_each_step = run_vllm(
        requests=requests, args=args
    )
    t3 = time.perf_counter()
    color_print(f"[benchmark-cache] run vllm time: {t3 - t2} s")

    # step4, show metrics
    print_metrics(
        args=args,
        requests=requests,
        outputs=outputs,
        prompt_durations=prompt_durations,
        decode_durations=decode_durations,
        requests_each_step=requests_each_step,
        verbose=args.verbose,
        filename=args.filename,
    )
    t4 = time.perf_counter()
    color_print(f"[benchmark-cache] print metrics time: {t4 - t3} s")
    color_print(f"[benchmark-cache] total time: {t1} {t4} {t4 - t1} s")


def parse_workloads(arg):
    import ast

    try:
        workloads = ast.literal_eval(arg)
        assert isinstance(workloads, list)
        for i, workload in enumerate(workloads):
            assert isinstance(workload, tuple) or (
                isinstance(workload, int)
                and workload < i
                and isinstance(workloads[workload], tuple)
            )
        return workloads
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError from ValueError(
            "Workloads must be a valid Python list of tuples."
        )


def get_unique_filename(filename):
    if not os.path.exists(filename):
        return filename

    dir_name = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    name, ext = os.path.splitext(base_name)

    counter = 1
    while True:
        new_name = f"{name}-v{counter}{ext}"
        new_path = os.path.join(dir_name, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark workload-aware policy")

    parser.add_argument(
        "--filename",
        type=str,
        default=f"logs_tmp/{datetime.now().strftime(r'%m%d-%H%M%S')}.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="get verbose metrics",
    )
    parser.add_argument(
        "--prefill-mode",
        action="store_true",
        help="Enable prefill mode, this will simulate \
            P-node in P-D disaggregation setup",
    )
    parser.add_argument(
        "--workloads",
        type=parse_workloads,
        default=None,
        help='Workloads in the format: [("qwen", -1)], \
            each item in the list can be a int or a tuple, \
            a tuple means a workload and request number, \
            -1 means use all requests in the workload \
            a int "x" means to repeat the workload \
            at the x-th position in the list \
            e.g. [("qwen", -1), 0] means repeat the qwen trace twice',
    )
    parser.add_argument(
        "--qwen-trace-file",
        type=str,
        default="qwen_trace.jsonl",
    )
    parser.add_argument(
        "--trace-block-size",
        type=int,
        default=16,
        help="block size for qwen trace, this is used \
            for constructing the prompt(token_ids) from \
            the hash_ids in trace, \
            which can be different from the block size of vLLM",
    )
    parser.add_argument(
        "--enable-wa-policy",
        action="store_true",
        help="evictor policy, such as LRU, WA (workload-aware)",
    )
    # parser.add_argument(
    #     "--wa-offline-param-path",
    #     type=str,
    #     default="",
    #     help="path to offline param for workload-aware cache eviction policy, \
    #         if not specified, wa cache eviction policy will do profile and \
    #         dump a hyperparameter file and use it",
    # )
    parser.add_argument(
        "--use-raw-token-ids",
        action="store_true",
        help="use raw token_ids rather than block hash",
    )

    # add default args of vllm
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.filename = get_unique_filename(args.filename)
    print(f"log file name: {args.filename=}")

    USE_RAW_TOKEN_IDS = args.use_raw_token_ids

    def validate_args(args, parser):
        if not args.enable_wa_policy and args.wa_offline_param_path:
            parser.error(
                "--wa-offline-param-path is only allowed when --enable-wa-policy"
            )

    validate_args(args, parser)

    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.prefill_mode:
        PREFILL_MODE = True
        color_print("Enable Prefill mode")

    main(args)
