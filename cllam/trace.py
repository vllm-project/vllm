from dataclasses import dataclass
import json
from os import path
from pathlib import Path
from typing import Dict, List, Tuple, Union


@dataclass
class Trace:
    tid: int = None


@dataclass
class Step(Trace):
    start: int = None
    end: int = None
    batch_start: int = None
    batch_end: int = None
    is_prompt_run: bool = None
    batched_token_num: int = None
    context_token_num: int = None
    batched_requests: List[int] = None
    preempted_requests: List[int] = None
    available_slots: int = None


@dataclass
class Request(Trace):
    start: int = None
    end: int = None
    prompt_len: int = None
    gen_len: int = None


class Tracer:
    TRACE_FOLDER = path.join(Path(__file__), "trace")

    def __init__(self):
        self.traces: Dict[str, Trace] = {}

    def add(self, trace_type: type) -> int:
        assert issubclass(trace_type, Trace), "Invalid trace type is provided!"
        tid = len(self.traces)
        self.traces[tid] = trace_type(tid=tid)
        return tid

    def get(self, tid: int) -> Union[Request, Step]:
        assert tid in self.traces, "Invalid trace id is provided!"
        return self.traces[tid]

    def export(self, filename: str = "trace"):
        trace_bundle: Dict[str, List[Trace]] = {}
        for trace in self.traces.values():
            type_name = trace.__class__.__name__
            if type_name not in trace_bundle:
                trace_bundle[type_name] = []
            trace_bundle[type_name].append(trace)

        trace_path = path.join(self.TRACE_FOLDER, f"{filename}.json")
        with open(trace_path, "w") as file:
            json.dump(trace_bundle, file)

    def clear(self):
        self.traces.clear()


TRACER = Tracer()