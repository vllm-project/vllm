# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OMP Aware Multiprocessing manager for running multiprocessing.Process()
Copyright (c) 2026 Red Hat Inc
Copyright (c) 2026 Cambridge Greys Ltd
"""

import json
import os
import subprocess


def _int(arg):
    """Relaxed parsing of ints which handles a - instead of a number.
    The lscpu json may contain that for nodes in some cases. If that
    is the case we parse it to zero
    """
    try:
        if int(arg) >= 0:
            return int(arg)
    except ValueError:
        pass
    return 0


def parse_mask(mask):
    """Expand a X-Y,Z list"""
    result = []
    for token in mask.split(","):
        try:
            start, finish = token.split("-")
            if int(start) > int(finish):
                raise IndexError("Invalid Indexes for cpu ranges")
            for cpu in range(int(start), int(finish) + 1):
                result.append(cpu)
        except ValueError:
            result.append(int(token))
    return set(result)


def enumerate_resources(resource_map, mask=None, allowed=None):
    """Enumerate system resources"""
    if allowed is None:
        allowed = os.sched_getaffinity(0)
    if mask is not None:
        allowed = allowed & mask

    try:
        allowed_nodes = parse_mask(os.environ["CPU_VISIBLE_MEMORY_NODES"])
    except KeyError:
        allowed_nodes = None

    lscpu: dict[str, dict] = {"cpus": {}, "cores": {}, "nodes": {}}
    for cpu in resource_map["cpus"]:
        cpunum = int(cpu["cpu"])
        if (
            cpunum in allowed
            and cpunum >= 0
            and (allowed_nodes is None or _int(cpu["node"]) in allowed_nodes)
        ):
            lscpu["cpus"][cpunum] = [cpu]
            core = _int(cpu["core"])
            if lscpu["cores"].get(core, None) is None:
                lscpu["cores"][core] = [cpu]
            else:
                lscpu["cores"][core].append(cpu)
            node = _int(cpu["node"])
            if lscpu["nodes"].get(node, None) is None:
                lscpu["nodes"][node] = [cpu]
            else:
                lscpu["nodes"][node].append(cpu)
    return lscpu


def produce_cpu_list(cpus, smt=1):
    """Produce a CPU list with/without SMT pairs - main cpu list case"""
    mask: list[int] = []
    for key, value in cpus.items():
        exists = 0
        for cpu in mask:
            if cpu == value[0]["core"]:
                exists += 1
                break
        if exists < smt:
            mask.append(int(key))
    return {"mask": set(mask), "available": True}


def produce_cpu_sublist(scpus, smt=1):
    """Produce a CPU list with/without SMT pairs - resource leaf case"""
    cpu_list: list[dict] = []
    for value in scpus:
        exists = 0
        for cpu in cpu_list:
            if int(cpu["core"]) == int(value["core"]):
                exists += 1
                break
        if exists < smt:
            cpu_list.append(value)
    mask = []
    for cpu in cpu_list:
        mask.append(int(cpu["cpu"]))

    return {"mask": set(mask), "available": True}


def create_omp_places(resources, strategy, smt=True):
    """Parse CPU topology and generate possible CPU masks"""
    omp_places = []
    if strategy == "all":
        omp_places.append(produce_cpu_list(resources["cpus"], smt))
    elif strategy == "cores":
        for value in resources["cores"].values():
            omp_places.append(produce_cpu_sublist(value, smt))
    elif strategy == "nodes":
        for value in resources["nodes"].values():
            omp_places.append(produce_cpu_sublist(value, smt))
    else:
        raise NotImplementedError("Unknown strategy")

    return omp_places


# pylint: disable=too-few-public-methods
class OMPProcessManager:
    """OMP aware wrapper to run mp Process()"""

    def __init__(self, strategy="nodes", smt=1, mock=None, affinity=None):
        self.strategy = strategy
        self.smt = smt
        self.omp_places = []
        vllm_mask = os.environ.get("VLLM_CPU_OMP_THREADS_BIND", None)
        self.setup_omp = vllm_mask != "nobind"
        if self.setup_omp:
            omp_places = []
            if vllm_mask is not None:
                masks = []
                for spec in vllm_mask.split("|"):
                    masks.append(parse_mask(spec))
            else:
                masks = [None]
            if mock is None:
                data = subprocess.run(
                    ["lscpu", "-Je"], check=True, capture_output=True
                ).stdout
            else:
                with open(mock, mode="rb") as jf:
                    data = jf.read()
            lscpu = json.loads(data)
            for mask in masks:
                resources = enumerate_resources(lscpu, mask, affinity)
                omp_places.extend(create_omp_places(resources, strategy, smt))
            self.omp_places = sorted(
                omp_places,
                key=lambda p: "{:04d}-{:04d}".format(len(p["mask"]), max(p["mask"])),
                reverse=True,
            )

    def run(self, what, *args, **kwargs):
        """Run arg with correct OMP environment"""
        if self.setup_omp:
            for place in self.omp_places:
                if place["available"]:
                    reserve = int(os.environ.get("VLLM_CPU_NUM_OF_RESERVED_CPU", 0))
                    place["available"] = False
                    # pylint: disable=consider-using-f-string
                    os.environ["OMP_PLACES"] = "{}".format(place["mask"])
                    os.environ["OMP_NUM_THREADS"] = "{}".format(
                        len(place["mask"]) - reserve
                    )
                    os.environ["OMP_PROC_BIND"] = "TRUE"
                    return what(*args, **kwargs)
            raise IndexError("Out of OMP places")
        return what(*args, **kwargs)
