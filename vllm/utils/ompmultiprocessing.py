# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OMP Aware Multiprocessing manager for running multiprocessing.Process()
Copyright (c) 2026 Red Hat Inc
Copyright (c) 2026 Cambridge Greys Ltd
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class OMPStrategy:
    """Class describing the generation strategy for OMP Places"""

    smt: int = 1
    merge: int = 1
    split: int = 1
    reserve: int = 0


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


# pylint: disable=too-few-public-methods
class OMPProcessManager:
    """OMP aware wrapper to run mp Process()"""

    def __init__(self, global_mask=None, strategy=None, mock=None, affinity=None):
        if strategy is None:
            self.strategy = OMPStrategy()
        else:
            self.strategy = strategy
        self.omp_places = []
        self.cpu_count: int | None = 1
        self.cores: dict[int, Any] | None = None
        self.nodes: dict[int, set[int]] | None = None
        self.reserved = {}

        self.setup_omp = global_mask != "nobind"

        if affinity is None:
            try:
                affinity = os.sched_getaffinity(0)
            except AttributeError:
                max_cpu = os.cpu_count()
                if max_cpu is None:
                    max_cpu = 1024  # large enough CPU mask
                affinity = set(range(0, max_cpu))  # type: ignore

        if self.setup_omp:
            if global_mask is not None:
                masks = []
                for spec in global_mask.split("|"):
                    masks.append(parse_mask(spec))
                self.create_topology(masks, affinity)
            else:
                try:
                    if mock is None:
                        data = subprocess.run(
                            ["lscpu", "-Je"], check=True, capture_output=True
                        ).stdout
                    else:
                        with open(mock, mode="rb") as jf:
                            data = jf.read()
                        # sometimes lscpu may return empty cpu, core and node lists
                    self.lscpu = json.loads(data)
                    if len(self.lscpu["cpus"]) > 0:
                        self.enumerate_resources(affinity)
                    # we set cpus to empty if we fail to parse the resource map

                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass
            if self.cores is not None and len(self.cores) > 0:
                self.fixup_topology()
                self.create_omp_places()

            if self.cores is None:
                try:
                    new_env = dict(os.environ)
                    new_env["PATH"] = new_env["PATH"] + ":/sbin:/usr/sbin"
                    data = subprocess.run(
                        ["sysctl", "-n", "hw.ncpu"],
                        check=True,
                        capture_output=True,
                        env=new_env,
                    ).stdout
                    self.cpu_count = int(data)
                    # final fallback - use python internal
                    if self.cpu_count <= 0:
                        self.cpu_count = os.cpu_count()
                    if self.cpu_count is None:
                        # runtime could not determine cpu count
                        self.cpu_count = 0
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

    def fixup_topology(self):
        """Fixup the topology to match special requirements for DP and
        simulations"""
        if self.strategy.split > 1 and self.nodes is not None:
            new_nodes = {}
            new_node_index = 0
            for cpulist in self.nodes.values():
                step = int(len(cpulist) / self.strategy.split)
                if step > 1:
                    for new_range in range(0, len(cpulist), step):
                        new_nodes[new_node_index] = set(
                            list(cpulist)[new_range : new_range + step]
                        )
                        new_node_index += 1
            self.nodes = new_nodes

        if self.strategy.merge > 1:
            new_nodes = {}
            new_node_index = 0
            node_list = sorted(self.nodes.keys())  # type:ignore
            for nodenum in range(0, len(node_list), self.strategy.merge):
                new_node: set[int] = set()
                for node in node_list[nodenum : nodenum + self.strategy.merge]:
                    new_node |= self.nodes[node]  # type:ignore
                new_nodes[new_node_index] = new_node
                new_node_index += 1
            self.nodes = new_nodes

    def create_topology(self, masks, affinity):
        """Create a topology from a specified mask list.
        We fake a core per each CPU set in this case and
        a node per each group of cpus.
        """

        node = 0
        self.cores = {}
        self.nodes = {}
        for mask in masks:
            for cpu in mask & affinity:
                if self.cores.get(cpu, None) is None:
                    self.cores[cpu] = []
                self.cores[cpu].append(
                    {
                        "cpu": cpu,
                        "node": node,
                        "socket": 0,
                        "core": cpu,
                        "online": True,
                    }
                )
                if self.nodes.get(node, None) is None:
                    self.nodes[node] = set([cpu])
                else:
                    self.nodes[node] |= set([cpu])
            # disabling ruff's desires to use enumerate, it makes the code unreadable
            node += 1  # noqa: SIM113

    def create_omp_places(self):
        """Parse CPU topology and generate possible CPU masks"""

        if self.nodes is not None:
            for node in sorted(self.nodes.keys(), reverse=True):
                cpulist = []
                for core in self.nodes[node]:
                    threads = []
                    for cpu in self.cores[core]:  # type: ignore
                        threads.append(int(cpu["cpu"]))
                    cpulist.extend(sorted(threads)[: self.strategy.smt])

                if self.strategy.reserve > 0 and self.strategy.reserve < len(cpulist):
                    self.reserved[node] = cpulist[-self.strategy.reserve :]
                    cpulist = cpulist[: -self.strategy.reserve]

                if len(cpulist) > 0:
                    self.omp_places.append(
                        {"mask": set(sorted(cpulist)), "available": True}
                    )

    def _detect_locality_key(self):
        """Detect the best locality grouping key from lscpu data.

        On most systems, NUMA "node" provides the right grouping. But on
        S390X, all CPUs may report a single NUMA node while having
        finer-grained locality boundaries via "book". When that's the
        case, use the book field instead.
        """
        cpus = self.lscpu.get("cpus", [])
        if not cpus:
            return "node"

        # Check if there's more than one NUMA node — if so, node is fine
        nodes = set()
        for cpu in cpus:
            nodes.add(_int(cpu.get("node", 0)))
        if len(nodes) > 1:
            return "node"

        # Single NUMA node — check for S390X book-based topology
        books = set()
        for cpu in cpus:
            if "book" in cpu:
                books.add(_int(cpu["book"]))
        if len(books) > 1:
            return "book"

        return "node"

    def enumerate_resources(self, allowed):
        """Enumerate system resources"""

        try:
            allowed_nodes = parse_mask(os.environ["CPU_VISIBLE_MEMORY_NODES"])
        except KeyError:
            allowed_nodes = None

        group_key = self._detect_locality_key()

        self.cores = {}
        self.nodes = {}

        for cpu in self.lscpu["cpus"]:
            cpunum = int(cpu["cpu"])
            if (
                cpunum in allowed
                and cpunum >= 0
                and (allowed_nodes is None or _int(cpu["node"]) in allowed_nodes)
            ):
                core = _int(cpu["core"])
                if self.cores.get(core, None) is None:
                    self.cores[core] = [cpu]
                else:
                    self.cores[core].append(cpu)

                group = _int(cpu.get(group_key, cpu["node"]))
                if self.nodes.get(group, None) is None:
                    self.nodes[group] = set([core])
                else:
                    self.nodes[group] |= set([core])

    def compute_cpus(self):
        """How many compute CPUs are available"""
        if len(self.omp_places) > 0:
            return sum(map(lambda p: len(p["mask"]), self.omp_places))
        return self.cpu_count

    def total_cpus(self):
        """How many CPUs are in total"""
        # mypy completely loses the plot on the typing of this one, so override
        return sum(map(lambda c: len(c), self.cores.values()))  # type: ignore

    def run(self, what, *args, **kwargs):
        """Run arg with correct OMP environment"""
        if self.setup_omp:
            if self.cores is not None:
                for place in self.omp_places:
                    if place["available"]:
                        place["available"] = False
                        # pylint: disable=consider-using-f-string
                        os.environ["OMP_PLACES"] = "{}".format(place["mask"])
                        os.environ["OMP_NUM_THREADS"] = "{}".format(len(place["mask"]))
                        os.environ["OMP_PROC_BIND"] = "TRUE"
                        return what(*args, **kwargs)
                raise IndexError("Out of OMP places")
            # cores == None, we have no clue about topology
            if self.cpu_count is not None and self.cpu_count > 0:
                os.environ["OMP_NUM_THREADS"] = f"{self.cpu_count}"
        return what(*args, **kwargs)
