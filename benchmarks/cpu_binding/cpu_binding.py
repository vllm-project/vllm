# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import os
from enum import Enum
from importlib import util
from pathlib import Path

from cards_topology import CardsTopology

REQUIRED_COLUMNS = [
    "model_id",
    "input_length",
    "output_length",
    "world_size",
    "data_type",
    "num_allocated_cpu",
]


class BindingPolicy(Enum):
    Evenly_on_NUMAs = "evenly"
    NUMAs_with_cards = "close2cards"


class CPU_Binding:
    """
    CPU binding helper supporting:
      - NUMA selection policies (evenly vs close-to-cards)
      - PCT (Priority Core Turbo / CLOS cpulist) "preferred CPUs first" ordering

    PCT integration:
      - Reads a Linux cpulist string from a file (default):
          priority_core_turbo/results/clos0_cpulist.txt
      - Reorders per-NUMA CPU lists so "preferred" CPUs are consumed first.
      - Uses "first N" selection from the ranked list.
    """

    def __init__(
        self, csv_path: str = "cpu_binding_gnr.csv", use_hyperthread: bool = False
    ):
        self.libnuma_found = util.find_spec("numa") is not None
        self.psutil_found = util.find_spec("psutil") is not None

        if self.libnuma_found and self.psutil_found:
            import psutil
            from numa import info

            # Get system Info
            self.cpu_count = psutil.cpu_count(logical=False)
            self.cpus_allow_list = psutil.Process().cpu_affinity()
            self.numa_size = info.get_num_configured_nodes()
            self.cpu_count_per_numa = self.cpu_count // self.numa_size

            # Get CSV info
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows or any(col not in rows[0] for col in REQUIRED_COLUMNS):
                found = list(rows[0].keys()) if rows else "EMPTY CSV"
                raise ValueError(
                    f"CSV missing required headers {REQUIRED_COLUMNS}. Found: {found}"
                )

            model = os.environ.get("MODEL")
            if not model:
                raise RuntimeError(
                    "Set environment variable MODEL to a model_id in the CSV."
                )
            input_tok = os.environ.get("INPUT_TOK")
            output_tok = os.environ.get("OUTPUT_TOK")
            con_req = os.environ.get("CONCURRENT_REQ")
            num_allocated_cpu = os.environ.get("NUM_CPUS")

            row = self.pick_row_by_parameters(
                rows, model, input_tok, output_tok, con_req
            )

            self.world_size = self.parse_int(row["world_size"], "world_size")
            binding_policy_index = self.parse_int(
                row["binding_policy"], "binding_policy"
            )
            self.binding_policy = list(BindingPolicy)[binding_policy_index]

            if num_allocated_cpu:
                self.num_allocated_cpu = int(num_allocated_cpu)
            elif row["num_allocated_cpu"] == "NA":
                raise RuntimeError(
                    "Invalid NUM_CPU value. Set environment variable NUM_CPUS instead ."
                )
            else:
                self.num_allocated_cpu = self.parse_int(
                    row["num_allocated_cpu"], "num_allocated_cpu"
                )

            # CPU: build allow-listed per-NUMA CPU lists (one logical CPU per core)
            self.node_to_cpus = []
            for i in range(self.numa_size):
                filtered_node_to_cpus = self.filter_one_cpu_per_core(
                    info.node_to_cpus(i)
                )
                node_intersect = [
                    cpu for cpu in filtered_node_to_cpus if cpu in self.cpus_allow_list
                ]
                if bool(node_intersect):
                    self.node_to_cpus.append(list(node_intersect))

            # -----------------------------
            # PCT preferred CPU integration
            # -----------------------------
            self.pct_preferred_cpus = self._load_pct_preferred_cpus_from_file()

            # Build ranked list per NUMA: PCT preferred CPUs first, then the rest
            self.node_to_cpus_ranked = []
            for node_cpus in self.node_to_cpus:
                pct_first = [c for c in node_cpus if c in self.pct_preferred_cpus]
                rest = [c for c in node_cpus if c not in self.pct_preferred_cpus]
                self.node_to_cpus_ranked.append(pct_first + rest)

            if self.pct_preferred_cpus:
                print(
                    f"PCT enabled. Preferred CPU count={len(self.pct_preferred_cpus)} "
                    f"(head={sorted(self.pct_preferred_cpus)[:16]})"
                )

            # Idle CPUs: initialize from per-NUMA CPU list;
            # drop HT or non-HT according to use_hyperthread
            # Note: we keep idle pool aligned with node_to_cpus (not ranked)
            # because we remove allocated CPUs anyway.
            self.node_to_idle_cpus = [lst.copy() for lst in self.node_to_cpus]
            for i in range(len(self.node_to_idle_cpus)):
                if use_hyperthread is False:
                    self.node_to_idle_cpus[i] = self.node_to_cpus[i][
                        : self.cpu_count_per_numa
                    ]
                else:
                    self.node_to_idle_cpus[i] = self.node_to_cpus[i][
                        self.cpu_count_per_numa :
                    ]

            # Gaudi
            topo = CardsTopology()
            self.cards = topo.get_cards()
            if self.cards is not None:
                self.card_numa_list = []
                # Assume to use cards from 0 to 7
                for card in self.cards[: self.world_size]:
                    if card["numa_node"] not in self.card_numa_list:
                        self.card_numa_list.append(card["numa_node"])
                        print(f"Card {card['card_id']} ({card['model']}):")
                        print(f"  Bus ID     : {card['bus_id']}")
                        print(f"  NUMA Node  : {card['numa_node']}")
                        print(f"  Local CPUs : {card['local_cpulist']}")
                    else:
                        print(
                            f"NOT Append Card {card['card_id']} ({card['model']}) "
                            f"{card['numa_node']}:"
                        )

        else:
            # Minimal init to avoid attribute errors if features unavailable
            self.cpu_count = 0
            self.cpus_allow_list = []
            self.numa_size = 1
            self.cpu_count_per_numa = 0
            self.world_size = 1
            self.binding_policy = BindingPolicy.Evenly_on_NUMAs
            self.num_allocated_cpu = 0
            self.node_to_cpus = []
            self.node_to_cpus_ranked = []
            self.node_to_idle_cpus = []
            self.cards = None
            self.card_numa_list = []

    def _get_env_bool(self, name: str, default: bool = False) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        return v.strip().lower() in ("1", "true", "yes", "y", "on")

    def _parse_cpulist(self, s: str) -> set[int]:
        """
        Parse Linux cpulist format like: "0-3,8,10-12" into a set of ints.
        """
        s = (s or "").strip()
        if not s:
            return set()

        out: set[int] = set()
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a_i, b_i = int(a), int(b)
                if b_i < a_i:
                    a_i, b_i = b_i, a_i
                out.update(range(a_i, b_i + 1))
            else:
                out.add(int(part))
        return out

    def _load_pct_preferred_cpus_from_file(self) -> set[int]:
        """
        Load preferred CPUs from a cpulist file, e.g.
          priority_core_turbo/results/clos0_cpulist.txt

        Controls:
          - PCT_ENABLE (default: 1)
          - PCT_CPULIST_FILE (default: repo-relative to this script)
        """
        if not self._get_env_bool("PCT_ENABLE", True):
            return set()

        default_pct = (
            Path(__file__).resolve().parent
            / "priority_core_turbo"
            / "results"
            / "clos0_cpulist.txt"
        )
        pct_path = os.environ.get("PCT_CPULIST_FILE", str(default_pct))

        p = Path(pct_path)
        if not p.exists():
            print(
                f"Warning: PCT cpulist file not found: {pct_path}. "
                "Falling back to non-PCT ordering."
            )
            return set()

        try:
            content = p.read_text(encoding="utf-8")
        except Exception as e:
            print(
                f"Warning: failed to read PCT cpulist file '{pct_path}': {e}."
                "Falling back to non-PCT ordering."
            )
            return set()

        # Accept first non-empty line (covers both single-line and multi-line files)
        line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
        return self._parse_cpulist(line)

    def parse_int(self, v: str, name: str) -> int:
        try:
            return int(v)
        except Exception as err:
            raise ValueError(f"Invalid integer for {name!r}: {v!r}") from err

    def pick_row_by_parameters(
        self,
        rows: list[dict],
        model: str,
        input_tok: str,
        output_tok: str,
        con_req: str,
    ) -> dict:
        matches = [
            r
            for r in rows
            if r.get("model_id", "").strip() == (model or "").strip()
            if r.get("input_length", "").strip() == (input_tok or "").strip()
            if r.get("output_length", "").strip() == (output_tok or "").strip()
        ]
        if not matches:
            # fallback: match only by model_id
            matches = [
                r
                for r in rows
                if r.get("model_id", "").strip() == (model or "").strip()
            ]
            print(
                f"Warning: using fallback entry for model '{model}'"
                " without exact input/output token match"
            )
        if not matches:
            available = ", ".join(sorted({r.get("model_id", "") for r in rows}))
            raise ValueError(
                f"MODEL '{model}', input_length '{input_tok}', "
                f"output_length '{output_tok}' "
                f"not found in CSV. Available: {available}"
            )
        return matches[0]

    def filter_one_cpu_per_core(self, cpus):
        """
        Given a list of CPU IDs (possibly with HT pairs),
        return a filtered list with only one logical CPU per physical core.
        """
        seen_cores = set()
        filtered = []
        for cpu in sorted(cpus):
            core_path = f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id"
            try:
                with open(core_path) as f:
                    core_id = int(f.read().strip())
            except FileNotFoundError:
                continue
            if core_id not in seen_cores:
                seen_cores.add(core_id)
                filtered.append(cpu)
        return filtered

    def get_cpus_id_binding_based_on_numa_nodes(self, rank: int) -> str:
        """Return CPUs id binding based on NUMA nodes."""
        rank_to_cpus = ""
        if not self.libnuma_found or not self.psutil_found:
            print(
                "Auto thread-binding is not supported due to "
                "the lack of package numa and psutil,"
                "fallback to no thread-binding. To get better performance,"
                "please try to manually bind threads."
            )
            return rank_to_cpus

        if self.binding_policy is BindingPolicy.Evenly_on_NUMAs or self.cards is None:
            self.allocated_cpu_per_numa = self.num_allocated_cpu // len(
                self.node_to_cpus
            )
            node_id = rank
        elif self.binding_policy is BindingPolicy.NUMAs_with_cards:
            self.allocated_cpu_per_numa = self.num_allocated_cpu // len(
                self.card_numa_list
            )
            numa_node_str = self.cards[rank]["numa_node"]
            if numa_node_str is None:
                raise ValueError(
                    f"Card {self.cards[rank]['card_id']} is "
                    "missing NUMA node information, "
                    "required for 'close2cards' policy."
                )

            node_id = int(numa_node_str)
        else:
            # safety fallback
            self.allocated_cpu_per_numa = self.num_allocated_cpu // len(
                self.node_to_cpus
            )
            node_id = rank

        print(
            f"binding numa node_id {node_id}  allocated_cpu_per_numa "
            f"{self.allocated_cpu_per_numa}"
        )

        # PCT-first selection: take first N from ranked list
        ranked = (
            self.node_to_cpus_ranked[node_id]
            if self.node_to_cpus_ranked
            else self.node_to_cpus[node_id]
        )
        rank_to_cpus_list = ranked[: self.allocated_cpu_per_numa]

        rank_to_cpus = ",".join(str(x) for x in rank_to_cpus_list)
        print(f"rank {rank} auto thread-binding list: {rank_to_cpus}")

        # Remove allocated CPUs from idle pool
        self.node_to_idle_cpus[node_id] = [
            cpu
            for cpu in self.node_to_idle_cpus[node_id]
            if cpu not in rank_to_cpus_list
        ]
        return rank_to_cpus


if __name__ == "__main__":
    libnuma_found = util.find_spec("numa") is not None
    if libnuma_found:
        from numa import info

        numa_size = info.get_num_configured_nodes()
    else:
        numa_size = 1

    world_size = numa_size
    cpu_binder = CPU_Binding(use_hyperthread=False)

    if (
        cpu_binder.binding_policy is BindingPolicy.Evenly_on_NUMAs
        or cpu_binder.cards is None
    ):
        max_needed_numa_size = len(cpu_binder.node_to_cpus)
    elif cpu_binder.binding_policy is BindingPolicy.NUMAs_with_cards:
        max_needed_numa_size = min(cpu_binder.world_size, len(cpu_binder.node_to_cpus))
    else:
        max_needed_numa_size = len(cpu_binder.node_to_cpus)

    for i in range(max_needed_numa_size):
        rank_to_cpus = cpu_binder.get_cpus_id_binding_based_on_numa_nodes(i)
        print(rank_to_cpus)

    rank_to_idle_cpus = ",".join(
        str(x) for row in cpu_binder.node_to_idle_cpus for x in row
    )
    print(rank_to_idle_cpus)
    for r in cpu_binder.node_to_idle_cpus:
        print(len(r))
