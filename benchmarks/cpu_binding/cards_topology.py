#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ==============================================================================
# cards_topology.py
#
# Provides CardsTopology class:
#   - discover all Gaudi cards via hl-smi
#   - discover all NVIDIA GPUs via nvidia-smi
#   - return NUMA node and CPU IDs per card
#
# Gaudi:
#   - hl-smi v1.22.0+ table format (HL-325L / Gaudi3)
#
# NVIDIA:
#   - inventory: nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
#   - locality (primary): /sys/bus/pci/devices/<BDF>/{numa_node,local_cpulist}
#   - locality (fallback): nvidia-smi topo -m (CPU Affinity / NUMA Affinity)
#
# ==============================================================================
import os
import shutil
import subprocess

import regex as re


class CardsTopology:
    """Utility class to discover accelerator cards and their NUMA / CPU locality."""

    def __init__(self):
        self.cards = self._discover_cards()

    # ------------------------------------------------------------------
    def _run_cmd(self, cmd: str, check: bool = True) -> str:
        """Run a shell command and return stdout."""
        try:
            result = subprocess.run(
                cmd, shell=True, check=check, capture_output=True, text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {cmd}\n{e.stderr}") from e

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_pci_bus_id(raw: str) -> str:
        """
        Normalize PCI bus id to sysfs-friendly BDF: DDDD:BB:DD.F (domain=4 hex).

        Handles:
          - "18:00.0"
          - "0000:18:00.0"
          - "00000000:18:00.0"
          - (defensive) "0000:00000000:18:00.0"
        """
        s = (raw or "").strip()

        # Fix broken double-domain: 0000:00000000:18:00.0
        m = re.fullmatch(
            r"([0-9a-fA-F]{4}):([0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2}\.[0-7])",
            s,
        )
        if m:
            _, dom8, bb, dd_fn = m.groups()
            return f"{dom8[-4:]}:{bb}:{dd_fn}".lower()

        # 8-hex domain: 00000000:18:00.0
        m = re.fullmatch(
            r"([0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2}\.[0-7])", s
        )
        if m:
            dom8, bb, dd_fn = m.groups()
            return f"{dom8[-4:]}:{bb}:{dd_fn}".lower()

        # 4-hex domain: 0000:18:00.0
        m = re.fullmatch(
            r"([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2}\.[0-7])", s
        )
        if m:
            return s.lower()

        # no domain: 18:00.0
        m = re.fullmatch(r"([0-9a-fA-F]{2}):([0-9a-fA-F]{2}\.[0-7])", s)
        if m:
            bb, dd_fn = m.groups()
            return f"0000:{bb}:{dd_fn}".lower()

        return s.lower()

    # ------------------------------------------------------------------
    @staticmethod
    def _clean_numa_node(v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if s in ("", "N/A", "-1"):
            return None
        return s

    # ------------------------------------------------------------------
    def _get_sysfs_info(self, bus_id: str) -> dict[str, str | None]:
        """Fetch NUMA node and local CPU list from sysfs."""
        sys_path = f"/sys/bus/pci/devices/{bus_id}"
        info: dict[str, str | None] = {
            "numa_node": None,
            "local_cpulist": None,
            "_sysfs_path": sys_path,
        }

        try:
            with open(os.path.join(sys_path, "numa_node"), encoding="utf-8") as f:
                info["numa_node"] = self._clean_numa_node(f.read().strip())
        except FileNotFoundError:
            pass

        try:
            with open(os.path.join(sys_path, "local_cpulist"), encoding="utf-8") as f:
                v = f.read().strip()
                info["local_cpulist"] = v if v else None
        except FileNotFoundError:
            pass

        return info

    # ------------------------------------------------------------------
    def _parse_hl_smi_table(self, text: str) -> list[dict]:
        """
        Parse hl-smi v1.22+ table format.
        Example line:
        |   0  HL-325L             N/A  | 0000:97:00.0     N/A | ...
        """
        cards = []
        pattern = re.compile(
            r"^\|\s*(\d+)\s+([A-Z0-9-]+)\s+N/A\s+\|\s*([0-9a-fA-F:.]+)\s+N/A\s*\|"
        )
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            card_id, model, bus_id = match.groups()
            cards.append(
                {
                    "vendor": "intel",
                    "card_id": int(card_id),
                    "model": model,
                    "bus_id": self._normalize_pci_bus_id(bus_id),
                }
            )
        return cards

    # ------------------------------------------------------------------
    def _discover_gaudi_cards(self) -> list[dict]:
        """Run hl-smi and discover Gaudi cards."""
        if shutil.which("hl-smi") is None:
            return []

        try:
            hl_smi_output = self._run_cmd("hl-smi")
        except Exception:
            return []

        cards = self._parse_hl_smi_table(hl_smi_output)
        for c in cards:
            c.update(self._get_sysfs_info(c["bus_id"]))
        return cards

    # ------------------------------------------------------------------
    def _parse_nvidia_smi_query(self, text: str) -> list[dict]:
        """
        Parse:
          nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
        """
        cards: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue

            idx_s, name, bus_id = parts[0], parts[1], parts[2]
            try:
                idx = int(idx_s)
            except ValueError:
                continue

            cards.append(
                {
                    "vendor": "nvidia",
                    "card_id": idx,
                    "model": name,
                    "bus_id": self._normalize_pci_bus_id(bus_id),
                }
            )
        return cards

    # ------------------------------------------------------------------
    def _parse_nvidia_topo_affinity(
        self, topo_text: str
    ) -> tuple[dict[int, str], dict[int, str]]:
        """
        Robustly parse `nvidia-smi topo -m` for CPU Affinity and NUMA Affinity.
        Uses character-column slicing based on header substring offsets.
        """
        cpu_aff: dict[int, str] = {}
        numa_aff: dict[int, str] = {}

        lines = topo_text.splitlines()
        header_i = None
        header_line = None
        for i, ln in enumerate(lines):
            if "GPU0" in ln and ("CPU Affinity" in ln or "NUMA Affinity" in ln):
                header_i = i
                header_line = ln
                break

        if header_i is None or header_line is None:
            return cpu_aff, numa_aff

        cpu_pos = header_line.find("CPU Affinity")
        numa_pos = header_line.find("NUMA Affinity")

        # Determine slice ranges
        cols = []
        if cpu_pos >= 0:
            cols.append((cpu_pos, "cpu"))
        if numa_pos >= 0:
            cols.append((numa_pos, "numa"))
        cols.sort()

        ranges = {}
        for j, (pos, name) in enumerate(cols):
            end = len(header_line) if j + 1 == len(cols) else cols[j + 1][0]
            ranges[name] = (pos, end)

        gpu_row_re = re.compile(r"^\s*GPU(\d+)\b")
        for ln in lines[header_i + 1 :]:
            m = gpu_row_re.match(ln)
            if not m:
                continue
            gi = int(m.group(1))

            if "cpu" in ranges:
                s, e = ranges["cpu"]
                v = ln[s:e].strip()
                if v and v != "N/A":
                    cpu_aff[gi] = v

            if "numa" in ranges:
                s, e = ranges["numa"]
                v = ln[s:e].strip()
                if v and v != "N/A":
                    numa_aff[gi] = v

        return cpu_aff, numa_aff

    # ------------------------------------------------------------------
    def _discover_nvidia_gpus(self) -> list[dict]:
        """Run nvidia-smi and discover NVIDIA GPUs."""
        if shutil.which("nvidia-smi") is None:
            return []

        try:
            out = self._run_cmd(
                "nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader"
            )
        except Exception:
            return []

        cards = self._parse_nvidia_smi_query(out)

        # Try to get topo-based affinity as fallback (non-fatal if it fails)
        topo_out = ""
        try:
            topo_out = self._run_cmd("nvidia-smi topo -m", check=False)
        except Exception:
            topo_out = ""

        cpu_aff_by_gpu, numa_aff_by_gpu = (
            self._parse_nvidia_topo_affinity(topo_out) if topo_out else ({}, {})
        )

        for c in cards:
            c.update(self._get_sysfs_info(c["bus_id"]))

            gi = int(c["card_id"])
            c["cpu_affinity"] = cpu_aff_by_gpu.get(gi)
            c["numa_affinity"] = numa_aff_by_gpu.get(gi)

            # Fallbacks if sysfs is missing/unknown
            if c.get("local_cpulist") is None and c.get("cpu_affinity"):
                c["local_cpulist"] = c["cpu_affinity"]

            if c.get("numa_node") is None and c.get("numa_affinity"):
                c["numa_node"] = self._clean_numa_node(c["numa_affinity"])

        return cards

    # ------------------------------------------------------------------
    def _discover_cards(self) -> list[dict]:
        """Prefer Gaudi if present; otherwise NVIDIA."""
        gaudi = self._discover_gaudi_cards()
        if gaudi:
            return gaudi

        nvidia = self._discover_nvidia_gpus()
        if nvidia:
            return nvidia

        return []

    # ------------------------------------------------------------------
    def get_cards(self) -> list[dict]:
        """Return list of all discovered cards sorted by NUMA node (then card_id)."""

        def sort_key(c):
            try:
                return (int(str(c.get("numa_node")).split("-")[0]), int(c["card_id"]))
            except Exception:
                return (999, int(c["card_id"]))

        return sorted(self.cards, key=sort_key)

    # ------------------------------------------------------------------
    def get_numa_for_card(self, card_id: int) -> str | None:
        for c in self.cards:
            if c["card_id"] == card_id:
                return c.get("numa_node")
        return None

    # ------------------------------------------------------------------
    def get_cpus_for_card(self, card_id: int) -> str | None:
        for c in self.cards:
            if c["card_id"] == card_id:
                return c.get("local_cpulist")
        return None


if __name__ == "__main__":
    topo = CardsTopology()
    cards = topo.get_cards()
    if not cards:
        print(
            "No Gaudi or NVIDIA devices discovered (hl-smi / nvidia-smi not found "
            "or no devices)."
        )

        raise SystemExit(0)

    vendor = cards[0].get("vendor", "unknown")
    print(f"Discovered vendor: {vendor} (count={len(cards)})\n")

    for card in cards:
        print(f"Card {card['card_id']} ({card['model']}):")
        print(f"  Vendor        : {card.get('vendor')}")
        print(f"  Bus ID        : {card.get('bus_id')}")
        print(f"  NUMA Node     : {card.get('numa_node')}")
        print(f"  Local CPUs    : {card.get('local_cpulist')}")
        if card.get("vendor") == "nvidia":
            print(f"  CPU Affinity  : {card.get('cpu_affinity')}")
            print(f"  NUMA Affinity : {card.get('numa_affinity')}")
            print(f"  Sysfs Path    : {card.get('_sysfs_path')}")
        print()
