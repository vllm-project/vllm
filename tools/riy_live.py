#!/usr/bin/env python3
"""
riy live — Live MoE Expert Activation Dashboard
Part of vllm-riy: Pruning on Air

Usage:
    python riy_live.py
    python riy_live.py --host localhost --port 8001
    python riy_live.py --interval 1.0 --block 16
    python riy_live.py --demo   # synthetic data, no vLLM needed

Keybindings:
    q         quit
    r         reset stats (POST /admin/expert_stats/reset)
    e         enable stats collection
    d         disable stats collection
    +/-       increase/decrease refresh interval
    [/]       decrease/increase block size
    h/l       scroll left/right (expert blocks)
    j/k       scroll up/down (layers)
    m         toggle mask overlay (show pruned experts as X)
    s         save current stats to stats.json
    p         export prune profile (prompted for threshold)
    ?         toggle help
"""

import curses
import time
import json
import threading
import sys
import os
import math
import argparse
import random
import urllib.request
import urllib.error
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExpertData:
    frequency: float = 0.0
    avg_gate:  float = 0.0
    count:     int   = 0

@dataclass
class Stats:
    token_total: int = 0
    experts: dict = field(default_factory=dict)  # (layer, expert) -> ExpertData
    max_layer:  int = 0
    max_expert: int = 0
    num_shared: int = 0
    timestamp:  float = 0.0
    error: Optional[str] = None
    model_name: str = ""

    @property
    def max_freq(self):
        vals = [v.frequency for v in self.experts.values()]
        return max(vals) if vals else 1.0

    @property
    def max_gate(self):
        vals = [v.avg_gate for v in self.experts.values() if v.avg_gate > 0]
        return max(vals) if vals else 1.0


# ── HTTP fetch ────────────────────────────────────────────────────────────────

def fetch_stats(host, port, timeout=3.0) -> Stats:
    url = f"http://{host}:{port}/riy/stats"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        s = Stats(); s.error = str(e); return s
    except Exception as e:
        s = Stats(); s.error = str(e); return s

    s = Stats()
    s.token_total = data.get("token_total", 0)
    s.timestamp = time.time()

    num_experts = data.get("num_experts", 0)
    layers = data.get("layers", [])

    for layer_data in layers:
        l = layer_data["layer"]
        freq_list = layer_data.get("freq", [])
        weight_list = layer_data.get("weight_sum", [])
        total_freq = sum(freq_list) if freq_list else 1
        for e in range(len(freq_list)):
            f = freq_list[e]
            w = weight_list[e] if e < len(weight_list) else 0.0
            s.experts[(l, e)] = ExpertData(
                frequency=f / total_freq if total_freq > 0 else 0.0,
                avg_gate=w / f if f > 0 else 0.0,
                count=int(f),
            )

    if s.experts:
        s.max_layer  = max(l for l, e in s.experts)
        s.max_expert = max(e for l, e in s.experts)
        s.num_shared = _detect_shared(s.experts)

    return s


def post_reset(host, port):
    url = f"http://{host}:{port}/riy/stats/reset"
    try:
        req = urllib.request.Request(url, data=b'', method='POST')
        urllib.request.urlopen(req, timeout=3.0)
        return True
    except:
        return False


def post_enable(host, port, enable=True):
    endpoint = "start" if enable else "stop"
    url = f"http://{host}:{port}/riy/stats/{endpoint}"
    try:
        req = urllib.request.Request(url, data=b'', method='POST')
        urllib.request.urlopen(req, timeout=3.0)
        return True
    except:
        return False


def _detect_shared(experts):
    freq_by_expert = defaultdict(list)
    for (l, e), v in experts.items():
        freq_by_expert[e].append(v.frequency)
    shared = 0
    for e in sorted(freq_by_expert.keys()):
        avg = sum(freq_by_expert[e]) / max(len(freq_by_expert[e]), 1)
        if avg > 0.8:
            shared += 1
        else:
            break
    return shared


# ── Demo / synthetic data ─────────────────────────────────────────────────────

class DemoSource:
    """Simulates live vLLM expert stats with animated activity."""

    def __init__(self, num_layers=94, num_shared=1, num_routed=128):
        self.num_layers  = num_layers
        self.num_shared  = num_shared
        self.num_routed  = num_routed
        self.token_total = 0
        self._base = {}
        self._noise_phase = 0.0
        random.seed(42)

        # Build base profile
        for layer in range(num_layers):
            for e in range(num_shared):
                self._base[(layer, e)] = (random.uniform(0.85, 1.0),
                                           random.uniform(0.6, 1.0))
            for e in range(num_shared, num_shared + num_routed):
                r = random.random()
                if   r < 0.15: freq, gate = 0.0,                           0.0
                elif r < 0.45: freq, gate = random.uniform(0.0001, 0.002), random.uniform(0.0, 0.3)
                elif r < 0.70: freq, gate = random.uniform(0.002,  0.01),  random.uniform(0.1, 0.5)
                elif r < 0.88: freq, gate = random.uniform(0.01,   0.05),  random.uniform(0.3, 0.7)
                else:          freq, gate = random.uniform(0.05,   0.15),  random.uniform(0.5, 1.0)
                self._base[(layer, e)] = (freq, gate)

    def fetch(self) -> Stats:
        self._noise_phase += 0.3
        self.token_total  += random.randint(500, 2000)

        s = Stats()
        s.token_total = self.token_total
        s.timestamp   = time.time()
        s.num_shared  = self.num_shared

        for (layer, e), (base_freq, base_gate) in self._base.items():
            noise = 0.0
            wave = math.sin(self._noise_phase + layer * 0.3 + e * 0.07)
            if base_freq > 0.001:
                noise = wave * base_freq * 0.4

            freq = max(0.0, min(1.0, base_freq + noise))
            gate = max(0.0, min(1.0, base_gate + wave * 0.1
                                  if base_freq > 0.001 else 0.0))

            s.experts[(layer, e)] = ExpertData(
                frequency=freq,
                avg_gate=gate,
                count=int(freq * self.token_total),
            )

        s.max_layer  = self.num_layers - 1
        s.max_expert = self.num_shared + self.num_routed - 1
        return s


# ── Color mapping ─────────────────────────────────────────────────────────────

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    if curses.COLORS >= 256:
        curses.init_pair(1, 236, -1)   # dead / very low
        curses.init_pair(2, 31,  -1)   # low  (cyan-ish)
        curses.init_pair(3, 40,  -1)   # mid  (green)
        curses.init_pair(4, 214, -1)   # high (orange)
        curses.init_pair(5, 196, -1)   # max  (red)
        curses.init_pair(6, 240, -1)   # masked
        curses.init_pair(7, 250, -1)   # header
        curses.init_pair(8, 226, -1)   # flash (bright yellow)
        curses.init_pair(9, 232, 250)  # status bar (dark on grey)
    else:
        curses.init_pair(1, curses.COLOR_WHITE,   -1)
        curses.init_pair(2, curses.COLOR_CYAN,    -1)
        curses.init_pair(3, curses.COLOR_GREEN,   -1)
        curses.init_pair(4, curses.COLOR_YELLOW,  -1)
        curses.init_pair(5, curses.COLOR_RED,     -1)
        curses.init_pair(6, curses.COLOR_BLACK,   -1)
        curses.init_pair(7, curses.COLOR_WHITE,   -1)
        curses.init_pair(8, curses.COLOR_YELLOW,  -1)
        curses.init_pair(9, curses.COLOR_BLACK,   curses.COLOR_WHITE)


def _log_normalize(value, max_value):
    """Log-scale normalization: makes rare experts visible.

    Maps [0, max_value] -> [0.0, 1.0] using log scale.
    A linear scale hides everything below 10% of max;
    log scale spreads the low end so pruning candidates
    are distinguishable from truly dead experts.
    """
    if max_value <= 0 or value <= 0:
        return 0.0
    # log(1 + x) / log(1 + max) gives 0..1 with log compression
    return math.log1p(value) / math.log1p(max_value)


def gate_color_pair(norm):
    if   norm <= 0.0:  return curses.color_pair(1)
    elif norm < 0.15:  return curses.color_pair(1) | curses.A_DIM
    elif norm < 0.35:  return curses.color_pair(2)
    elif norm < 0.55:  return curses.color_pair(3)
    elif norm < 0.75:  return curses.color_pair(4)
    else:              return curses.color_pair(5) | curses.A_BOLD


def freq_char(freq, max_freq):
    if max_freq <= 0 or freq <= 0:
        return '\u00b7'
    r = _log_normalize(freq, max_freq)
    chars = '\u00b7\u2591\u2592\u2593\u2588'
    return chars[min(int(r * (len(chars) - 1) + 0.5), len(chars) - 1)]


# ── Main dashboard ────────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self, host, port, interval, block_size, demo=False,
                 vllm_port=8011):
        self.host       = host
        self.port       = port
        self.vllm_port  = vllm_port
        self.interval   = interval
        self.block_size = block_size  # 0 = auto (fit to terminal width)
        self.demo       = demo
        self.model_name = ""

        self.demo_src   = DemoSource() if demo else None
        self.current    = Stats()
        self.previous   = Stats()
        self.mask       = set()
        self.show_mask  = False
        self.show_help  = False
        self.status_msg = ""
        self.status_ts  = 0.0

        self.scroll_x   = 0
        self.scroll_y   = 0
        self.lock       = threading.Lock()
        self.running    = True
        self.flash_set  = set()

    def _fetch_model_name(self):
        """Fetch model name from vLLM /v1/models endpoint."""
        try:
            url = f"http://{self.host}:{self.vllm_port}/v1/models"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("data", [])
                if models:
                    self.model_name = models[0].get("id", "")
        except Exception:
            pass

    def fetch_loop(self):
        self._fetch_model_name()
        if not self.demo:
            post_enable(self.host, self.port, True)
        while self.running:
            new = self.demo_src.fetch() if self.demo else fetch_stats(self.host, self.port)
            new.model_name = self.model_name
            with self.lock:
                self.previous = self.current
                self.current  = new
                self._compute_flash()
            time.sleep(self.interval)

    def _compute_flash(self):
        self.flash_set = set()
        prev = self.previous
        curr = self.current
        if not prev.experts or not curr.experts:
            return
        for key, cv in curr.experts.items():
            pv = prev.experts.get(key)
            if pv is None:
                continue
            delta = cv.frequency - pv.frequency
            if delta > curr.max_freq * 0.1:
                self.flash_set.add(key)

    def set_status(self, msg, duration=2.0):
        self.status_msg = msg
        self.status_ts  = time.time() + duration

    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)
        init_colors()

        t = threading.Thread(target=self.fetch_loop, daemon=True)
        t.start()

        while self.running:
            key = stdscr.getch()
            self._handle_key(key, stdscr)

            with self.lock:
                stats = self.current
                flash = self.flash_set.copy()

            self._draw(stdscr, stats, flash)
            stdscr.refresh()

    def _handle_key(self, key, stdscr):
        if key == curses.KEY_RESIZE:
            stdscr.clear()
            return
        if key in (ord('q'), ord('Q')):
            self.running = False
        elif key == ord('r'):
            if self.demo:
                self.demo_src.token_total = 0
                self.set_status("Stats reset (demo)")
            elif post_reset(self.host, self.port):
                self.set_status("Stats reset")
            else:
                self.set_status("Reset failed")
        elif key == ord('e'):
            post_enable(self.host, self.port, True)
            self.set_status("Stats enabled")
        elif key == ord('d'):
            post_enable(self.host, self.port, False)
            self.set_status("Stats disabled")
        elif key in (ord('+'), ord('=')):
            self.interval = min(10.0, self.interval + 0.5)
            self.set_status(f"Interval: {self.interval:.1f}s")
        elif key == ord('-'):
            self.interval = max(0.2, self.interval - 0.5)
            self.set_status(f"Interval: {self.interval:.1f}s")
        elif key == ord(']'):
            self.block_size = min(64, self.block_size + 8)
            self.set_status(f"Block: {self.block_size}")
        elif key == ord('['):
            self.block_size = max(4, self.block_size - 8)
            self.set_status(f"Block: {self.block_size}")
        elif key in (ord('h'), curses.KEY_LEFT):
            self.scroll_x = max(0, self.scroll_x - 1)
        elif key in (ord('l'), curses.KEY_RIGHT):
            self.scroll_x += 1
        elif key in (ord('j'), curses.KEY_DOWN):
            self.scroll_y += 1
        elif key in (ord('k'), curses.KEY_UP):
            self.scroll_y = max(0, self.scroll_y - 1)
        elif key == ord('m'):
            self.show_mask = not self.show_mask
            self.set_status(f"Mask overlay: {'on' if self.show_mask else 'off'}")
        elif key == ord('s'):
            self._save_stats()
        elif key == ord('?'):
            self.show_help = not self.show_help

    def _save_stats(self):
        with self.lock:
            stats = self.current
        if not stats.experts:
            self.set_status("No data to save")
            return
        out = {
            "token_total": stats.token_total,
            "experts": {
                f"{l},{e}": {
                    "count": v.count,
                    "frequency": v.frequency,
                    "avg_gate": v.avg_gate
                }
                for (l, e), v in stats.experts.items()
            }
        }
        path = "riy_stats_export.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        self.set_status(f"Saved -> {path}")

    def _draw(self, stdscr, stats, flash):
        h, w = stdscr.getmaxyx()
        stdscr.erase()

        if self.show_help:
            self._draw_help(stdscr, h, w)
            return

        # Header
        connected = "DEMO" if self.demo else f"{self.host}:{self.port}"
        err = f" ERR:{stats.error[:30]}" if stats.error else ""
        age = f"{time.time()-stats.timestamp:.1f}s ago" if stats.timestamp else "---"
        model = stats.model_name or "?"
        hdr = (f" riy live | {model} | {connected}{err} | "
               f"layers:{stats.max_layer+1} exp:{stats.max_expert+1} | "
               f"{self.interval:.1f}s | {age} | ?=help ")
        try:
            stdscr.addstr(0, 0, hdr[:w-1], curses.color_pair(9) | curses.A_BOLD)
        except:
            pass

        if stats.error and not self.demo:
            msg = f" Cannot reach {self.host}:{self.port} -- {stats.error}"
            try: stdscr.addstr(2, 2, msg[:w-3], curses.color_pair(5))
            except: pass
            return

        if not stats.experts:
            try: stdscr.addstr(2, 2, " Waiting for data...", curses.color_pair(7))
            except: pass
            return

        # Column layout — 1 char per expert, auto-fit to terminal width
        num_shared = stats.num_shared
        routed_start = num_shared
        routed_all = list(range(routed_start, stats.max_expert + 1))
        prefix_w = 8  # "  L0094 "
        shared_w = num_shared + 1 if num_shared else 0  # 1 char each + separator
        avail_w = w - prefix_w - shared_w - 2  # available for routed experts
        effective_block = self.block_size if self.block_size > 0 else max(avail_w, 1)

        blocks = [routed_all[i:i+effective_block]
                  for i in range(0, len(routed_all), effective_block)]

        self.scroll_x = min(self.scroll_x, max(0, len(blocks) - 1))
        block_experts = blocks[self.scroll_x] if blocks else []
        num_blocks = len(blocks)

        # Column header — two rows: tens/hundreds on row 1, ones on row 2
        row = 1
        try:
            tens_hdr = "".join(
                str((e // 10) % 10) if e % 10 == 0 else " "
                for e in block_experts)
            ones_hdr = "".join(str(e % 10) for e in block_experts)
            block_info = (f" [{block_experts[0] if block_experts else 0}-"
                          f"{block_experts[-1] if block_experts else 0}]"
                          f" {self.scroll_x+1}/{num_blocks}")
            # Row 1: tens + block info
            hdr_prefix = f"{'':>{prefix_w}}"
            if num_shared:
                hdr_prefix += "|" + " " * num_shared
            stdscr.addstr(row, 0,
                          (hdr_prefix + "|" + tens_hdr + block_info)[:w-1],
                          curses.color_pair(7))
            # Row 2: ones
            row = 2
            hdr_prefix2 = f"{'':>{prefix_w}}"
            if num_shared:
                shared_ones = "".join(str(e % 10) for e in range(num_shared))
                hdr_prefix2 += "|" + shared_ones
            stdscr.addstr(row, 0,
                          (hdr_prefix2 + "|" + ones_hdr)[:w-1],
                          curses.color_pair(7))
        except:
            pass

        # Layer rows
        max_rows = h - 6  # header(1) + col_hdr(2) + summary + legend + status + prune
        num_layers = stats.max_layer + 1
        self.scroll_y = min(self.scroll_y, max(0, num_layers - max_rows))

        mf = stats.max_freq
        mg = stats.max_gate

        for i in range(max_rows):
            layer = i + self.scroll_y
            if layer > stats.max_layer:
                break
            row = i + 3  # after header(1) + 2 col_hdr rows

            line_prefix = f"  L{layer:04d} "
            try:
                stdscr.addstr(row, 0, line_prefix, curses.color_pair(7))
            except:
                pass

            col = len(line_prefix)

            if num_shared:
                try: stdscr.addstr(row, col, "|", curses.color_pair(7))
                except: pass
                col += 1
                for e in range(num_shared):
                    v = stats.experts.get((layer, e), ExpertData())
                    self._draw_expert(stdscr, row, col, layer, e, v,
                                      mf, mg, flash, shared=True)
                    col += 1  # 1 char per expert

            try: stdscr.addstr(row, col, "|", curses.color_pair(7))
            except: pass
            col += 1

            for e in block_experts:
                v = stats.experts.get((layer, e), ExpertData())
                self._draw_expert(stdscr, row, col, layer, e, v,
                                  mf, mg, flash, shared=False)
                col += 1  # 1 char per expert
                if col >= w - 1:
                    break

        # Summary statistics (REAP-style)
        summary_row = h - 4
        try:
            total_pairs = stats.max_layer * stats.max_expert if stats.max_layer > 0 else 1
            all_freqs = []
            all_gates = []
            dead = 0
            rare = 0      # freq > 0 but < 1% of max
            low = 0       # freq 1-10% of max
            active = 0    # freq > 10% of max
            high_gate_low_freq = 0  # specialists: low freq but high contribution
            for (l, e), v in stats.experts.items():
                all_freqs.append(v.frequency)
                all_gates.append(v.avg_gate)
                if v.frequency <= 0:
                    dead += 1
                elif v.frequency < mf * 0.01:
                    rare += 1
                    if mg > 0 and v.avg_gate > mg * 0.3:
                        high_gate_low_freq += 1
                elif v.frequency < mf * 0.1:
                    low += 1
                else:
                    active += 1

            pct_dead = dead * 100 // max(len(stats.experts), 1)
            pct_prunable = (dead + rare) * 100 // max(len(stats.experts), 1)

            summary = (
                f" dead:{dead}({pct_dead}%)"
                f"  rare:{rare}"
                f"  low:{low}"
                f"  active:{active}"
                f"  |  prunable:{dead+rare}({pct_prunable}%)"
                f"  specialists:{high_gate_low_freq}"
                f"  |  total:{len(stats.experts)}"
            )
            stdscr.addstr(summary_row, 0, summary[:w-1], curses.color_pair(7))
        except:
            pass

        # Legend
        legend_row = h - 3
        legend = (" freq: \u00b7 dead  "
                  "\u2591 rare  "
                  "\u2592 low  "
                  "\u2593 high  "
                  "\u2588 dominant  |  color=contribution: ")
        try:
            stdscr.addstr(legend_row, 0, legend, curses.color_pair(7))
            lc = len(legend)
            for pair, label in [(1,"low"),(2,"\u00b7"),(3,"mid"),(4,"\u00b7"),(5,"high")]:
                stdscr.addstr(legend_row, lc, "\u25a0", curses.color_pair(pair))
                lc += 1
        except:
            pass

        # Status bar
        status = self.status_msg if time.time() < self.status_ts else (
            "  q=quit  r=reset  e=enable  d=disable  m=mask  s=save  ?=help  "
            "j/k=scroll  h/l=blocks  [/]=size  +/-=interval"
        )
        try:
            stdscr.addstr(h-1, 0, status[:w-1].ljust(w-1),
                          curses.color_pair(9))
        except:
            pass

        # Pruning bar (visual: how much could be pruned)
        prune_row = h - 2
        try:
            bar_w = min(w - 30, 60)
            if len(stats.experts) > 0:
                pct = (dead + rare) / len(stats.experts)
                filled = int(pct * bar_w)
                bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
                prune_label = f" prune potential: [{bar}] {pct*100:.0f}%"
                stdscr.addstr(prune_row, 0, prune_label[:w-1],
                              curses.color_pair(5) if pct > 0.3 else curses.color_pair(3))
        except:
            pass

    def _draw_expert(self, stdscr, row, col, layer, expert, v,
                     max_freq, max_gate, flash, shared):
        is_masked = self.show_mask and (layer, expert) in self.mask
        is_flash  = (layer, expert) in flash

        if is_masked:
            ch   = "X"
            attr = curses.color_pair(6) | curses.A_DIM
        else:
            ch   = freq_char(v.frequency, max_freq)
            g    = _log_normalize(v.avg_gate, max_gate)
            attr = gate_color_pair(g)
            if is_flash:
                attr = curses.color_pair(8) | curses.A_BOLD
            if shared:
                attr |= curses.A_UNDERLINE

        try:
            stdscr.addstr(row, col, ch, attr)
        except:
            pass

    def _draw_help(self, stdscr, h, w):
        lines = [
            "  vllm-riy live -- Help",
            "",
            "  q         quit",
            "  r         reset stats (POST /riy/stats/reset)",
            "  e / d     enable / disable stats collection",
            "  s         save current stats to riy_stats_export.json",
            "",
            "  h / l     scroll expert blocks left / right",
            "  j / k     scroll layers up / down",
            "  [ / ]     decrease / increase block size",
            "  + / -     increase / decrease refresh interval",
            "",
            "  m         toggle mask overlay (masked experts shown as X)",
            "  ?         toggle this help",
            "",
            "  Display:",
            "  \u00b7 = never    \u2591 = rare    \u2592 = medium    \u2593 = high    \u2588 = dominant",
            "  Color = avg gate magnitude, log scale (dark->cyan->green->orange->red)",
            "  Flash (yellow) = expert activation spiked since last poll",
            "  Underline = shared expert",
            "",
            "  Press ? to close",
        ]
        for i, line in enumerate(lines):
            if i >= h - 1:
                break
            try:
                attr = curses.color_pair(7)
                if i == 0:
                    attr |= curses.A_BOLD
                stdscr.addstr(i, 0, line[:w-1], attr)
            except:
                pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="riy live -- Live MoE expert activation dashboard"
    )
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=8019)
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Refresh interval in seconds (default: 2.0)")
    parser.add_argument("--block",    type=int, default=0,
                        help="Experts per column block (0=auto-fit to terminal)")
    parser.add_argument("--vllm-port", type=int, default=8011,
                        help="vLLM API port for model name (default: 8011)")
    parser.add_argument("--demo",     action="store_true",
                        help="Run with synthetic animated data (no vLLM needed)")
    args = parser.parse_args()

    dash = Dashboard(
        host=args.host,
        port=args.port,
        interval=args.interval,
        block_size=args.block,
        demo=args.demo,
        vllm_port=args.vllm_port,
    )

    curses.wrapper(dash.run)
    print("riy live -- bye.")


if __name__ == "__main__":
    main()
