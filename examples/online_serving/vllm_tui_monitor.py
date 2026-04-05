#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM TUI Monitor
================

A retro-futuristic terminal dashboard for monitoring vLLM instances.
Visualizes KV cache usage, throughput, and request load in real-time.

Usage:
    python vllm_tui_monitor.py --url http://localhost:8000/metrics
    python vllm_tui_monitor.py --mock
"""

import argparse
import asyncio
import contextlib
import math
import random
import re
from collections import deque
from datetime import datetime

import requests
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid
from textual.reactive import reactive
from textual.widgets import Digits, Footer, Label, RichLog, Sparkline, Static

# --- Constants & Configuration ---

REFRESH_RATE = 1.0  # Seconds between metric polls
HISTORY_SIZE = 60  # Number of data points for sparklines

# Theme Colors
COLOR_PRIMARY = "#00ff00"  # Phosphor Green
COLOR_WARNING = "#ffff00"  # Amber
COLOR_DANGER = "#ff0000"  # Red
COLOR_BG = "#000000"  # Deep Black

# ASCII Art Header
ASCII_LOGO = """
      ___       ___           ___           ___     
     /\__\     /\__\         /\__\         /\__\    
    /:/  /    /:/  /        /:/  /        /::|  |   
   /:/  /    /:/  /        /:/  /        /:|:|  |   
  /:/__/    /:/  /        /:/  /        /:/|:|__|__ 
  |:|  |    |:|__|__     /:/__/        /:/ |::::\__\\
  |:|  |    |:|  |  |   /::\  \        \/__/~~/:/  /
  |:|  |    |:|  |  |  /:/\:\  \             /:/  / 
  |:|__|    |:|__|__|  \/__\:\  \           /:/  /  
   \____\    \____\         \:\__\         /:/  /   
                             \/__/         \/__/    
"""


# --- Metric Parsing & Fetching ---


class MetricPoller:
    """Handles fetching and parsing metrics from vLLM or generating mock data."""

    def __init__(self, url: str, mock: bool = False):
        self.url = url
        self.mock = mock
        self._mock_time = 0.0

        # Metric storage
        self.metrics = {
            "gpu_cache_usage_perc": 0.0,
            "num_requests_running": 0,
            "num_requests_waiting": 0,
            "num_requests_swapped": 0,
            "avg_generation_throughput_toks_per_s": 0.0,
            "avg_prompt_throughput_toks_per_s": 0.0,
        }

    def fetch(self) -> dict[str, float]:
        if self.mock:
            return self._generate_mock_metrics()

        try:
            response = requests.get(self.url, timeout=2)
            response.raise_for_status()
            return self._parse_prometheus(response.text)
        except requests.exceptions.RequestException as e:
            # Re-raise as is to be handled by the caller
            raise e
        except Exception as e:
            # Wrap other errors
            raise RuntimeError(f"Failed to fetch metrics: {e}") from e

    def _parse_prometheus(self, text: str) -> dict[str, float]:
        """Simple regex-based Prometheus parser for specific vLLM metrics."""
        parsed = {}

        # Regex to match: name{labels} value
        # Handles scientific notation (e.g. 1.23e-05), integers, and floats
        number_pattern = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"

        # Patterns broken down for line length
        p_cache = rf"vllm:gpu_cache_usage_perc{{?[^}}]*}}?\s+{number_pattern}"
        p_run = rf"vllm:num_requests_running{{?[^}}]*}}?\s+{number_pattern}"
        p_wait = rf"vllm:num_requests_waiting{{?[^}}]*}}?\s+{number_pattern}"
        p_swap = rf"vllm:num_requests_swapped{{?[^}}]*}}?\s+{number_pattern}"
        p_gen = (
            rf"vllm:avg_generation_throughput_toks_per_s{{?[^}}]*}}?\s+{number_pattern}"
        )
        p_prompt = (
            rf"vllm:avg_prompt_throughput_toks_per_s{{?[^}}]*}}?\s+{number_pattern}"
        )

        patterns = {
            "gpu_cache_usage_perc": p_cache,
            "num_requests_running": p_run,
            "num_requests_waiting": p_wait,
            "num_requests_swapped": p_swap,
            "avg_generation_throughput_toks_per_s": p_gen,
            "avg_prompt_throughput_toks_per_s": p_prompt,
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                try:
                    parsed[key] = float(match.group(1))
                except ValueError:
                    parsed[key] = 0.0
            else:
                # Keep previous value if missing or default to 0 if never found
                parsed[key] = self.metrics.get(key, 0.0)

        self.metrics.update(parsed)
        return self.metrics

    def _generate_mock_metrics(self) -> dict[str, float]:
        self._mock_time += 0.1

        # Simulate load with sine waves and noise
        load_factor = (math.sin(self._mock_time) + 1) / 2  # 0 to 1

        self.metrics["gpu_cache_usage_perc"] = max(
            0.0, min(1.0, load_factor * 0.8 + random.uniform(-0.05, 0.05))
        )
        self.metrics["num_requests_running"] = int(
            load_factor * 50 + random.randint(0, 5)
        )
        self.metrics["num_requests_waiting"] = int(max(0, load_factor * 20 - 10))
        self.metrics["avg_generation_throughput_toks_per_s"] = (
            load_factor * 2000 + random.uniform(0, 100)
        )

        return self.metrics


# --- UI Widgets ---


class RetroHeader(Static):
    """Displays the ASCII logo and connection status."""

    status = reactive("CONNECTING")

    def compose(self) -> ComposeResult:
        yield Label(ASCII_LOGO, id="logo")
        yield Label(f"STATUS: {self.status}", id="status_label")

    def watch_status(self, status: str) -> None:
        """Update the status label when the reactive status changes."""
        with contextlib.suppress(Exception):
            self.query_one("#status_label").update(f"STATUS: {status}")

    def update_status(self, status: str, color: str = "green"):
        self.status = status
        with contextlib.suppress(Exception):
            self.query_one("#status_label").styles.color = color


class ReactorCore(Static):
    """Visualizes GPU Cache Usage as a grid of blocks."""

    usage = reactive(0.0)

    def watch_usage(self, usage: float) -> None:
        self.update(self._render_core(usage))

    def _render_core(self, usage: float) -> str:
        # Create a 10x20 grid (approx)
        rows = 10
        cols = 20
        total_cells = rows * cols
        filled_cells = int(usage * total_cells)

        # Block characters
        FULL_BLOCK = "█"
        EMPTY_BLOCK = "·"

        out = []
        for i in range(total_cells):
            if i < filled_cells:
                out.append(f"[{self._get_color(i, total_cells)}]{FULL_BLOCK}[/]")
            else:
                out.append(f"[#333333]{EMPTY_BLOCK}[/]")

            if (i + 1) % cols == 0:
                out.append("\n")

        return "".join(out)

    def _get_color(self, index: int, total: int) -> str:
        ratio = index / total
        if ratio < 0.6:
            return COLOR_PRIMARY
        elif ratio < 0.85:
            return COLOR_WARNING
        else:
            return COLOR_DANGER


class MetricSparkline(Static):
    """A labelled sparkline graph."""

    def __init__(self, title: str, color: str = "green", **kwargs):
        super().__init__(**kwargs)
        self.title_text = title
        self.spark_color = color
        self.data = deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)

    def compose(self) -> ComposeResult:
        yield Label(self.title_text, classes="spark_title")
        yield Sparkline(self.data, summary_function=max, color=self.spark_color)
        yield Label("0.0", classes="spark_value")

    def add_data(self, value: float):
        self.data.append(value)
        self.query_one(Sparkline).data = self.data
        self.query_one(".spark_value").update(f"{value:.1f}")


class LogStream(RichLog):
    """Scrolling log of events."""

    pass


# --- Main Application ---


class VLLMTUIApp(App):
    """The main TUI Application."""

    CSS = """
    Screen {
        background: #000000;
        color: #00ff00;
        font-family: monospace;
    }

    #logo {
        color: #00ff00;
        text-align: center;
        width: 100%;
    }

    #status_label {
        text-align: center;
        width: 100%;
        background: #111111;
        padding: 1;
        text-style: bold;
    }

    RetroHeader {
        height: auto;
        dock: top;
        margin-bottom: 1;
    }

    Grid {
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr;
    }

    Container {
        border: solid #00ff00;
        padding: 1;
        margin: 1;
    }

    .box_title {
        background: #00ff00;
        color: #000000;
        padding: 0 1;
        margin-bottom: 1;
        text-style: bold;
    }

    ReactorCore {
        height: 100%;
        content-align: center middle;
    }

    MetricSparkline {
        height: auto;
        margin-bottom: 1;
        border-bottom: solid #333333;
        padding-bottom: 1;
    }

    .spark_title {
        color: #888888;
    }

    .spark_value {
        text-align: right;
        color: #ffffff;
        text-style: bold;
    }
    
    LogStream {
        height: 10fr;
        border-top: solid #00ff00;
        background: #050505;
        color: #aaaaaa;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.poller = MetricPoller(args.url, args.mock)

    def compose(self) -> ComposeResult:
        yield RetroHeader()

        with Grid():
            # Left Column: Reactor Core (KV Cache)
            with Container(classes="panel"):
                yield Label(" REACTOR CORE (KV CACHE) ", classes="box_title")
                yield ReactorCore(id="reactor")
                yield Label("\nSYSTEM LOAD", classes="box_title")
                yield Digits(id="load_digits")

            # Right Column: Metrics & Logs
            with Container(classes="panel"):
                yield Label(" TELEMETRY ", classes="box_title")
                yield MetricSparkline(
                    "Throughput (tok/s)", color="green", id="spark_throughput"
                )
                yield MetricSparkline(
                    "Active Requests", color="yellow", id="spark_active"
                )
                yield MetricSparkline(
                    "Waiting Requests", color="red", id="spark_waiting"
                )

                yield Label(" EVENT LOG ", classes="box_title")
                yield LogStream(highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        self.title = "vLLM OPS TERMINAL"
        self.set_interval(REFRESH_RATE, self.update_metrics)
        self.query_one(LogStream).write(
            f"[bold green]SYSTEM ONLINE[/] - Connecting to {self.args.url}..."
        )
        if self.args.mock:
            self.query_one(LogStream).write(
                "[bold yellow]WARNING: MOCK MODE ENGAGED[/]"
            )

    async def update_metrics(self) -> None:
        """Fetch metrics in a background thread and update UI."""
        try:
            # Run the blocking fetch in a separate thread to avoid freezing the UI
            metrics = await asyncio.to_thread(self.poller.fetch)

            # Update Header
            self.query_one(RetroHeader).update_status("CONNECTED", "green")

            # Update Reactor Core
            self.query_one(ReactorCore).usage = metrics.get("gpu_cache_usage_perc", 0.0)

            # Update Load Digits (just showing running requests as a big number)
            self.query_one("#load_digits", Digits).update(
                f"{int(metrics.get('num_requests_running', 0)):03d}"
            )

            # Update Sparklines using IDs
            self.query_one("#spark_throughput", MetricSparkline).add_data(
                metrics.get("avg_generation_throughput_toks_per_s", 0.0)
            )
            self.query_one("#spark_active", MetricSparkline).add_data(
                metrics.get("num_requests_running", 0.0)
            )
            self.query_one("#spark_waiting", MetricSparkline).add_data(
                metrics.get("num_requests_waiting", 0.0)
            )

            # Log significant events
            log = self.query_one(LogStream)
            if metrics.get("num_requests_waiting", 0) > 5:
                log.write(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "[bold red]ALERT:[/] High queue depth detected!"
                )

        except requests.exceptions.RequestException as e:
            self.query_one(RetroHeader).update_status("CONNECTION LOST", "red")
            self.query_one(LogStream).write(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"[red]Connection Error: {str(e)}[/]"
            )
        except Exception as e:
            self.query_one(RetroHeader).update_status("ERROR", "red")
            self.query_one(LogStream).write(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"[red]System Error: {str(e)}[/]"
            )


def main():
    parser = argparse.ArgumentParser(description="vLLM TUI Monitor")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/metrics",
        help="vLLM Prometheus metrics URL",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Run in mock mode with fake data"
    )
    args = parser.parse_args()

    app = VLLMTUIApp(args)
    app.run()


if __name__ == "__main__":
    main()
