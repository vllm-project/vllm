# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Transfer timing for the filesystem-tier diagnostic path."""

from time import monotonic

from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker
from vllm.distributed.ec_transfer.ec_connector.tiering.metrics import ECTieringStats


class ECTieringWorker(ECCPUWorker):
    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self._timing_stats = ECTieringStats()

    def start_load_caches(self, encoder_cache, connector_metadata):
        if not connector_metadata.loads:
            return super().start_load_caches(encoder_cache, connector_metadata)
        started = monotonic()
        super().start_load_caches(encoder_cache, connector_metadata)
        self._timing_stats.observe("h2d_host_submit_seconds", monotonic() - started)

    def flush_saves(self):
        if self._save_count == 0:
            return super().flush_saves()
        started = monotonic()
        super().flush_saves()
        self._timing_stats.observe("d2h_host_submit_seconds", monotonic() - started)

    def _collect_finished(self, inflight, direction):
        # The base class returns only transfers whose event has already fired.
        before = tuple(inflight)
        completions = super()._collect_finished(inflight, direction)
        metric = "h2d_device_seconds" if direction == "load" else "d2h_device_seconds"
        for transfer in before[: len(before) - len(inflight)]:
            self._timing_stats.observe(
                metric, transfer.start_event.elapsed_time(transfer.end_event) / 1000
            )
        return completions

    def get_stats(self):
        if self._timing_stats.is_empty():
            return None
        return self._timing_stats.clone_and_reset()
