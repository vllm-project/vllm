"""
DecodeBenchConnector worker hook for simulated PD decode instances.

The scheduler side already reports all but the last prompt token as an
external hit; the worker's dummy KV fill is skipped since the KV pool is
not physically backed.
"""

from vllm_simulator.hook import BaseHook


class C_VLLMDecodeBenchConnectorWorkerHook(BaseHook):
    """Hook DecodeBenchConnectorWorker to skip the dummy KV fill."""

    HOOK_CLASS_NAME = "DecodeBenchConnectorWorker"
    HOOK_MODULE_NAME = (
        "vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector"
    )

    @classmethod
    def hook(cls, target):
        def override_start_fill_kv(self, metadata):
            """No-op: the transferred KV is accounting-only."""
            pass

        target.start_fill_kv = override_start_fill_kv
