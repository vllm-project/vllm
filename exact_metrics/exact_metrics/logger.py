import json
import logging
import os
import time

from vllm.v1.metrics.loggers import StatLoggerBase

log = logging.getLogger(__name__)
OUT_DIR = os.environ.get("EXACT_METRICS_DIR", "/metrics")


def _sanitize(obj):
    if isinstance(obj, dict):
        return {
            (
                k if isinstance(k, (str, int, float, bool, type(None))) else str(k)
            ): _sanitize(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _sanitize(v) for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


class ExactMetricsLogger(StatLoggerBase):
    def __init__(self, vllm_config, engine_index: int = 0):
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"engine{engine_index}.jsonl")
        self.f = open(path, "a", buffering=1)  # noqa: SIM115

    def _write(self, **payload):
        try:
            record = {"ts": time.time(), "engine": self.engine_index, **payload}
            self.f.write(json.dumps(_sanitize(record)) + "\n")
        except Exception:
            log.exception("ExactMetricsLogger failed")

    def record(
        self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx: int = 0
    ):
        if not (iteration_stats or scheduler_stats or mm_cache_stats):
            return
        self._write(
            engine=engine_idx,
            iteration=iteration_stats,
            scheduler=scheduler_stats,
            mm_cache=mm_cache_stats,
        )

    def log_engine_initialized(self):
        cc = getattr(self.vllm_config, "cache_config", None)
        info = cc.metrics_info() if hasattr(cc, "metrics_info") else cc  # type: ignore
        self._write(event="engine_initialized", cache_config=info)

    def record_sleep_state(self, is_awake: int, level: int):
        self._write(event="sleep_state", is_awake=is_awake, level=level)
