import dataclasses
import logging
import os
import time
from typing import List


def _get_logger():
    if not __name__ == "__main__":
        from vllm.logger import init_logger
        return init_logger(__name__)
    else:
        lg = logging.getLogger(__name__)
        hd = logging.StreamHandler()
        lg.setLevel(logging.DEBUG)
        hd.setLevel(logging.DEBUG)
        lg.addHandler(hd)
        return lg


logger = _get_logger()


def set_logger(lg):
    global logger
    logger = lg


g_self_rank = 0
g_self_pid = os.getpid()

CALL_LAYERS = 6
SLEEP_TIME = 300  # 10 minutes


class EngineDebugStat:
    def __init__(self):
        self.send_rpc = 0
        self.recv_rpc = 0
        self.execute = 0
        self.sample = 0
        self.dummy = 0
        self.take_draft = 0

    def print(self):
        logger.info(f"[ENGINE-DEBUG-STAT][pid={g_self_pid}] send={self.send_rpc}, recv={self.recv_rpc}, "
                    f"execute={self.execute}, sample={self.sample}, dummy={self.dummy}, take_draft={self.take_draft}")


class WorkerDebugStat:
    def __init__(self):
        self.recv_rpc = 0
        self.resp_rpc = 0
        self.execute = 0
        self.sample = 0
        self.dummy = 0
        self.take_draft = 0
        self.call_step = [0] * CALL_LAYERS
        self.step_count = [0] * CALL_LAYERS
        self.reduce_count = 0
        self.skip_count = 0
        self.skip_reason = 0

    def set_call_step(self, layer, step):
        self.call_step[layer] = step
        self.step_count[layer] += 1

    def set_reduce_step(self, layer, step):
        self.reduce_count += 1
        self.set_call_step(layer, step)

    def set_skip_step(self, layer, step):
        self.skip_count += 1
        self.skip_reason = step
        self.set_call_step(layer, step)

    def print(self):
        logger.info(f"[WORKER-DEBUG-STAT][pid={g_self_pid}] recv={self.recv_rpc}, resp={self.resp_rpc}, "
                    f"execute={self.execute}, sample={self.sample}, dummy={self.dummy}, take_draft={self.take_draft}, "
                    f"call_step={self.call_step}, step_count={self.step_count}, "
                    f"all_reduce={self.reduce_count}, skip={self.skip_count}, skip_reason={self.skip_reason}")


g_engine_debug_stat = EngineDebugStat()
g_worker_debug_stat = WorkerDebugStat()


def get_engine_debug_stat():
    return g_engine_debug_stat


def get_worker_debug_stat():
    return g_worker_debug_stat


def engine_debug_stat_loop():
    while True:
        g_engine_debug_stat.print()
        time.sleep(SLEEP_TIME)


def worker_debug_stat_loop():
    while True:
        g_worker_debug_stat.print()
        time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    g_engine_debug_stat.send_rpc = 10
    g_engine_debug_stat.print()
    g_worker_debug_stat.set_call_step(2, 1001)
    g_worker_debug_stat.print()
