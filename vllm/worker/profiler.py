###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import json
import os
import queue
import threading
import time
from contextlib import contextmanager

from vllm.logger import init_logger
from vllm.utils import get_vllm_instance_id
from typing import List, Any

logger = init_logger(__name__)


class FileWriter(threading.Thread):

    def __init__(self, filename, event_queue):
        super().__init__()
        self.filename = filename
        self.event_queue = event_queue
        self.daemon = True
        self.timer_event = threading.Event()

    def _drain_event_queue(self):
        content = ''
        while True:
            try:
                element = self.event_queue.get_nowait()
                content += element
            except queue.Empty:
                break
        return content

    def run(self):
        # don't check the queue too often
        while not self.timer_event.wait(1):
            # Block and wait for the next item in the queue
            content = self.event_queue.get()
            # Collect any other items in the queue
            content += self._drain_event_queue()

            with open(self.filename, 'a') as outfile:
                outfile.write(content)


class Profiler:
    profiling_trace_events: queue.Queue = queue.Queue()
    event_tid = {'counter': 1, 'external': 2, 'internal': 3}
    vllm_instance_id = get_vllm_instance_id()
    filename = f'server_events_{vllm_instance_id}.json'
    event_cache: List[Any] = []

    def __init__(self):
        self.enabled = os.getenv('VLLM_PROFILER_ENABLED',
                                 'false').lower() == 'true' and int(
                                     os.getenv('RANK', '0')) == 0
        msg = f'Profiler enabled for: {self.vllm_instance_id}'
        logger.info(msg)
        if self.enabled:
            # initialize the trace file (JSON Array Format)
            with open(self.filename, 'w') as outfile:
                outfile.write('[')
            file_writer = FileWriter(self.filename,
                                     self.profiling_trace_events)
            file_writer.start()

    def _dump_with_sep(self, entry):
        entry = json.dumps(entry) + ','
        self.profiling_trace_events.put(entry)

    def get_timestamp_us(self):
        return time.time() * 1000000.0

    def record_counter(self, ts, counter):
        if self.enabled:
            self._dump_with_sep({
                'pid': 1,
                'tid': self.event_tid['counter'],
                'ph': 'C',
                'name': 'utils',
                'ts': ts,
                'args': counter
            })

    def start(self, type, name, args=None):
        if self.enabled:
            ts = self.get_timestamp_us()
            if args is not None and 'counter' in args:
                self.record_counter(ts, args['counter'])
                del args['counter']
            event = {
                'pid': 1,
                'tid': self.event_tid[type],
                'ph': 'X',
                'name': name,
                'ts': ts,
                'dur': None,
                'args': args
            }
            self.event_cache.append(event)

    def end(self):
        if self.enabled:
            ts = self.get_timestamp_us()
            if not self.event_cache:
                logger.warning(
                    'Profiler: end() call does not have matching start() call. '
                    'Disabling profiler.')
                self.enabled = False
                return
            event = self.event_cache.pop()
            event['dur'] = ts - event['ts']
            self._dump_with_sep(event)

    @contextmanager
    def record_event(self, type, name, args=None):
        if self.enabled:
            self.start(type, name, args)
            yield
            self.end()
        else:
            yield
