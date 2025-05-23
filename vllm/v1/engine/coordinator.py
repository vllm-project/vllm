# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import time
import weakref
from typing import Optional

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import get_mp_context, get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


class DPCoordinator:
    """Coordinator process used for data-parallel deployments (DP>1).

    Intermediates between multiple DP engine rank processes and one or more
    front-end API server processes.

    * Collects stats from each DP engine (currently just waiting and running
      queue lengths), and publishes these to all front-ends for use in
      load-balancing decisions.

    * Keeps track of the current DP "request wave" number and running state
      of the engines. This is received from the DP rank 0 engine and published
      to the front-end processes along with the current load stats.

      The engines alternate between a global running/paused state. The global
      "request wave" number is a count of the number of times that the workers
      collectively move from running paused state. This transition is
      synchronized via the all-reduce operation performed in the
      DPEngineCoreProc._has_global_unfinished_reqs method.

    * Broadcasts the START_DP_WAVE message to engines to move them from paused
      to running state when one engine receives a new request. This can happen
      in two cases:
      1) A front-end sending a new request while the engines are paused will
         concurrently notify the coordinator.
      2) An engine receiving a request for a stale request wave while in paused
         state will notify the coordinator.

    Engines will move into running state when receiving a new request or
    START_DP_WAVE message.
    """

    def __init__(self, parallel_config: ParallelConfig):

        # Assume coordinator is colocated with front-end procs.
        front_publish_address = get_open_zmq_ipc_path()

        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        local_only = dp_size == parallel_config.data_parallel_size_local
        host = parallel_config.data_parallel_master_ip
        back_publish_address = get_engine_client_zmq_addr(local_only, host)
        back_output_address = get_engine_client_zmq_addr(local_only, host)

        context = get_mp_context()
        self.proc: multiprocessing.Process = context.Process(
            target=CoordinatorProc.run_coordinator,
            name="VLLM_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
            },
            daemon=True)
        self.proc.start()

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """Returns tuple of ZMQ input address, output address."""
        return self.coord_in_address, self.coord_out_address

    def close(self):
        self._finalizer()


class EngineState:

    def __init__(self):
        self.request_counts = [0, 0]  # [waiting, running]


class CoordinatorProc:

    def __init__(self, engine_count: int):

        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.current_wave = 0
        self.engines_running = False
        self.stats_changed = False

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
    ):
        coordinator = CoordinatorProc(engine_count=engine_count)

        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")

    def process_input_socket(self, front_publish_address: str,
                             back_output_address: str,
                             back_publish_address: str):

        decoder = MsgpackDecoder(EngineCoreOutputs)

        with make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
        ) as publish_front, make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
        ) as output_back, make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
        ) as publish_back:

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish
                wait_for = 100 if self.stats_changed else 3000
                events = poller.poll(timeout=max(0, wait_for - elapsed))
                if not events:
                    engine_list = self._get_engine_counts()
                    to_publish = (engine_list, self.current_wave,
                                  self.engines_running)
                    msg = msgspec.msgpack.encode(to_publish)
                    publish_front.send(msg)
                    last_publish = int(time.time() * 1000)
                    self.stats_changed = False
                    continue

                events = dict(events)

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer == b'\x01':
                        # Ignore subscription messages.
                        continue
                    engine_index, wave = msgspec.msgpack.decode(buffer)
                    if wave < self.current_wave:
                        engine_index = None
                    if not self.engines_running:
                        self.engines_running = True
                        self.stats_changed = True
                        self._send_start_wave(publish_back, self.current_wave,
                                              engine_index)

                if output_back in events:
                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    if outputs.scheduler_stats:
                        stats = self.engines[eng_index].request_counts
                        stats[0] = outputs.scheduler_stats.num_waiting_reqs
                        stats[1] = outputs.scheduler_stats.num_running_reqs
                        self.stats_changed = True

                    if outputs.wave_complete is not None:
                        if self.current_wave <= wave:
                            self.current_wave = wave + 1
                            self.engines_running = False
                            self.stats_changed = True
                    elif outputs.start_wave is not None and (
                            wave > self.current_wave or
                        (wave == self.current_wave
                         and not self.engines_running)):
                        # Engine received request for a non-current wave so
                        # we must ensure that other engines progress to the
                        # next wave.
                        self.current_wave = wave
                        self.engines_running = True
                        self.stats_changed = True
                        self._send_start_wave(publish_back, wave, eng_index)

    @staticmethod
    def _send_start_wave(socket: zmq.Socket, wave: int,
                         exclude_engine_index: Optional[int]):
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart(
            (EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    def _get_engine_counts(self) -> list[list[int]]:
        return [e.request_counts for e in self.engines]

    # def _get_engine_list(self) -> Optional[list[int]]:
    #     shortlist: list[int] = []
    #     min_counts = [sys.maxsize, sys.maxsize]
    #     for i, e in enumerate(self.engines):
    #         if e.request_counts <= min_counts:
    #             if e.request_counts < min_counts:
    #                 min_counts = e.request_counts
    #                 shortlist.clear()
    #             shortlist.append(i)
    #     return None if len(shortlist) == len(self.engines) else shortlist
