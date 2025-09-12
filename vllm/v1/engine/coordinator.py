# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import multiprocessing
import time
import weakref
from typing import Optional

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import get_mp_context, make_zmq_socket, set_process_title
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
      collectively move from a running state to a paused state. This transition
      is synchronized via the all-reduce operation performed in the
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

    Note that when deployed in External LB mode, no stats will be published by
    the engines and thus updates will only be sent to front-ends when the
    request wave / running state changes.
    """

    def __init__(self, parallel_config: ParallelConfig):

        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip
        external_lb = parallel_config.data_parallel_external_lb
        hybrid_lb = parallel_config.data_parallel_hybrid_lb

        # Assume coordinator is colocated with front-end procs when not in
        # either external or hybrid DP LB mode.
        local_only = not (external_lb or hybrid_lb)
        front_publish_address = get_engine_client_zmq_addr(
            local_only=local_only, host=host)

        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        back_publish_address = get_engine_client_zmq_addr(local_only_eng, host)
        back_output_address = get_engine_client_zmq_addr(local_only_eng, host)

        context = get_mp_context()
        self.proc: multiprocessing.Process = context.Process(
            target=DPCoordinatorProc.run_coordinator,
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


class DPCoordinatorProc:

    def __init__(self,
                 engine_count: int,
                 min_stats_update_interval_ms: int = 100):
        set_process_title("DPCoordinator")
        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.stats_update_interval_ms = min_stats_update_interval_ms

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        min_stats_update_interval_ms: int = 100,
    ):
        coordinator = DPCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms)
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

        # For tracking request wave progression.
        current_wave = 0
        engines_running = False

        # For tracking request counts for internal load-balancing.
        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: Optional[list[list[int]]] = None

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

            # Wait until all engines subscribe.
            for _ in self.engines:
                if publish_back.recv() != b'\x01':
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe")
                    return
            # Send ready message to engines.
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 5 seconds.
                wait_for = (self.stats_update_interval_ms
                            if stats_changed else 5000)

                # Wait at least 50ms to ensure we've received all stats for
                # the current step.
                min_timeout = 50 if last_step_counts is None else 0

                events = poller.poll(timeout=max(min_timeout, wait_for -
                                                 elapsed))
                if not events:
                    # Poller timeout - publish current stats to front-ends.
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False

                    to_publish = (engine_req_counts_list, current_wave,
                                  engines_running)
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b'\x01', b'\x00'):
                        # Ignore subscription messages.
                        continue

                    decoded = msgspec.msgpack.decode(buffer)
                    if isinstance(decoded, (list, tuple)) and len(
                            decoded) == 2 and decoded[0] == "SCALE_ELASTIC_EP":
                        # Handle scale up notification
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            for _ in range(new_engine_count - current_count):
                                self.engines.append(EngineState())
                            # NOTE(yongji): handle the case
                            # where newly started engines have current_wave = 0
                            # if existing engines just finished a wave
                            # and engine_running isn't updated yet at
                            # CoordinatorProc requests routed to newly started
                            # engines may not wake up existing engines, as long
                            # as 0 < request.wave < existing engines'
                            # current_wave
                            # we note that 0 is the wave number for the new
                            # engine
                            engines_running = False
                            logger.info(
                                "DPCoordinator scaled up from %s to %s "
                                "engines", current_count, new_engine_count)
                        else:
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "DPCoordinator scaled down from %s to %s "
                                "engines", current_count, new_engine_count)
                        continue  # Skip normal engine notification processing

                    # We received a message on the front-end XPUB socket,
                    # from an API server sending a new request while the
                    # engines are paused, so that we can wake the other
                    # engines.
                    engine_to_exclude, wave = decoded
                    if not engines_running:
                        if wave < current_wave:
                            # If the wave number is stale, ensure the message
                            # is handled by all the engines.
                            engine_to_exclude = None

                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(publish_back, current_wave,
                                              engine_to_exclude)

                if output_back in events:
                    # We received a message from one of the engines.

                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    scheduler_stats = outputs.scheduler_stats
                    if scheduler_stats:
                        # 1. Updated request load stats - update our local
                        # state with these.
                        stats = self.engines[eng_index].request_counts
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        if (stats_wave > last_stats_wave
                                or stats_wave == last_stats_wave
                                and stats_step > last_stats_step):
                            if stats_changed:
                                last_step_counts = self._get_engine_counts(
                                    do_copy=True)
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                                stats_step != last_stats_step):
                            logger.warning(
                                "Received stats for out-of-order "
                                "step (%d, %d) from engine %d (expected "
                                "> (%d, %d))", stats_wave, stats_step,
                                eng_index, last_stats_wave, last_stats_step)
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats_changed = True

                    if (wave := outputs.wave_complete) is not None:
                        # 2. Notification from rank 0 engine that we've
                        # moved into the global paused state
                        # (engines_running==False).
                        if current_wave <= wave:
                            new_wave = wave + 1
                            logger.debug("Moving DP wave from %d to %d.",
                                         current_wave, new_wave)
                            current_wave = new_wave
                            engines_running = False
                            wave_state_changed = True
                    elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave or
                        (wave == current_wave and not engines_running)):
                        # 3. The engine received request for a non-current wave
                        # so we must ensure that other engines progress to the
                        # next wave (race condition handling).
                        logger.debug(
                            "Starting wave %d after notification of "
                            "stale wave request from engine.", wave)
                        current_wave = wave
                        engines_running = True
                        wave_state_changed = True
                        self._send_start_wave(publish_back, wave, eng_index)

                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))

    @staticmethod
    def _send_start_wave(socket: zmq.Socket, wave: int,
                         exclude_engine_index: Optional[int]):
        """Broadcast the START_DP_WAVE message to all the engines.
        It includes the current wave number and index of engine which
        has already received a request with this wave number and so doesn't
        require additional notification.
        """
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart(
            (EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    def _get_engine_counts(self, do_copy=False) -> list[list[int]]:
        """Return list of [waiting, running] count lists for each engine."""
        if do_copy:
            return [copy.copy(e.request_counts) for e in self.engines]
        return [e.request_counts for e in self.engines]
