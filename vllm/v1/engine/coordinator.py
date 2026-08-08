# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import multiprocessing
import multiprocessing.connection
import time
import weakref
from dataclasses import dataclass

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import get_mp_context, set_process_title
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)

_PREFILL_ALIGNMENT_OBSERVE = 1
_PREFILL_ALIGNMENT_ACTUAL = 2


@dataclass(frozen=True)
class PrefillAlignmentRelease:
    wave: int
    release_id: int
    target_step: int
    reason: str


class PrefillAlignmentCoordinator:
    """Nonblocking coordinator state for adaptive DP prefill alignment."""

    def __init__(
        self,
        engine_count: int,
        max_delay_passes: int = 30,
        target_step_lead: int = 2,
    ) -> None:
        self.engine_count = engine_count
        self.max_delay_passes = max_delay_passes
        self.target_step_lead = target_step_lead
        self.current_wave = 0
        self.current_release_id = 0
        self.delayed_passes = 0
        self.pending_release: PrefillAlignmentRelease | None = None
        self.pending_acks: set[int] = set()
        self.snapshots: dict[int, dict[int, SchedulerStats]] = {}
        self.last_actual_prefill: dict[int, tuple[int, int]] = {}
        self.skip_first_delay = True
        self.first_completion_logged = False

    def reset_wave(self, wave: int) -> None:
        self.current_wave = wave
        self.current_release_id = 0
        self.delayed_passes = 0
        self.pending_release = None
        self.pending_acks.clear()
        self.snapshots.clear()
        self.last_actual_prefill.clear()
        self.skip_first_delay = True

    def resize(self, engine_count: int) -> None:
        self.engine_count = engine_count
        self.reset_wave(self.current_wave)

    def update(
        self, engine_index: int, stats: SchedulerStats
    ) -> PrefillAlignmentRelease | None:
        if stats.current_wave < self.current_wave:
            return None
        if stats.current_wave > self.current_wave:
            self.reset_wave(stats.current_wave)

        if stats.prefill_alignment_ack_generation >= 0:
            self._acknowledge(engine_index, stats)

        if stats.prefill_alignment_phase == _PREFILL_ALIGNMENT_ACTUAL:
            return None
        if stats.prefill_alignment_phase != _PREFILL_ALIGNMENT_OBSERVE:
            return None

        if self.pending_release is not None:
            if (
                stats.step_counter - self.pending_release.target_step
                < self.max_delay_passes
            ):
                return None
            logger.warning(
                "Prefill alignment release %d timed out waiting for acks; "
                "advancing and broadcasting a fail-open resync.",
                self.pending_release.release_id,
            )
            self._finish_release()
            return self._release(stats.step_counter, "ack_timeout_resync")

        if stats.prefill_alignment_generation != self.current_release_id:
            return None

        step_snapshots = self.snapshots.setdefault(stats.step_counter, {})
        step_snapshots[engine_index] = stats
        if len(step_snapshots) < self.engine_count:
            return None

        self.snapshots = {
            step: values
            for step, values in self.snapshots.items()
            if step > stats.step_counter
        }
        ordered = [step_snapshots[i] for i in range(self.engine_count)]
        num_prefillable = sum(item.prefill_deferred for item in ordered)
        if num_prefillable == 0:
            self.delayed_passes = 0
            return None

        if any(item.prefill_force_allow for item in ordered):
            return self._release(stats.step_counter, "capacity_force_allow")

        if num_prefillable == self.engine_count:
            max_running = max(item.prefill_running_batch for item in ordered)
            max_prefill = max(item.prefill_max_batch for item in ordered)
            max_running_requests = max(
                item.prefill_max_running_requests for item in ordered
            )
            slot_limited = max_running_requests - max_running < max_prefill
            if slot_limited:
                if self.skip_first_delay:
                    self.skip_first_delay = False
                    return self._release(stats.step_counter, "first_delay_skip")
                self.delayed_passes += 1
                if self.delayed_passes >= self.max_delay_passes:
                    return self._release(stats.step_counter, "max_delay_fail_open")
                return None
            return self._release(stats.step_counter, "all_prefillable")

        self.delayed_passes += 1
        if self.delayed_passes >= self.max_delay_passes:
            return self._release(stats.step_counter, "max_delay_fail_open")
        return None

    def _release(self, step: int, reason: str) -> PrefillAlignmentRelease:
        release = PrefillAlignmentRelease(
            wave=self.current_wave,
            release_id=self.current_release_id,
            target_step=step + self.target_step_lead,
            reason=reason,
        )
        self.pending_release = release
        self.pending_acks.clear()
        self.last_actual_prefill.clear()
        self.delayed_passes = 0
        return release

    def _acknowledge(self, engine_index: int, stats: SchedulerStats) -> None:
        release = self.pending_release
        if release is None:
            return
        if (
            stats.prefill_alignment_ack_generation != release.release_id
            or stats.prefill_alignment_ack_target_step != release.target_step
        ):
            return
        self.pending_acks.add(engine_index)
        self.last_actual_prefill[engine_index] = (
            stats.actual_prefill_requests,
            stats.actual_prefill_tokens,
        )
        if stats.prefill_alignment_release_late:
            logger.warning(
                "Prefill alignment late release application: "
                "engine=%d release_id=%d target_step=%d current_step=%d",
                engine_index,
                release.release_id,
                release.target_step,
                stats.step_counter,
            )
        if len(self.pending_acks) == self.engine_count:
            if not self.first_completion_logged:
                logger.info(
                    "Prefill alignment first release completed across all "
                    "%d engines: release_id=%d actual=%s",
                    self.engine_count,
                    release.release_id,
                    self.last_actual_prefill,
                )
                self.first_completion_logged = True
            logger.debug(
                "Prefill alignment release %d completed: %s",
                release.release_id,
                self.last_actual_prefill,
            )
            self._finish_release()

    def _finish_release(self) -> None:
        self.current_release_id += 1
        self.pending_release = None
        self.pending_acks.clear()
        self.snapshots.clear()


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

    def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
        try:
            timeout = 120
            ready = multiprocessing.connection.wait(
                [zmq_addr_pipe, self.proc.sentinel], timeout=timeout
            )
            if not ready:
                raise RuntimeError(
                    "DP Coordinator process failed to report ZMQ addresses "
                    f"within timeout={timeout} seconds during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError(
                    "DP Coordinator process failed during startup."
                ) from None
        finally:
            zmq_addr_pipe.close()

    def __init__(
        self,
        parallel_config: ParallelConfig,
        enable_wave_coordination: bool = True,
        enable_prefill_alignment: bool = False,
        prefill_alignment_max_delay_passes: int = 30,
        prefill_alignment_target_step_lead: int = 2,
    ):
        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip

        # Assume coordinator is colocated with front-end procs when not in
        # either external or hybrid DP LB mode.
        local_only = not parallel_config.local_engines_only
        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        # NOTE(yongji): handling scaling from intra-node to inter-node
        if parallel_config.enable_elastic_ep:
            local_only_eng = False

        front_publish_address = get_engine_client_zmq_addr(local_only, host=host)
        back_publish_address = get_engine_client_zmq_addr(local_only_eng, host=host)
        back_output_address = get_engine_client_zmq_addr(local_only_eng, host=host)

        context = get_mp_context()
        parent_zmq_addr_pipe, child_zmq_addr_pipe = context.Pipe(duplex=False)
        self.proc: multiprocessing.Process = context.Process(
            target=DPCoordinatorProc.run_coordinator,
            name="VLLM_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "zmq_addr_pipe": child_zmq_addr_pipe,
                "enable_wave_coordination": enable_wave_coordination,
                "enable_prefill_alignment": enable_prefill_alignment,
                "prefill_alignment_max_delay_passes": (
                    prefill_alignment_max_delay_passes
                ),
                "prefill_alignment_target_step_lead": (
                    prefill_alignment_target_step_lead
                ),
            },
            daemon=True,
        )
        self.proc.start()
        child_zmq_addr_pipe.close()
        (
            front_publish_address,
            back_output_address,
            back_publish_address,
        ) = self._wait_for_zmq_addrs(parent_zmq_addr_pipe)

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """Returns tuple of ZMQ input address, output address."""
        return self.coord_in_address, self.coord_out_address

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown coordinator process with configurable timeout."""
        if self._finalizer.detach() is not None:
            shutdown([self.proc], timeout=timeout)


class EngineState:
    def __init__(self):
        # [waiting, running, kv_cache_usage]
        self.request_counts: list[int | float] = [0, 0, 0.0]


class DPCoordinatorProc:
    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
        enable_prefill_alignment: bool = False,
        prefill_alignment_max_delay_passes: int = 30,
        prefill_alignment_target_step_lead: int = 2,
    ):
        set_process_title("DPCoordinator")
        self.ctx = zmq.Context()

        self.engines = [EngineState() for _ in range(engine_count)]

        self.stats_update_interval_ms = min_stats_update_interval_ms
        self.enable_wave_coordination = enable_wave_coordination
        self.prefill_alignment = (
            PrefillAlignmentCoordinator(
                engine_count,
                prefill_alignment_max_delay_passes,
                prefill_alignment_target_step_lead,
            )
            if enable_prefill_alignment
            else None
        )

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
        enable_prefill_alignment: bool = False,
        prefill_alignment_max_delay_passes: int = 30,
        prefill_alignment_target_step_lead: int = 2,
    ):
        coordinator = DPCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
            enable_wave_coordination=enable_wave_coordination,
            enable_prefill_alignment=enable_prefill_alignment,
            prefill_alignment_max_delay_passes=(prefill_alignment_max_delay_passes),
            prefill_alignment_target_step_lead=(prefill_alignment_target_step_lead),
        )
        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
                zmq_addr_pipe,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")
        finally:
            if zmq_addr_pipe is not None:
                zmq_addr_pipe.close()

    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
    ):
        decoder = MsgpackDecoder(EngineCoreOutputs)

        # For tracking request wave progression.
        current_wave = 0
        engines_running = False

        # For tracking request counts for internal load-balancing.
        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: list[list[int | float]] | None = None

        with (
            make_zmq_socket(
                path=front_publish_address,  # IPC
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            if zmq_addr_pipe is not None:
                try:
                    zmq_addr_pipe.send(
                        (
                            publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        )
                    )
                finally:
                    zmq_addr_pipe.close()
            # Wait until all engines subscribe.
            for _ in self.engines:
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
            # Send ready message to engines.
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(publish_back, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 5 seconds.
                wait_for = self.stats_update_interval_ms if stats_changed else 5000

                # Wait at least 50ms to ensure we've received all stats for
                # the current step. Only applicable to lockstep (MoE) DP;
                # non-lockstep engines have no synchronized step boundaries.
                if self.enable_wave_coordination and last_step_counts is None:
                    min_timeout = 50
                else:
                    min_timeout = 0

                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
                if not events:
                    # Poller timeout - publish current stats to front-ends.
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False

                    to_publish = (engine_req_counts_list, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                if publish_back in events:
                    buffer = publish_back.recv()
                    if buffer == b"\x01":
                        # NOTE(yongji): newly started engine subscribed
                        # We need to send READY message here instead of receiving
                        # SCALE_ELASTIC_EP notification from engine core client
                        # as SCALE_ELASTIC_EP is only sent when
                        # new engines finished initialization.
                        # Subscription message, on the other hand, is sent
                        # by each engine during initialization
                        publish_back.send(b"READY")
                    elif buffer != b"\x00":
                        logger.error(
                            "DP Coordinator received unexpected message from engines"
                        )

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b"\x01", b"\x00"):
                        # Ignore subscription messages.
                        continue

                    decoded = msgspec.msgpack.decode(buffer)
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 2
                        and decoded[0] == "SCALE_ELASTIC_EP"
                    ):
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
                            logger.info(
                                "DPCoordinator scaled up from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        else:
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "DPCoordinator scaled down from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        if self.prefill_alignment is not None:
                            self.prefill_alignment.resize(new_engine_count)
                        continue  # Skip normal engine notification processing

                    # Wave coordination: handle new-request messages from front-end.
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:
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
                            self._send_start_wave(
                                publish_back, current_wave, engine_to_exclude
                            )

                if output_back in events:
                    # We received a message from one of the engines.

                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index
                    scheduler_stats = outputs.scheduler_stats
                    # Elastic EP stats may arrive while the engine list changes.
                    if scheduler_stats and eng_index >= len(self.engines):
                        continue

                    if scheduler_stats and (
                        scheduler_stats.prefill_alignment_phase
                        != _PREFILL_ALIGNMENT_ACTUAL
                    ):
                        # 1. Updated request load stats - update our local
                        # state with these.
                        stats = self.engines[eng_index].request_counts
                        if self.enable_wave_coordination:
                            # Steps are synchronized across lockstep (MoE) DP
                            # ranks; snapshot counts at step boundaries.
                            stats_step = scheduler_stats.step_counter
                            stats_wave = scheduler_stats.current_wave
                            if (
                                stats_wave > last_stats_wave
                                or stats_wave == last_stats_wave
                                and stats_step > last_stats_step
                            ):
                                if stats_changed:
                                    last_step_counts = self._get_engine_counts(
                                        do_copy=True
                                    )
                                last_stats_step = stats_step
                                last_stats_wave = stats_wave
                            elif stats_wave != last_stats_wave or (
                                stats_step != last_stats_step
                            ):
                                logger.warning(
                                    "Received stats for out-of-order "
                                    "step (%d, %d) from engine %d (expected "
                                    "> (%d, %d))",
                                    stats_wave,
                                    stats_step,
                                    eng_index,
                                    last_stats_wave,
                                    last_stats_step,
                                )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats[2] = scheduler_stats.kv_cache_usage
                        stats_changed = True

                    if scheduler_stats and self.prefill_alignment is not None:
                        release = self.prefill_alignment.update(
                            eng_index, scheduler_stats
                        )
                        if release is not None:
                            self._send_prefill_alignment_release(publish_back, release)

                    # Wave coordination: handle wave completion and start notifications
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:
                        if (wave := outputs.wave_complete) is not None:
                            # 2. Notification from rank 0 engine that we've
                            # moved into the global paused state
                            # (engines_running==False).
                            if current_wave <= wave:
                                new_wave = wave + 1
                                logger.debug(
                                    "Moving DP wave from %d to %d.",
                                    current_wave,
                                    new_wave,
                                )
                                current_wave = new_wave
                                engines_running = False
                                wave_state_changed = True
                                if self.prefill_alignment is not None:
                                    self.prefill_alignment.reset_wave(new_wave)
                        elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave
                            or (wave == current_wave and not engines_running)
                        ):
                            # 3. The engine received request for a non-current wave
                            # so we must ensure that other engines progress to the
                            # next wave (race condition handling).
                            logger.debug(
                                "Starting wave %d after notification of "
                                "stale wave request from engine.",
                                wave,
                            )
                            current_wave = wave
                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(publish_back, wave, eng_index)

                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))

    @staticmethod
    def _send_start_wave(
        socket: zmq.Socket, wave: int, exclude_engine_index: int | None
    ):
        """Broadcast the START_DP_WAVE message to all the engines.
        It includes the current wave number and index of engine which
        has already received a request with this wave number and so doesn't
        require additional notification.
        """
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart((EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))

    @staticmethod
    def _send_prefill_alignment_release(
        socket: zmq.Socket, release: PrefillAlignmentRelease
    ) -> None:
        payload = msgspec.msgpack.encode(
            (
                release.wave,
                release.release_id,
                release.target_step,
                release.reason,
            )
        )
        socket.send_multipart(
            (EngineCoreRequestType.PREFILL_ALIGNMENT_RELEASE.value, payload)
        )

    def _get_engine_counts(self, do_copy=False) -> list[list[int | float]]:
        """Return list of [waiting, running] count lists for each engine."""
        if do_copy:
            return [copy.copy(e.request_counts) for e in self.engines]
        return [e.request_counts for e in self.engines]
