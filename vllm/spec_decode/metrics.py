import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.utils import is_pin_memory_available


@dataclass
class SpecDecodeWorkerMetrics:
    """Dataclass holding metrics emitted from the spec decode worker.
    """

    # The empirical acceptance rate of the proposal method on a per-token basis.
    # This is useful for evaluating how well the proposal method aligns with the
    # scoring method.
    draft_acceptance_rate: float

    # The empirical efficiency, measured as the number of tokens emitted by the
    # system divided by the number of tokens that could be emitted by the system
    # if the proposal method were perfect.
    system_efficiency: float

    # The number of speculative tokens produced by the proposal method.
    draft_tokens: int

    # The number of tokens emitted by the entire system.
    emitted_tokens: int

    # The number of tokens accepted by the scoring model and verification
    # routine, e.g. Llama2-70B and lossless rejection sampling.
    #
    # NOTE: Any token accepted by the verification routine is considered
    # accepted (regardless of if the speculative prefix is also accepted). The
    # user will usually see less accepted tokens. This metric is helpful when
    # evaluating alignment of the proposal method with the scoring model.
    accepted_tokens: int

    # The number of speculative tokens per sequence.
    num_spec_tokens: int


Timer = Callable[[], float]


class AsyncMetricsCollector:
    """Class which copies rejection sampler metrics from the device to CPU on a
    non-default Torch stream.
    """

    def __init__(self,
                 rejection_sampler: RejectionSampler,
                 timer: Optional[Timer] = None,
                 collect_interval_s: float = 5.0):
        self._rejection_sampler = rejection_sampler
        self._timer = time.time if timer is None else timer

        self._rank: Optional[int] = None

        # We don't have a device set yet.
        self._copy_stream: Optional[torch.cuda.Stream] = None

        self._in_flight_copy: Optional[torch.cuda.Event] = None

        pin_memory = is_pin_memory_available()
        self._aggregate_num_accepted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_emitted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_draft_tokens = 0

        self._rejsample_metrics_collect_interval_s = collect_interval_s
        self._last_metrics_collect_time = self._timer()

    def init_gpu_tensors(self, rank: int) -> None:
        self._rank = rank
        self._copy_stream = torch.cuda.Stream()

    def maybe_collect_rejsample_metrics(
            self, k: int) -> Optional[SpecDecodeWorkerMetrics]:

        # If a copy was initiated in the previous call, collect and return.
        if self._in_flight_copy is not None:
            ready_event = self._in_flight_copy
            self._in_flight_copy = None
            return self._collect_rejsample_metrics(k, ready_event)

        # Otherwise, check if we should start a new copy.
        if self._should_collect_rejsample_metrics(self._timer()):
            assert self._in_flight_copy is None
            self._in_flight_copy = self._copy_rejsample_metrics_async()

        return None

    def _should_collect_rejsample_metrics(self, now: float) -> bool:
        """Return whether or not this iteration should print rejection sampling
        metrics.
        """
        if self._rank != 0:
            return False

        if (now - self._last_metrics_collect_time <
                self._rejsample_metrics_collect_interval_s):
            return False
        return True

    def _copy_rejsample_metrics_async(self) -> torch.cuda.Event:
        """Copy rejection sampling metrics (number of accepted tokens, etc) to
        CPU asynchronously.

        Returns a CUDA event recording when the copy is complete.
        """
        self._copy_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._copy_stream):
            self._aggregate_num_accepted_tokens.copy_(
                self._rejection_sampler.num_accepted_tokens, non_blocking=True)
            self._aggregate_num_emitted_tokens.copy_(
                self._rejection_sampler.num_emitted_tokens, non_blocking=True)
            # Number of draft tokens is calculated on CPU, so no copy is
            # required.
            self._aggregate_num_draft_tokens = (
                self._rejection_sampler.num_draft_tokens)

        aggregate_metrics_ready = torch.cuda.Event()
        aggregate_metrics_ready.record(self._copy_stream)

        return aggregate_metrics_ready

    def _collect_rejsample_metrics(
            self, k: int,
            ready_event: torch.cuda.Event) -> SpecDecodeWorkerMetrics:
        """Create metrics object from statistics copied asynchronously.

        Args:
            k: int. The number of speculative tokens; used to determine system
                efficiency.
            ready_event: torch.cuda.Event. The CUDA event recording when the
                async GPU->CPU copy is complete.
        """

        ready_event.synchronize()
        accepted_tokens = self._aggregate_num_accepted_tokens.item()
        emitted_tokens = self._aggregate_num_emitted_tokens.item()
        draft_tokens = self._aggregate_num_draft_tokens

        num_possible_tokens = self.get_max_num_accepted_tokens(draft_tokens, k)

        if draft_tokens > 0:
            draft_acceptance_rate = accepted_tokens / draft_tokens
        else:
            draft_acceptance_rate = float("nan")

        if num_possible_tokens > 0:
            system_efficiency = emitted_tokens / num_possible_tokens
        else:
            system_efficiency = float("nan")

        return SpecDecodeWorkerMetrics(
            num_spec_tokens=k,
            draft_acceptance_rate=draft_acceptance_rate,
            system_efficiency=system_efficiency,
            accepted_tokens=accepted_tokens,
            draft_tokens=draft_tokens,
            emitted_tokens=emitted_tokens,
        )

    @staticmethod
    def get_max_num_accepted_tokens(draft_tokens: int, k: int) -> int:
        # Divide by k since batch size can be variable.
        total_num_spec_seqs = draft_tokens / k
        num_accepted_per_seq_if_all_accepted = k + 1
        return int(total_num_spec_seqs / num_accepted_per_seq_if_all_accepted)
