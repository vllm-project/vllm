# SPDX-License-Identifier: Apache-2.0

import time
from typing import Callable, Optional, Union

import msgspec
import torch

from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler)
from vllm.platforms import current_platform
from vllm.utils import is_pin_memory_available


class SpecDecodeWorkerMetrics(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
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
    """Class which copies rejection/typical-acceptance sampler metrics
    from the device to CPU on a non-default Torch stream.
    """

    def __init__(self,
                 spec_decode_sampler: SpecDecodeBaseSampler,
                 timer: Optional[Timer] = None,
                 collect_interval_s: float = 5.0):
        self.spec_decode_sampler = spec_decode_sampler
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

    def init_tensors(self,
                     rank: int,
                     device_type: Union[torch.device, str] = 'cuda') -> None:
        self._rank = rank
        if isinstance(device_type, torch.device):
            device_type = device_type.type
        stream = current_platform.Stream
        if stream is not None:
            self._copy_stream = stream()

    def maybe_collect_rejsample_metrics(
            self, k: int) -> Optional[SpecDecodeWorkerMetrics]:
        # Skip for any platform that doesn't have device Event
        if current_platform.Event is None:
            return None

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
        """Return whether or not this iteration should print sampling
        metrics.
        """
        if self._rank != 0:
            return False

        return now - self._last_metrics_collect_time >= self._rejsample_metrics_collect_interval_s  # noqa: E501

    def _copy_rejsample_metrics_async(self) -> torch.cuda.Event:
        """Copy rejection/typical-acceptance sampling metrics
        (number of accepted tokens, etc) to CPU asynchronously.

        Returns a device event recording when the copy is complete.
        """
        assert self._copy_stream is not None
        self._copy_stream.wait_stream(current_platform.current_stream())

        with current_platform.stream(self._copy_stream):
            self._aggregate_num_accepted_tokens.copy_(
                self.spec_decode_sampler.num_accepted_tokens,
                non_blocking=True)
            self._aggregate_num_emitted_tokens.copy_(
                self.spec_decode_sampler.num_emitted_tokens, non_blocking=True)
            # Number of draft tokens is calculated on CPU, so no copy is
            # required.
            self._aggregate_num_draft_tokens = (
                self.spec_decode_sampler.num_draft_tokens)

        aggregate_metrics_ready = current_platform.Event()
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

        # update time of last collection
        self._last_metrics_collect_time = self._timer()

        accepted_tokens = self._aggregate_num_accepted_tokens.item()
        emitted_tokens = self._aggregate_num_emitted_tokens.item()
        draft_tokens = self._aggregate_num_draft_tokens

        max_num_emitted_tokens = self.get_max_num_emitted_tokens(
            draft_tokens, k)

        if draft_tokens > 0:
            draft_acceptance_rate = accepted_tokens / draft_tokens
        else:
            draft_acceptance_rate = float("nan")

        if max_num_emitted_tokens > 0:
            system_efficiency = emitted_tokens / max_num_emitted_tokens
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
    def get_max_num_emitted_tokens(draft_tokens: int, k: int) -> int:
        """Calculate the number of emitted tokens, assuming all tokens are
        accepted.

        This is equal to the number of sequences that have been speculated on,
        times (speculation len + 1). The +1 comes from the bonus token.
        """
        # Determine the number of sequences that have been speculated on. Since
        # the batch size can be variable, we divide by k.
        assert draft_tokens % k == 0
        total_num_spec_seqs = draft_tokens // k

        # A single sequence may emit k accepted tokens and one bonus token in
        # the best case.
        num_emitted_per_seq_if_all_accepted = k + 1

        # The max num of emitted tokens is the number of speculated sequences
        # times the max emitted per seq.
        return total_num_spec_seqs * num_emitted_per_seq_if_all_accepted
