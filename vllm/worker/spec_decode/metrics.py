import torch
from dataclasses import dataclass
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from typing import Optional
from vllm.utils import in_wsl


@dataclass
class DraftTargetWorkerMetrics:
    num_spec_tokens: int
    draft_acceptance_rate: float
    system_efficiency: float
    accepted_tokens: int
    draft_tokens: int
    emitted_tokens: int

    def __repr__(self) -> str:
        return (
            f"DraftTargetWorkerMetrics(num_spec_tokens={self.num_spec_tokens},"
            f"draft_acceptance_rate={self.draft_acceptance_rate:.3f}, "
            f"system_efficiency={self.system_efficiency:.3f}, "
            f"accepted_tokens={self.accepted_tokens}, "
            f"draft_tokens={self.draft_tokens}, "
            f"emitted_tokens={self.emitted_tokens})")


class AsyncMetricsCollector:
    def __init__(self, rejection_sampler: RejectionSampler):
        self.rejection_sampler = rejection_sampler

        self.rank: Optional[int] = None

        # We don't have a device set yet.
        self._copy_stream: Optional[torch.cuda.Stream] = None
        
        self._in_flight_copy: Optional[torch.cuda.Event] = None

        pin_memory = not in_wsl()
        self._aggregate_num_accepted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_emitted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_draft_tokens = 0

        self._rejsample_metrics_collect_interval_s = 5.0
        self._last_metrics_collect_time = 0


    def init_gpu_tensors(self, rank: int) -> None:
        self._rank = rank
        self._copy_stream = torch.cuda.Stream()


    def maybe_collect_rejsample_metrics(self, k: int) -> Optional[DraftTargetWorkerMetrics]:

        # If a copy was initiated in the previous call, collect and return.
        if self._in_flight_copy is not None:
            ready_event = self._in_flight_copy
            self._in_flight_copy = None
            return self._collect_rejsample_metrics(k, ready_event)
        
        # Otherwise, check if we should start a new copy.
        if self._should_collect_rejsample_metrics(time.time()):
            assert self._in_flight_copy is None
            self._in_flight_copy = self._copy_rejsample_metrics_async()
        
        return None
        

    def _should_collect_rejsample_metrics(self, now: float) -> bool:
        """Return whether or not this iteration should print rejection sampling
        metrics.
        """
        if self.rank != 0:
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
                self.rejection_sampler.num_accepted_tokens, non_blocking=True)
            self._aggregate_num_emitted_tokens.copy_(
                self.rejection_sampler.num_emitted_tokens, non_blocking=True)
            # Number of draft tokens is calculated on CPU, so no copy is
            # required.
            self._aggregate_num_draft_tokens = (
                self.rejection_sampler.num_draft_tokens)

        aggregate_metrics_ready = torch.cuda.Event()
        aggregate_metrics_ready.record(self._copy_stream)

        return aggregate_metrics_ready


    def _collect_rejsample_metrics(
            self, k: int,
            ready_event: torch.cuda.Event) -> DraftTargetWorkerMetrics:
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

        # Divide by k since batch size can be variable.
        num_possible_tokens = (draft_tokens / k) * (k + 1)

        if draft_tokens > 0:
            draft_acceptance_rate = accepted_tokens / draft_tokens
        else:
            draft_acceptance_rate = float("nan")

        if num_possible_tokens > 0:
            system_efficiency = emitted_tokens / num_possible_tokens
        else:
            system_efficiency = float("nan")

        return DraftTargetWorkerMetrics(
            num_spec_tokens=k,
            draft_acceptance_rate=draft_acceptance_rate,
            system_efficiency=system_efficiency,
            accepted_tokens=accepted_tokens,
            draft_tokens=draft_tokens,
            emitted_tokens=emitted_tokens,
        )
