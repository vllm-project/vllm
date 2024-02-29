from unittest.mock import MagicMock
import torch

from vllm.worker.spec_decode.metrics import DraftTargetWorkerMetrics, AsyncMetricsCollector

def test_initial_call_returns_none():
    rej_sampler = MagicMock()
    rej_sampler.num_accepted_tokens = torch.tensor(0, dtype=torch.long, device='cuda')
    rej_sampler.num_emitted_tokens = torch.tensor(0, dtype=torch.long, device='cuda')
    rej_sampler.num_draft_tokens = 0

    collector = AsyncMetricsCollector(rej_sampler)
    collector.init_gpu_tensors(rank=0)
    maybe_metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert maybe_metrics is None

def test_second_call_returns_metrics():
    rej_sampler = MagicMock()
    rej_sampler.num_accepted_tokens = torch.tensor(0, dtype=torch.long, device='cuda')
    rej_sampler.num_emitted_tokens = torch.tensor(0, dtype=torch.long, device='cuda')
    rej_sampler.num_draft_tokens = 0

    collector = AsyncMetricsCollector(rej_sampler)
    collector.init_gpu_tensors(rank=0)
    _ = collector.maybe_collect_rejsample_metrics(k=5)
    metrics = collector.maybe_collect_rejsample_metrics(k=5)
    assert metrics is not None

def test_initial_metrics_correct_values():
    num_accepted_tokens = 103
    num_emitted_tokens = 104
    num_draft_tokens = 105
    k = 5

    rej_sampler = MagicMock()
    rej_sampler.num_accepted_tokens = torch.tensor(num_accepted_tokens, dtype=torch.long, device='cuda')
    rej_sampler.num_emitted_tokens = torch.tensor(num_emitted_tokens, dtype=torch.long, device='cuda')
    rej_sampler.num_draft_tokens = num_draft_tokens

    collector = AsyncMetricsCollector(rej_sampler)
    collector.init_gpu_tensors(rank=0)
    _ = collector.maybe_collect_rejsample_metrics(k)
    metrics = collector.maybe_collect_rejsample_metrics(k)

    assert metrics.num_spec_tokens == k
    assert metrics.accepted_tokens == num_accepted_tokens
    assert metrics.draft_tokens == num_draft_tokens
    assert metrics.emitted_tokens == num_emitted_tokens
