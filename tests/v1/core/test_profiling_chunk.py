# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for profiling-based dynamic chunk sizing."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.profiling_chunk_predictor import (
    ChunkSizePredictor,
    ProfilingChunkManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

MODEL = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 16
MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 16


def create_requests(num_requests, num_tokens=10, max_tokens=16):
    init_none_hash(sha256)
    sampling_params = SamplingParams(
        ignore_eos=False, max_tokens=max_tokens
    )
    requests = []
    for i in range(num_requests):
        request = Request(
            request_id=f"{i}",
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )
        requests.append(request)
    return requests


def make_output(scheduler):
    req_ids = [req.request_id for req in scheduler.running]
    req_id_to_index = {
        req.request_id: i for i, req in enumerate(scheduler.running)
    }
    sampled_token_ids = [[1000]] * len(scheduler.running)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


# ===================================================================
# ChunkSizePredictor
# ===================================================================


class TestChunkSizePredictor(unittest.TestCase):

    @staticmethod
    def _make_data(a, b, c, seq_lens):
        return [a * l * l + b * l + c for l in seq_lens]

    def test_fit_and_predict(self):
        predictor = ChunkSizePredictor()
        seq_lens = list(range(64, 8256, 128))
        latencies = self._make_data(1e-6, 0.01, 1.0, seq_lens)

        self.assertTrue(predictor.fit(seq_lens, latencies))
        predictor.set_target_latency(8192)
        predictor.is_ready = True

        chunk = predictor.predict(
            num_computed_tokens=0, base_chunk_size=8192, page_size=128
        )
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk % 128, 0)

    def test_predict_decreases_with_history(self):
        predictor = ChunkSizePredictor()
        seq_lens = list(range(64, 8256, 128))
        latencies = self._make_data(1e-6, 0.01, 1.0, seq_lens)
        predictor.fit(seq_lens, latencies)
        predictor.set_target_latency(8192)
        predictor.is_ready = True

        c0 = predictor.predict(0, 8192, 128)
        c1 = predictor.predict(4096, 8192, 128)
        c2 = predictor.predict(16384, 8192, 128)
        self.assertGreaterEqual(c0, c1)
        self.assertGreaterEqual(c1, c2)

    def test_predict_not_ready_returns_none(self):
        predictor = ChunkSizePredictor()
        self.assertIsNone(predictor.predict(0, 8192, 128))

    def test_fit_chunk_and_predict_with_history(self):
        predictor = ChunkSizePredictor()
        predictor.is_ready = True
        predictor.target_latency = 50.0

        data = []
        for i in range(10):
            c, h = 1000 + i * 100, i * 500
            data.append(
                [
                    (c + h) * c,
                    c + h,
                    1,
                    1e-9 * (c + h) * c + 0.001 * (c + h) + 0.5,
                ]
            )
        self.assertTrue(predictor.fit_chunk(data))
        predictor.with_history_ready = True

        result = predictor.predict_with_history(1000, 8192, 128)
        self.assertIsNotNone(result)
        self.assertEqual(result % 128, 0)


# ===================================================================
# ProfilingChunkManager
# ===================================================================


class TestProfilingChunkManager(unittest.TestCase):

    def test_not_ready_before_profiling(self):
        mgr = ProfilingChunkManager(base_chunk_size=8192, page_size=128)
        self.assertFalse(mgr.is_ready)
        self.assertIsNone(mgr.predict_chunk_size(0, 1.0))

    def test_run_profiling_success(self):
        mgr = ProfilingChunkManager(base_chunk_size=8192, page_size=128)
        seq_lens = list(range(64, 8256, 128))
        latencies = [1e-6 * l * l + 0.01 * l + 1.0 for l in seq_lens]
        self.assertTrue(mgr.predictor.fit(seq_lens, latencies))
        mgr.predictor.set_target_latency(8192)
        mgr.predictor.is_ready = True
        mgr._profiling_done = True

        self.assertTrue(mgr.is_ready)
        self.assertIsNotNone(mgr.predict_chunk_size(0, 1.0))

    def test_run_profiling_all_fail(self):
        mgr = ProfilingChunkManager(base_chunk_size=8192, page_size=128)
        too_few_seq_lens = [64, 128, 256]
        too_few_latencies = [1.0, 2.0, 3.0]
        self.assertFalse(
            mgr.predictor.fit(too_few_seq_lens, too_few_latencies)
        )
        self.assertFalse(mgr.is_ready)
        self.assertIsNone(mgr.predict_chunk_size(0, 1.0))

    def test_record_batch_refines_model(self):
        mgr = ProfilingChunkManager(base_chunk_size=8192, page_size=128)
        seq_lens = list(range(64, 8256, 128))
        latencies = [1e-6 * l * l + 0.01 * l + 1.0 for l in seq_lens]
        mgr.predictor.fit(seq_lens, latencies)
        mgr.predictor.set_target_latency(8192)
        mgr.predictor.is_ready = True
        mgr._profiling_done = True

        for i in range(10):
            mgr.record_batch_execution_time(
                [(4096 - i * 100, i * 500)], 0.05 + i * 0.01
            )
        self.assertGreaterEqual(len(mgr.chunked_fit_data), 10)
        self.assertTrue(mgr.history_ready)


# ===================================================================
# Scheduler
# ===================================================================


class TestScheduler(unittest.TestCase):

    @patch("vllm.config.ModelConfig.__post_init__", MagicMock())
    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    def create_scheduler(self):
        from vllm.v1.core.sched.scheduler import Scheduler

        mock_hf_config = MagicMock()
        mock_hf_config.model_type = "qwen3"
        mock_hf_config.is_encoder_decoder = False
        mock_hf_config.architectures = ["Qwen3ForCausalLM"]
        model_config = ModelConfig(
            model=MODEL,
            tokenizer=MODEL,
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            max_model_len=MAX_NUM_BATCHED_TOKENS,
        )
        model_config.hf_config = mock_hf_config
        model_config.hf_text_config = MagicMock()
        model_config.hf_text_config.is_encoder_decoder = False

        scheduler_config = SchedulerConfig(
            max_num_seqs=MAX_NUM_SEQS,
            max_model_len=MAX_NUM_BATCHED_TOKENS,
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            is_encoder_decoder=False,
        )
        scheduler_config.max_num_encoder_input_tokens = 10000
        scheduler_config.encoder_cache_size = 10000
        scheduler_config.chunked_prefill_enabled = True

        cache_config = CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )

        vllm_config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            additional_config={
                "profiling_chunk_config": {
                    "enabled": True,
                    "smooth_factor": 0.8,
                    "min_chunk": 256,
                }
            },
        )
        vllm_config.parallel_config.pipeline_parallel_size = 2
        from unittest.mock import PropertyMock

        type(model_config).is_encoder_decoder = PropertyMock(
            return_value=False
        )
        vllm_config.model_config.hf_config.is_encoder_decoder = False

        kv_cache_config = KVCacheConfig(
            num_blocks=10000,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        )
        kv_cache_config.hash_block_size = BLOCK_SIZE
        cache_config.num_gpu_blocks = 10000

        scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            block_size=BLOCK_SIZE,
            log_stats=True,
            structured_output_manager=MagicMock(
                spec=StructuredOutputManager
            ),
        )

        should_advance = MagicMock()
        should_advance.return_value = False
        scheduler.structured_output_manager.should_advance = should_advance

        return scheduler

    def test_scheduler_init(self):
        scheduler = self.create_scheduler()
        self.assertIsNotNone(scheduler.profiling_chunk_manager)
        self.assertFalse(scheduler._profiling_initialized)

    def test_run_profiling_chunk_init_success(self):
        scheduler = self.create_scheduler()
        mock_executor = MagicMock()
        mock_executor.collective_rpc.return_value = [10.0]

        scheduler.run_profiling_chunk_init(mock_executor)

        self.assertTrue(scheduler._profiling_initialized)
        self.assertTrue(scheduler.profiling_chunk_manager.is_ready)

    def test_run_profiling_chunk_init_skips_second_call(self):
        scheduler = self.create_scheduler()
        mock_executor = MagicMock()
        mock_executor.collective_rpc.return_value = [10.0]

        scheduler.run_profiling_chunk_init(mock_executor)
        call_count = mock_executor.collective_rpc.call_count

        scheduler.run_profiling_chunk_init(mock_executor)
        self.assertEqual(
            mock_executor.collective_rpc.call_count, call_count
        )

    def test_run_profiling_chunk_init_none_executor(self):
        scheduler = self.create_scheduler()
        scheduler.run_profiling_chunk_init(None)
        self.assertTrue(scheduler._profiling_initialized)
        self.assertFalse(scheduler.profiling_chunk_manager.is_ready)

    def test_schedule_new_requests(self):
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=5)
        for req in requests:
            scheduler.add_request(req)

        output = scheduler.schedule()
        self.assertEqual(len(output.scheduled_new_reqs), 5)
        self.assertEqual(len(scheduler.waiting), 0)
        self.assertEqual(len(scheduler.running), 5)

    def test_schedule_with_profiling_ready(self):
        """After profiling is ready, schedule() should still work."""
        scheduler = self.create_scheduler()
        mock_executor = MagicMock()
        mock_executor.collective_rpc.return_value = [10.0]
        scheduler.run_profiling_chunk_init(mock_executor)
        self.assertTrue(scheduler.profiling_chunk_manager.is_ready)

        requests = create_requests(num_requests=3, num_tokens=100)
        for req in requests:
            scheduler.add_request(req)

        output = scheduler.schedule()
        self.assertGreater(len(output.scheduled_new_reqs), 0)
        total = sum(output.num_scheduled_tokens.values())
        self.assertGreater(total, 0)

    def test_schedule_chunked_prefill_running(self):
        """Running requests with num_computed_tokens > 0 get dynamic
        chunk."""
        scheduler = self.create_scheduler()
        mock_executor = MagicMock()
        mock_executor.collective_rpc.return_value = [10.0]
        scheduler.run_profiling_chunk_init(mock_executor)

        requests = create_requests(
            num_requests=1, num_tokens=2000, max_tokens=16
        )
        for req in requests:
            scheduler.add_request(req)

        output1 = scheduler.schedule()
        self.assertEqual(len(output1.scheduled_new_reqs), 1)

        model_output = make_output(scheduler)
        scheduler.update_from_output(output1, model_output)

        output2 = scheduler.schedule()
        self.assertGreater(output2.total_num_scheduled_tokens, 0)

    def test_update_from_output(self):
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=3)
        for req in requests:
            scheduler.add_request(req)

        output = scheduler.schedule()
        model_output = make_output(scheduler)
        scheduler.update_from_output(output, model_output)

        self.assertEqual(len(scheduler.running), 3)


if __name__ == "__main__":
    unittest.main()
