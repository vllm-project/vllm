# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from unittest.mock import MagicMock

from vllm.compilation.cuda_graph import CUDAGraphLogging, CUDAGraphStat
from vllm.config import CUDAGraphMode, ObservabilityConfig, VllmConfig
from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput


class TestCudaGraphMetrics(unittest.TestCase):
    def test_cudagraph_stat_dataclass(self):
        stat = CUDAGraphStat(
            num_unpadded_tokens=10,
            num_padded_tokens=16,
            num_paddings=6,
            runtime_mode="CUDAGraphMode.PIECEWISE",
        )
        self.assertEqual(stat.num_unpadded_tokens, 10)
        self.assertEqual(stat.num_padded_tokens, 16)
        self.assertEqual(stat.num_paddings, 6)
        self.assertEqual(stat.runtime_mode, "CUDAGraphMode.PIECEWISE")

    def test_cudagraph_logging_observe_and_table_generation(self):
        cg_logging = CUDAGraphLogging(
            cg_mode=CUDAGraphMode.PIECEWISE,
            cg_capture_sizes=[8, 16, 32],
        )

        stat1 = CUDAGraphStat(
            num_unpadded_tokens=10,
            num_padded_tokens=16,
            num_paddings=6,
            runtime_mode="CUDAGraphMode.PIECEWISE",
        )
        stat2 = CUDAGraphStat(
            num_unpadded_tokens=4,
            num_padded_tokens=8,
            num_paddings=4,
            runtime_mode="CUDAGraphMode.FULL",
        )

        cg_logging.observe(stat1)
        cg_logging.observe(stat1)
        cg_logging.observe(stat2)

        table = cg_logging.generate_metric_table()
        self.assertIn("**CUDAGraph Config Settings:**", table)
        self.assertIn("**CUDAGraph Stats:**", table)
        self.assertIn("Unpadded Tokens", table)
        self.assertIn("Padded Tokens", table)
        self.assertIn("Num Paddings", table)
        self.assertIn("Runtime Mode", table)
        self.assertIn("Count", table)

        # Verify rows
        self.assertIn("10", table)
        self.assertIn("16", table)
        self.assertIn("6", table)

        # Verify log output and reset
        mock_log = MagicMock()
        cg_logging.log(log_fn=mock_log)
        self.assertEqual(mock_log.call_count, 1)
        self.assertIn("**CUDAGraph Stats:**", mock_log.call_args[0][0])

        # After log, stats should be reset
        self.assertEqual(len(cg_logging.stats), 0)

    def test_logging_stat_logger_records_and_emits_cudagraph_metrics(self):
        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.observability_config = ObservabilityConfig(cudagraph_metrics=True)
        vllm_config.compilation_config = MagicMock()
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        vllm_config.compilation_config.cudagraph_capture_sizes = [8, 16, 32]
        vllm_config.model_config = None
        vllm_config.kv_transfer_config = None

        logger = LoggingStatLogger(vllm_config=vllm_config, engine_index=0)
        self.assertIsNotNone(logger.cudagraph_logging)

        stat = CUDAGraphStat(
            num_unpadded_tokens=7,
            num_padded_tokens=8,
            num_paddings=1,
            runtime_mode="CUDAGraphMode.PIECEWISE",
        )
        scheduler_stats = SchedulerStats(cudagraph_stats=stat)

        # Record should pass stat to cudagraph_logging
        logger.record(
            scheduler_stats=scheduler_stats,
            iteration_stats=None,
        )
        self.assertEqual(len(logger.cudagraph_logging.stats), 1)
        self.assertEqual(logger.cudagraph_logging.stats[0], stat)

        # Log should emit cudagraph stats
        mock_log = MagicMock()
        logger.cudagraph_logging.log(log_fn=mock_log)
        self.assertEqual(mock_log.call_count, 1)
        self.assertIn("7", mock_log.call_args[0][0])
        self.assertIn("8", mock_log.call_args[0][0])

    def test_model_runner_output_carries_cudagraph_stats(self):
        stat = CUDAGraphStat(
            num_unpadded_tokens=12,
            num_padded_tokens=16,
            num_paddings=4,
            runtime_mode="CUDAGraphMode.FULL",
        )
        output = ModelRunnerOutput(
            req_ids=["req-1"],
            req_id_to_index={"req-1": 0},
            cudagraph_stats=stat,
        )
        self.assertIsNotNone(output.cudagraph_stats)
        self.assertEqual(output.cudagraph_stats.num_unpadded_tokens, 12)
        self.assertEqual(output.cudagraph_stats.num_padded_tokens, 16)
        self.assertEqual(output.cudagraph_stats.runtime_mode, "CUDAGraphMode.FULL")


if __name__ == "__main__":
    unittest.main()
