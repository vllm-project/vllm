# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from benchmarks.offloading.bench_offloading_e2e import (
    BENCHMARK_NAME,
    atomic_write_json,
    cudagraph_dispatch_metrics,
    decode_verification_needs_refresh,
    expected_replay_sources,
    metric_value,
    refresh_result,
    resume_config_errors,
    run_decode_block_verification,
    run_offload_attempt,
    summarize,
    validate_cold,
    validate_decode_block_stores,
    validate_replay,
    validate_server_metadata,
)
from benchmarks.offloading.summarize_offloading_benchmark import (
    load_result,
    render_markdown,
    validate_compatible,
)


def measurement(
    *,
    tokens: int,
    compute: int,
    local_hit: int,
    external: int,
    output_token: int = 7,
    cpu_to_gpu_bytes: int = 0,
) -> dict:
    return {
        "cache_salt": "x" * 64,
        "prompt_token": 1,
        "prompt_tokens": tokens,
        "client_e2e_seconds": 0.1,
        "server_ttft_seconds": 0.08,
        "server_prefill_seconds": 0.07,
        "local_compute_tokens": float(compute),
        "local_cache_hit_tokens": float(local_hit),
        "external_kv_tokens": float(external),
        "cpu_to_gpu_bytes": float(cpu_to_gpu_bytes),
        "gpu_to_cpu_bytes": 0.0,
        "output_token_ids": [output_token],
        "output_text": "",
        "request_id": "request-id",
    }


class FakeClient:
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.seen_salts: set[str] = set()
        self.calls: list[tuple[int, int, str]] = []

    def measure_completion(
        self, *, prompt_token: int, num_tokens: int, cache_salt: str
    ) -> dict:
        self.calls.append((prompt_token, num_tokens, cache_salt))
        if cache_salt not in self.seen_salts:
            self.seen_salts.add(cache_salt)
            return measurement(
                tokens=num_tokens,
                compute=num_tokens,
                local_hit=0,
                external=0,
                output_token=prompt_token,
            )

        external, compute = expected_replay_sources(num_tokens, self.block_size)
        return measurement(
            tokens=num_tokens,
            compute=compute,
            local_hit=0,
            external=external,
            output_token=prompt_token,
            cpu_to_gpu_bytes=external * 128,
        )


class FakeDecodeClient:
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.decode_tokens: list[int] = []

    def measure_decode_store(
        self, *, prompt_token: int, decode_tokens: int, cache_salt: str
    ) -> dict:
        self.decode_tokens.append(decode_tokens)
        blocks = (decode_tokens - 1) // self.block_size
        return {
            "cache_salt": cache_salt,
            "prompt_tokens": 1,
            "completion_tokens": decode_tokens,
            "client_e2e_seconds": 0.1,
            "gpu_to_cpu_operations": float(blocks * 8),
            "gpu_to_cpu_bytes": float(blocks * 100),
            "request_id": f"request-{prompt_token}",
        }

    def measure_decode_store_lifecycle(
        self,
        *,
        prompt_token: int,
        decode_tokens: int,
        cache_salt: str,
        drain_prompt_token: int,
    ) -> dict:
        result = self.measure_decode_store(
            prompt_token=prompt_token,
            decode_tokens=decode_tokens,
            cache_salt=cache_salt,
        )
        self.measure_decode_store(
            prompt_token=drain_prompt_token,
            decode_tokens=1,
            cache_salt=f"{cache_salt}-drain",
        )
        return result


class BenchmarkTests(unittest.TestCase):
    def test_cudagraph_dispatch_metrics_records_runtime_mode(self):
        before = "\n".join(
            f'vllm:cudagraph_dispatch_total{{engine="0",runtime_mode="{mode}"}} 4'
            for mode in ("NONE", "PIECEWISE", "FULL")
        )
        after = before.replace(
            'runtime_mode="PIECEWISE"} 4', 'runtime_mode="PIECEWISE"} 5'
        )
        result = cudagraph_dispatch_metrics(before, after, 0)
        self.assertEqual(
            result["cudagraph_runtime_mode_counts"],
            {"NONE": 0, "PIECEWISE": 1, "FULL": 0},
        )
        self.assertEqual(result["cudagraph_runtime_mode"], "PIECEWISE")
        self.assertTrue(result["cudagraph_metrics_available"])
        self.assertTrue(result["cudagraph_used"])

    def test_cudagraph_dispatch_metrics_reports_unavailable(self):
        result = cudagraph_dispatch_metrics("other_metric 1", "other_metric 2", 0)
        self.assertFalse(result["cudagraph_metrics_available"])
        self.assertEqual(
            result["cudagraph_runtime_mode_counts"],
            {"NONE": 0, "PIECEWISE": 0, "FULL": 0},
        )
        self.assertIsNone(result["cudagraph_runtime_mode"])
        self.assertIsNone(result["cudagraph_used"])

    def test_cudagraph_dispatch_metrics_requires_request_delta(self):
        metrics = 'vllm:cudagraph_dispatch_total{engine="0",runtime_mode="NONE"} 4'
        result = cudagraph_dispatch_metrics(metrics, metrics, 0)
        self.assertTrue(result["cudagraph_metrics_available"])
        self.assertIsNone(result["cudagraph_runtime_mode"])
        self.assertIsNone(result["cudagraph_used"])

    def test_decode_verification_runs_one_step_past_each_boundary(self):
        client = FakeDecodeClient()
        result = run_decode_block_verification(
            client,
            run_id="test-run",
            prompt_token=100,
            block_size=16,
            blocks=4,
        )
        self.assertTrue(result["valid"], result["validation_errors"])
        self.assertEqual(client.decode_tokens, [17, 1, 65, 1])

    def test_decode_verification_supports_speculative_tail(self):
        client = FakeDecodeClient()
        result = run_decode_block_verification(
            client,
            run_id="test-run",
            prompt_token=100,
            block_size=16,
            blocks=4,
            tail_tokens=3,
        )
        self.assertTrue(result["valid"], result["validation_errors"])
        self.assertEqual(result["tail_tokens"], 3)
        self.assertEqual(client.decode_tokens, [19, 1, 67, 1])

    def test_decode_block_store_validation_scales_from_calibration(self):
        calibration = {
            "prompt_tokens": 1,
            "gpu_to_cpu_operations": 8.0,
            "gpu_to_cpu_bytes": 100.0,
        }
        multi_block = {
            "prompt_tokens": 1,
            "gpu_to_cpu_operations": 32.0,
            "gpu_to_cpu_bytes": 400.0,
        }
        self.assertEqual(validate_decode_block_stores(calibration, multi_block, 4), [])
        multi_block["gpu_to_cpu_operations"] = 8.0
        self.assertTrue(validate_decode_block_stores(calibration, multi_block, 4))

    def test_summary_rejects_result_without_server_metadata(self):
        result = {
            "schema_version": 2,
            "benchmark": BENCHMARK_NAME,
            "mode": "recompute",
            "status": "completed",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.json"
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(RuntimeError, "missing server metadata"):
                load_result(path, "recompute")

    def test_report_compatibility_checks_config_and_server(self):
        base = {
            "config": {
                "model": "model",
                "engine": 0,
                "prompt_token": 1,
                "block_size": 16,
                "max_tokens": 1,
            },
            "server": {"version": "1", "model": {"id": "model"}},
        }
        validate_compatible(base, json.loads(json.dumps(base)))
        different = json.loads(json.dumps(base))
        different["config"]["block_size"] = 32
        with self.assertRaisesRegex(RuntimeError, "block_size"):
            validate_compatible(base, different)
        different = json.loads(json.dumps(base))
        different["server"]["version"] = "2"
        with self.assertRaisesRegex(RuntimeError, "server metadata"):
            validate_compatible(base, different)

    def test_report_renders_decode_verification(self):
        def result(mode: str) -> dict:
            attempt = {
                "valid": True,
                "request": measurement(tokens=16, compute=16, local_hit=0, external=0),
            }
            if mode == "offload":
                attempt = {
                    "valid": True,
                    "cold": measurement(tokens=16, compute=16, local_hit=0, external=0),
                    "replay": measurement(
                        tokens=16,
                        compute=1,
                        local_hit=0,
                        external=16,
                        cpu_to_gpu_bytes=100,
                    ),
                }
            item = {"tokens": 16, "attempts": [attempt]}
            refresh_result(mode, item)
            return {
                "config": {"model": "model", "block_size": 16},
                "server": {
                    "version": "1",
                    "model": {"root": "model-root"},
                },
                "results": [item],
            }

        recompute = result("recompute")
        offload = result("offload")
        mixed = {
            "tokens": 24,
            "attempts": [
                {
                    "valid": True,
                    "cold": measurement(tokens=24, compute=24, local_hit=0, external=0),
                    "replay": measurement(
                        tokens=24,
                        compute=8,
                        local_hit=0,
                        external=16,
                        cpu_to_gpu_bytes=100,
                    ),
                }
            ],
        }
        refresh_result("offload", mixed)
        offload["results"].append(mixed)
        offload["decode_block_verification"] = {
            "valid": True,
            "blocks": 4,
            "calibration": {
                "gpu_to_cpu_operations": 8,
                "gpu_to_cpu_bytes": 100,
            },
            "multi_block": {
                "gpu_to_cpu_operations": 32,
                "gpu_to_cpu_bytes": 400,
            },
        }
        report = render_markdown(
            recompute,
            offload,
            Path("recompute.json"),
            Path("offload.json"),
        )
        self.assertIn("## Decode Block Store", report)
        self.assertIn("8 -> 32", report)
        self.assertIn("## Offload-Only Mixed Replay Validation", report)
        self.assertIn("| 24 | 1 / 1 | 0 | 16 | 8 | unavailable | 100 |", report)

    def test_summarize(self):
        result = summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 4.0)

    def test_expected_replay_sources(self):
        self.assertEqual(expected_replay_sources(256, 16), (256, 1))
        self.assertEqual(expected_replay_sources(512, 16), (512, 1))
        self.assertEqual(expected_replay_sources(1024, 16), (1024, 1))
        self.assertEqual(expected_replay_sources(2048, 16), (2048, 1))
        self.assertEqual(expected_replay_sources(2050, 16), (2048, 2))
        for aligned in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
            with self.subTest(tokens=aligned + 8):
                self.assertEqual(expected_replay_sources(aligned + 8, 16), (aligned, 8))

    def test_metric_value_filters_engine_and_labels(self):
        metrics = "\n".join(
            [
                'metric{engine="0",source="local"} 1',
                'metric{engine="1",source="external"} 2',
            ]
        )
        self.assertEqual(metric_value(metrics, "metric", 1, ('source="external"',)), 2)
        self.assertEqual(metric_value(metrics, "metric", 0, ('source="external"',)), 0)

    def test_cold_validation(self):
        cold = measurement(tokens=2048, compute=2048, local_hit=0, external=0)
        self.assertEqual(validate_cold(cold, 2048), [])
        cold["local_cache_hit_tokens"] = 16.0
        self.assertIn("local_cache_hit_tokens", validate_cold(cold, 2048)[0])

    def test_replay_requires_zero_hbm_hit_and_transfer(self):
        replay = measurement(
            tokens=2048,
            compute=1,
            local_hit=0,
            external=2048,
            cpu_to_gpu_bytes=1024,
        )
        self.assertEqual(
            validate_replay(
                replay,
                tokens=2048,
                block_size=16,
                require_transfer_bytes=True,
            ),
            [],
        )
        replay["local_cache_hit_tokens"] = 16.0
        errors = validate_replay(
            replay,
            tokens=2048,
            block_size=16,
            require_transfer_bytes=True,
        )
        self.assertTrue(any("local_cache_hit_tokens" in error for error in errors))

    def test_offload_attempt_reuses_only_target_salt(self):
        client = FakeClient()
        attempt = run_offload_attempt(
            client,
            run_id="test-run",
            tokens=2048,
            attempt=1,
            prompt_token=31318,
            eviction_prompt_token=31319,
            eviction_plan=[12000, 12000],
            block_size=16,
            require_transfer_bytes=True,
        )
        self.assertTrue(attempt["valid"], attempt["validation_errors"])
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(client.calls[0][2], client.calls[3][2])
        self.assertNotEqual(client.calls[0][2], client.calls[1][2])
        self.assertNotEqual(client.calls[1][2], client.calls[2][2])

    def test_refresh_result_uses_only_valid_attempts(self):
        valid = {
            "valid": True,
            "request": measurement(tokens=16, compute=16, local_hit=0, external=0),
        }
        invalid = {"valid": False, "validation_errors": ["bad"]}
        result: dict[str, Any] = {"attempts": [valid, invalid]}
        refresh_result("recompute", result)
        self.assertEqual(result["valid_samples"], 1)
        self.assertEqual(result["total_attempts"], 2)
        self.assertEqual(result["summary"]["server_ttft_seconds"]["count"], 1)

    def test_refresh_result_summarizes_cudagraph_dispatches(self):
        request = measurement(tokens=16, compute=16, local_hit=0, external=0)
        request.update(
            cudagraph_metrics_available=True,
            cudagraph_runtime_mode_counts={"NONE": 0, "PIECEWISE": 1, "FULL": 0},
            cudagraph_runtime_mode="PIECEWISE",
            cudagraph_used=True,
        )
        result: dict[str, Any] = {"attempts": [{"valid": True, "request": request}]}
        refresh_result("recompute", result)
        self.assertEqual(
            result["cudagraph_summary"],
            {
                "runtime_mode_counts": {"NONE": 0, "PIECEWISE": 1, "FULL": 0},
                "used_samples": 1,
                "not_used_samples": 0,
                "unavailable_samples": 0,
            },
        )

    def test_refresh_result_tracks_unavailable_cudagraph_metrics(self):
        request = measurement(tokens=16, compute=16, local_hit=0, external=0)
        request.update(
            cudagraph_metrics_available=False,
            cudagraph_runtime_mode_counts={"NONE": 0, "PIECEWISE": 0, "FULL": 0},
            cudagraph_runtime_mode=None,
            cudagraph_used=None,
        )
        result: dict[str, Any] = {"attempts": [{"valid": True, "request": request}]}
        refresh_result("recompute", result)
        self.assertEqual(result["cudagraph_summary"]["unavailable_samples"], 1)
        self.assertEqual(result["cudagraph_summary"]["not_used_samples"], 0)

    def test_atomic_write_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"status": "running"})
            self.assertEqual(json.loads(path.read_text()), {"status": "running"})
            self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_server_metadata_must_not_change_on_resume(self):
        metadata = {"version": "1", "model": {"id": "model"}}
        validate_server_metadata(metadata, dict(metadata))
        with self.assertRaises(RuntimeError):
            validate_server_metadata(
                metadata, {"version": "2", "model": {"id": "model"}}
            )

    def test_decode_verification_resume_checks_tail_tokens(self):
        verification = {
            "valid": True,
            "blocks": 4,
            "block_size": 16,
            "tail_tokens": 1,
        }

        self.assertFalse(
            decode_verification_needs_refresh(
                verification,
                blocks=4,
                block_size=16,
                tail_tokens=1,
            )
        )
        self.assertTrue(
            decode_verification_needs_refresh(
                verification,
                blocks=4,
                block_size=16,
                tail_tokens=3,
            )
        )

        verification.pop("tail_tokens")
        self.assertTrue(
            decode_verification_needs_refresh(
                verification,
                blocks=4,
                block_size=16,
                tail_tokens=1,
            )
        )

    def test_resume_only_allows_operational_fields_to_change(self):
        old = {
            "model": "model",
            "block_size": 16,
            "repeats": 1,
            "max_attempts": 2,
            "notes": "old",
            "verify_decode_blocks": 4,
            "verify_decode_tail_tokens": 1,
        }
        new = dict(
            old,
            repeats=2,
            max_attempts=4,
            notes="new",
            verify_decode_blocks=0,
            verify_decode_tail_tokens=3,
        )
        self.assertEqual(resume_config_errors(old, new), [])
        new["block_size"] = 32
        self.assertTrue(resume_config_errors(old, new))


if __name__ == "__main__":
    unittest.main()
