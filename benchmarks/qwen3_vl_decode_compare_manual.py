# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare full-resolution OpenCV and TorchCodec decoding through vLLM."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import resource
import statistics
import subprocess
import threading
from pathlib import Path
from time import perf_counter

import cv2
import psutil
import pynvml
import torch

from vllm.multimodal.video import Qwen3VLVideoBackend

MIB = 1024**2


class MemoryMonitor:
    def __init__(self, process, handle):
        self.process = process
        self.handle = handle
        self.rss_peak = 0
        self.gpu_peak = 0
        self.error = None
        self.stop_event = threading.Event()
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
        used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
        self.gpu_peak = max(self.gpu_peak, used)

    def run(self):
        try:
            while not self.stop_event.wait(0.005):
                self.sample()
        except Exception as exc:
            self.error = exc

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        if self.error is not None:
            raise RuntimeError("Memory monitor failed") from self.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--backend", choices=("opencv", "torchcodec"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    torch.set_num_threads(1)
    gpu = args.backend == "torchcodec"
    fallback_checks = []

    if gpu:
        from vllm.multimodal.video_decoders.torchcodec import (
            TorchCodecVideoBackendMixin,
        )

        original = TorchCodecVideoBackendMixin.decode_torchcodec_frames

        def strict_decode(decoder, indices, *, device="cpu"):
            result = original(decoder, indices, device=device)
            status = getattr(decoder, "cpu_fallback", None)
            if status is None or bool(status):
                raise RuntimeError(f"NVDEC verification failed: {status}")
            if not result[0].is_cuda:
                raise RuntimeError("TorchCodec returned non-CUDA frames")
            fallback_checks.append(str(status))
            return result

        TorchCodecVideoBackendMixin.decode_torchcodec_frames = staticmethod(
            strict_decode
        )
        torch.cuda.init()
        torch.accelerator.synchronize()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    process = psutil.Process()
    data = args.video.read_bytes()
    input_sha256 = hashlib.sha256(data).hexdigest()

    kwargs = {}
    if gpu:
        kwargs = {
            "device": "cuda",
            "seek_mode": "exact",
            "num_ffmpeg_threads": 0,
        }

    def decode():
        return Qwen3VLVideoBackend.load_bytes(
            data,
            backend=args.backend,
            num_frames=-1,
            fps=2,
            max_frames=600,
            frame_recovery=False,
            **kwargs,
        )

    # Warmup is excluded from timing, but included in lifetime RSS high water.
    warm_frames, info = decode()
    if gpu:
        torch.accelerator.synchronize()

    expected_indices = list(info["frames_indices"])
    expected_shape = list(warm_frames.shape)
    assert expected_shape[0] == len(expected_indices)
    assert expected_shape[1:] == [1080, 1920, 3]

    if args.reference:
        reference = json.loads(args.reference.read_text())
        assert reference["input_sha256"] == input_sha256, "Input differs"
        assert reference["sampled_indices"] == expected_indices, (
            "CPU/GPU sampled indices differ"
        )
        assert reference["output_shape"] == expected_shape, "Shape differs"

    del warm_frames
    gc.collect()
    if gpu:
        torch.accelerator.synchronize()

    rows = []
    for repeat in range(args.repeats):
        if gpu:
            torch.accelerator.synchronize()
            torch.accelerator.memory.reset_peak_memory_stats()

        rss_before = process.memory_info().rss
        gpu_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        monitor = MemoryMonitor(process, handle)
        monitor.start()

        try:
            start = perf_counter()
            frames, current = decode()
            if gpu:
                torch.accelerator.synchronize()
            seconds = perf_counter() - start
        finally:
            monitor.stop()

        assert list(current["frames_indices"]) == expected_indices
        assert list(frames.shape) == expected_shape
        dtype = str(frames.dtype).removeprefix("torch.")
        assert dtype == "uint8"
        if gpu:
            assert frames.is_cuda

        output_bytes = frames.numel() * frames.element_size() if gpu else frames.nbytes
        row = {
            "repeat": repeat + 1,
            "seconds": seconds,
            "rss_before_mib": rss_before / MIB,
            "rss_after_mib": process.memory_info().rss / MIB,
            "rss_peak_sampled_mib": monitor.rss_peak / MIB,
            "gpu_device_before_mib": gpu_before / MIB,
            "gpu_device_peak_sampled_mib": monitor.gpu_peak / MIB,
            "torch_peak_allocated_mib": (
                torch.accelerator.memory.max_memory_allocated() / MIB if gpu else None
            ),
            "torch_peak_reserved_mib": (
                torch.accelerator.memory.max_memory_reserved() / MIB if gpu else None
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del frames
        gc.collect()
        if gpu:
            torch.accelerator.synchronize()

    result = {
        "video": str(args.video.resolve()),
        "backend": args.backend,
        "input_sha256": input_sha256,
        "input_bytes": len(data),
        "source_frames": info["total_num_frames"],
        "source_fps": info["fps"],
        "sample_fps": 2,
        "sampled_indices": expected_indices,
        "output_shape": expected_shape,
        "output_dtype": dtype,
        "output_device": "cuda:0" if gpu else "cpu",
        "output_mib": output_bytes / MIB,
        "warmups": 1,
        "repeats": args.repeats,
        "seconds_median": statistics.median(x["seconds"] for x in rows),
        "rss_lifetime_peak_mib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "memory_sampling_interval_ms": 5,
        "cpu_fallback_checks": fallback_checks,
        "torch_threads": torch.get_num_threads(),
        "decoder_options": kwargs,
        "decoder_thread_policy": "vLLM backend defaults",
        "cuda_cache_policy": "retained between repetitions",
        "timing_scope": (
            "vLLM load_bytes, including decoder construction and metadata; "
            "CUDA synchronized; excludes file read and preprocessing"
        ),
        "gpu_uuid": pynvml.nvmlDeviceGetUUID(handle),
        "driver_version": pynvml.nvmlSystemGetDriverVersion(),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "cpu_affinity": process.cpu_affinity(),
        "vllm_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchcodec", "transformers", "vllm")
        },
        "opencv_version": cv2.__version__,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    pynvml.nvmlShutdown()

    print("\nSaved:", args.output)
    print("Backend:", args.backend)
    print("Sampled frames:", len(expected_indices))
    print("Output shape:", expected_shape)
    print("Median seconds:", result["seconds_median"])
    print("Lifetime RSS peak MiB:", result["rss_lifetime_peak_mib"])


if __name__ == "__main__":
    main()
