# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract the operators a vLLM model executes, without running it.

With `device="meta"` a model is built from its HF config alone -- no weights
read, no accelerator memory allocated, no kernel executed -- while the real host
platform stays in control, so the attention backend, KV cache layout, dtype and
block sizes are the genuine ones for this machine. Custom ops stay visible as
themselves rather than being decomposed.

    from vllm.profiler.op_capture import capture_model_ops, format_report

    print(format_report(capture_model_ops("Qwen/Qwen3-30B-A3B")))
"""

from vllm.profiler.op_capture.capture import (
    BatchSpec,
    CaptureFailure,
    ForwardHarness,
    OpCapture,
    SelectionMetadata,
    capture_batches,
    capture_model_ops,
)
from vllm.profiler.op_capture.meta_ops import (
    UnsupportedMetaOpError,
    register_meta_impls,
)
from vllm.profiler.op_capture.parallel import capture_ranks
from vllm.profiler.op_capture.recorder import OpRecorder, RecordedOp
from vllm.profiler.op_capture.report import (
    format_batches,
    format_report,
    write_capture_files,
)
from vllm.profiler.op_capture.trace import (
    TraceDiff,
    capture_execution_trace,
    compare_devices,
    compare_traces,
    format_diff,
    load_trace,
)

__all__ = [
    "BatchSpec",
    "CaptureFailure",
    "ForwardHarness",
    "OpCapture",
    "OpRecorder",
    "RecordedOp",
    "SelectionMetadata",
    "TraceDiff",
    "UnsupportedMetaOpError",
    "capture_batches",
    "capture_execution_trace",
    "capture_model_ops",
    "capture_ranks",
    "compare_devices",
    "compare_traces",
    "format_batches",
    "format_diff",
    "format_report",
    "load_trace",
    "register_meta_impls",
    "write_capture_files",
]
