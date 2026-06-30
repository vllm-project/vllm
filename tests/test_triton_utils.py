# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from unittest import mock

from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder


def test_triton_placeholder_is_module():
    triton = TritonPlaceholder()
    assert isinstance(triton, types.ModuleType)
    assert triton.__name__ == "triton"


def test_triton_language_placeholder_is_module():
    triton_language = TritonLanguagePlaceholder()
    assert isinstance(triton_language, types.ModuleType)
    assert triton_language.__name__ == "triton.language"


def test_triton_placeholder_decorators():
    triton = TritonPlaceholder()

    @triton.jit
    def foo(x):
        return x

    @triton.autotune
    def bar(x):
        return x

    @triton.heuristics
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_decorators_with_args():
    triton = TritonPlaceholder()

    @triton.jit(debug=True)
    def foo(x):
        return x

    @triton.autotune(configs=[], key="x")
    def bar(x):
        return x

    @triton.heuristics({"BLOCK_SIZE": lambda args: 128 if args["x"] > 1024 else 64})
    def baz(x):
        return x

    assert foo(1) == 1
    assert bar(2) == 2
    assert baz(3) == 3


def test_triton_placeholder_language():
    lang = TritonLanguagePlaceholder()
    assert isinstance(lang, types.ModuleType)
    assert lang.__name__ == "triton.language"
    assert lang.constexpr is None
    assert lang.dtype is None
    assert lang.int64 is None
    assert lang.int32 is None
    assert lang.tensor is None


def test_triton_placeholder_language_from_parent():
    triton = TritonPlaceholder()
    lang = triton.language
    assert isinstance(lang, TritonLanguagePlaceholder)


def test_no_triton_fallback():
    # clear existing triton modules
    sys.modules.pop("triton", None)
    sys.modules.pop("triton.language", None)
    sys.modules.pop("vllm.triton_utils", None)
    sys.modules.pop("vllm.triton_utils.importing", None)

    # mock triton not being installed
    with mock.patch.dict(sys.modules, {"triton": None}):
        from vllm.triton_utils import HAS_TRITON, tl, triton

        assert HAS_TRITON is False
        assert triton.__class__.__name__ == "TritonPlaceholder"
        assert triton.language.__class__.__name__ == "TritonLanguagePlaceholder"
        assert tl.__class__.__name__ == "TritonLanguagePlaceholder"


def test_prefill_chunk_metadata_compile_keys_trace_specializations():
    for module_name in (
        "triton",
        "triton.language",
        "vllm.triton_utils",
        "vllm.triton_utils.importing",
        "vllm.triton_utils.warmup",
        "vllm.v1.attention.backends.mla.indexer",
        "vllm.v1.attention.backends.mla.sparse_swa",
    ):
        sys.modules.pop(module_name, None)

    with mock.patch.dict(sys.modules, {"triton": None}):
        from vllm.triton_utils.warmup import (
            WarmupIntRange,
            trace_triton_kernel_specialization_args,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            BUILD_PREFILL_CHUNK_METADATA_KERNEL,
            PrefillChunkMetadataKernelCompileKey,
            _build_prefill_chunk_metadata_kernel,
        )

        assert trace_triton_kernel_specialization_args(
            _build_prefill_chunk_metadata_kernel
        ) == (
            "query_slice_start",
            "query_slice_stop",
            "BLOCK_SIZE",
            "COMPRESS_RATIO",
        )

        keys = tuple(
            dict.fromkeys(
                BUILD_PREFILL_CHUNK_METADATA_KERNEL.compile_key(
                    {
                        "query_slice_start": query_slice_start,
                        "query_slice_stop": query_slice_stop,
                        "BLOCK_SIZE": 1024,
                        "COMPRESS_RATIO": compress_ratio,
                    }
                )
                for compress_ratio in (1, 4)
                for query_slice_start, query_slice_stop in ((0, 16), (1, 15))
            )
        )

        assert keys == (
            PrefillChunkMetadataKernelCompileKey(
                query_slice_start=0,
                query_slice_stop=16,
                BLOCK_SIZE=1024,
                COMPRESS_RATIO=1,
            ),
            PrefillChunkMetadataKernelCompileKey(
                query_slice_start=1,
                query_slice_stop=15,
                BLOCK_SIZE=1024,
                COMPRESS_RATIO=1,
            ),
            PrefillChunkMetadataKernelCompileKey(
                query_slice_start=0,
                query_slice_stop=16,
                BLOCK_SIZE=1024,
                COMPRESS_RATIO=4,
            ),
            PrefillChunkMetadataKernelCompileKey(
                query_slice_start=1,
                query_slice_stop=15,
                BLOCK_SIZE=1024,
                COMPRESS_RATIO=4,
            ),
        )

        from vllm.v1.attention.backends.mla.sparse_swa import (
            ComputePrefillMetadataKernel,
            _compute_prefill_metadata_kernel,
        )

        assert trace_triton_kernel_specialization_args(
            _compute_prefill_metadata_kernel
        ) == ("BLOCK_SIZE",)

        kernel = ComputePrefillMetadataKernel()
        assert kernel.compile_key(
            {
                "num_prefills": 3,
            }
        ) == ComputePrefillMetadataKernel.CompileKey(
            BLOCK_SIZE=4,
        )
        assert kernel._trace_dispatch(kernel.dispatch)(
            num_prefills=WarmupIntRange(1, 5),
        ) == [
            ComputePrefillMetadataKernel.CompileKey(
                BLOCK_SIZE=1,
            ),
            ComputePrefillMetadataKernel.CompileKey(
                BLOCK_SIZE=2,
            ),
            ComputePrefillMetadataKernel.CompileKey(
                BLOCK_SIZE=4,
            ),
        ]

        import torch

        from vllm.v1.attention.backends.mla import indexer, sparse_swa

        class FakeKernel:
            def __init__(self):
                self.warmup_calls = []
                self.launch_calls = []

            def warmup(self, *args, grid, **kwargs):
                self.warmup_calls.append((args, grid, kwargs))

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    self.launch_calls.append((grid, args, kwargs))

                return launch

        fake_build_kernel = FakeKernel()
        build_key = PrefillChunkMetadataKernelCompileKey(
            query_slice_start=0,
            query_slice_stop=16,
            BLOCK_SIZE=1024,
            COMPRESS_RATIO=4,
        )
        with mock.patch.object(
            indexer, "_build_prefill_chunk_metadata_kernel", fake_build_kernel
        ):
            build_call = BUILD_PREFILL_CHUNK_METADATA_KERNEL.compile(build_key)
            build_call(
                num_reqs=2,
                query_start_loc_ptr="query_start_loc",
                uncompressed_seq_lens_ptr="uncompressed_seq_lens",
                cu_compressed_seq_lens_ptr="cu_seq_lens",
                token_to_seq_ptr="token_to_seq",
                cu_compressed_seq_len_ks_ptr="cu_seq_len_ks",
                cu_compressed_seq_len_ke_ptr="cu_seq_len_ke",
                query_slice_start=0,
                query_slice_stop=16,
            )

        assert fake_build_kernel.warmup_calls == [
            (
                (
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    0,
                    16,
                ),
                (1,),
                {
                    "BLOCK_SIZE": 1024,
                    "COMPRESS_RATIO": 4,
                },
            )
        ]
        assert fake_build_kernel.launch_calls == [
            (
                (2,),
                (),
                {
                    "query_start_loc_ptr": "query_start_loc",
                    "uncompressed_seq_lens_ptr": "uncompressed_seq_lens",
                    "cu_compressed_seq_lens_ptr": "cu_seq_lens",
                    "token_to_seq_ptr": "token_to_seq",
                    "cu_compressed_seq_len_ks_ptr": "cu_seq_len_ks",
                    "cu_compressed_seq_len_ke_ptr": "cu_seq_len_ke",
                    "query_slice_start": 0,
                    "query_slice_stop": 16,
                    "BLOCK_SIZE": 1024,
                    "COMPRESS_RATIO": 4,
                },
            )
        ]

        fake_compute_kernel = FakeKernel()
        compute_key = ComputePrefillMetadataKernel.CompileKey(BLOCK_SIZE=4)
        with mock.patch.object(
            sparse_swa, "_compute_prefill_metadata_kernel", fake_compute_kernel
        ):
            compute_call = kernel.compile(compute_key)
            compute_call(
                prefill_gather_lens_ptr="prefill_gather_lens",
                seq_lens_ptr="seq_lens",
                query_start_loc_ptr="query_start_loc",
                num_prefills=3,
                num_decodes=2,
                window_size=128,
            )

        assert fake_compute_kernel.warmup_calls == [
            (
                (torch.int32, torch.int32, torch.int32, 4, 0, 1),
                (1,),
                {"BLOCK_SIZE": 4},
            )
        ]
        assert fake_compute_kernel.launch_calls == [
            (
                (1,),
                (),
                {
                    "prefill_gather_lens_ptr": "prefill_gather_lens",
                    "seq_lens_ptr": "seq_lens",
                    "query_start_loc_ptr": "query_start_loc",
                    "num_prefills": 3,
                    "num_decodes": 2,
                    "window_size": 128,
                    "BLOCK_SIZE": 4,
                },
            )
        ]
