# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the inline-vectors wire-format packing path.

Covers:

- ``pack_effective_steering`` produces the right dtype + values
- ``_torch_dtype_to_pack_dtype`` mapping
- ``maybe_pack_inline_steering_for_request`` mutates SamplingParams correctly
- Hash determinism: packed and unpacked submissions of the same logical
  request share a prefix-cache hash
- msgspec round-trip of SamplingParams with packed fields preserves
  ndarray dtype + values
"""

import numpy as np
import torch

from vllm.config.steering_types import (
    _torch_dtype_to_pack_dtype,
    maybe_pack_inline_steering_for_request,
    pack_effective_steering,
    pack_steering_for_dtype,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

# ---------------------------------------------------------------------------
# pack helpers
# ---------------------------------------------------------------------------


class TestPackHelpers:
    def test_torch_dtype_mapping(self):
        assert _torch_dtype_to_pack_dtype(torch.float16) == np.dtype(np.float16)
        assert _torch_dtype_to_pack_dtype(torch.float32) == np.dtype(np.float32)
        assert _torch_dtype_to_pack_dtype(torch.float64) == np.dtype(np.float64)
        # bf16 has no numpy equivalent without ml_dtypes; should fall back
        # to float32 (still a 2.25× IPC reduction over msgpack-encoded floats).
        assert _torch_dtype_to_pack_dtype(torch.bfloat16) == np.dtype(np.float32)

    def test_pack_steering_for_dtype_bare_list(self):
        spec = {"post_mlp": {0: [1.0, 2.0, 3.0]}}
        out = pack_steering_for_dtype(spec, np.float32)
        assert out is not None
        arr = out["post_mlp"][0]
        assert arr.dtype == np.float32
        assert arr.tolist() == [1.0, 2.0, 3.0]

    def test_pack_steering_for_dtype_with_scale(self):
        spec = {"post_mlp": {0: {"vector": [1.0, 2.0], "scale": 3.0}}}
        out = pack_steering_for_dtype(spec, np.float32)
        assert out is not None
        assert out["post_mlp"][0].tolist() == [3.0, 6.0]

    def test_pack_effective_steering_resolves_then_casts(self):
        base = {"post_mlp": {0: [1.0, 2.0]}}
        prefill = {"post_mlp": {0: [10.0, 20.0]}}
        out = pack_effective_steering(base, prefill, np.float32)
        assert out is not None
        # 1.0+10.0=11.0, 2.0+20.0=22.0
        assert out["post_mlp"][0].dtype == np.float32
        assert out["post_mlp"][0].tolist() == [11.0, 22.0]

    def test_pack_effective_steering_handles_none(self):
        assert pack_effective_steering(None, None, np.float32) is None
        assert pack_effective_steering({}, {}, np.float32) is None

    def test_pack_dtype_fp16_loses_some_precision_but_preserves_shape(self):
        spec = {"post_mlp": {0: list(range(16))}}
        out = pack_steering_for_dtype(spec, np.float16)
        assert out is not None
        arr = out["post_mlp"][0]
        assert arr.dtype == np.float16
        assert arr.shape == (16,)
        # fp16 represents small ints exactly.
        assert arr.tolist() == list(float(i) for i in range(16))


# ---------------------------------------------------------------------------
# maybe_pack_inline_steering_for_request
# ---------------------------------------------------------------------------


class TestMaybePack:
    def test_no_steering_is_noop(self):
        sp = SamplingParams(max_tokens=1)
        maybe_pack_inline_steering_for_request(sp, torch.float32)
        assert sp._effective_prefill_steering_packed is None
        assert sp._effective_decode_steering_packed is None

    def test_named_only_is_noop(self):
        sp = SamplingParams(max_tokens=1, steering_module_ref=("m", 1.0))
        maybe_pack_inline_steering_for_request(sp, torch.float32)
        assert sp._effective_prefill_steering_packed is None
        assert sp._effective_decode_steering_packed is None
        assert sp.steering_module_ref == ("m", 1.0)

    def test_inline_packs_and_clears_originals(self):
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0]}},
        )
        maybe_pack_inline_steering_for_request(sp, torch.float32)
        assert sp.steering_vectors is None
        assert sp.prefill_steering_vectors is None
        assert sp.decode_steering_vectors is None
        assert sp._effective_prefill_steering_packed is not None
        assert sp._effective_decode_steering_packed is not None
        # Both phases resolve to the same result when only base is set.
        assert sp._effective_prefill_steering_packed["post_mlp"][0].tolist() == [
            1.0,
            2.0,
        ]
        assert sp._effective_decode_steering_packed["post_mlp"][0].tolist() == [1.0, 2.0]

    def test_phase_specific_resolves_per_phase(self):
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0]}},
            prefill_steering_vectors={"post_mlp": {0: [10.0, 20.0]}},
            decode_steering_vectors={"post_mlp": {0: [100.0, 200.0]}},
        )
        maybe_pack_inline_steering_for_request(sp, torch.float32)
        assert sp._effective_prefill_steering_packed["post_mlp"][0].tolist() == [
            11.0,
            22.0,
        ]
        assert sp._effective_decode_steering_packed["post_mlp"][0].tolist() == [
            101.0,
            202.0,
        ]

    def test_idempotent_when_already_packed(self):
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0]}},
        )
        maybe_pack_inline_steering_for_request(sp, torch.float32)
        first = sp._effective_prefill_steering_packed
        maybe_pack_inline_steering_for_request(sp, torch.float64)
        # Second call should bail immediately and not re-pack.
        assert sp._effective_prefill_steering_packed is first

    def test_effective_steering_returns_packed_after_pack(self):
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0]}},
        )
        maybe_pack_inline_steering_for_request(sp, torch.float32)
        # The cached_property fallback should now return packed values.
        assert sp.effective_prefill_steering is not None
        assert sp.effective_prefill_steering["post_mlp"][0].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Hash determinism across packing
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    def test_packed_request_hash_matches_unpacked(self):
        """A packed and unpacked submission of the same logical request
        must produce the same prefix-cache hash."""
        vectors = {"post_mlp": {0: [1.0, 2.0, 3.0]}}
        sp_unpacked = SamplingParams(max_tokens=1, steering_vectors=vectors)
        unpacked_hash = sp_unpacked.prefill_steering_config_hash

        sp_packed = SamplingParams(max_tokens=1, steering_vectors=vectors)
        maybe_pack_inline_steering_for_request(sp_packed, torch.float32)
        packed_hash = sp_packed.prefill_steering_config_hash

        assert packed_hash == unpacked_hash

    def test_different_vectors_different_hash(self):
        sp_a = SamplingParams(
            max_tokens=1, steering_vectors={"post_mlp": {0: [1.0, 2.0]}}
        )
        sp_b = SamplingParams(
            max_tokens=1, steering_vectors={"post_mlp": {0: [1.0, 3.0]}}
        )
        maybe_pack_inline_steering_for_request(sp_a, torch.float32)
        maybe_pack_inline_steering_for_request(sp_b, torch.float32)
        assert sp_a.prefill_steering_config_hash != sp_b.prefill_steering_config_hash


# ---------------------------------------------------------------------------
# msgspec round-trip
# ---------------------------------------------------------------------------


class TestMsgspecRoundtrip:
    def test_packed_field_round_trips_through_msgspec(self):
        """Packed ndarrays survive msgspec encode/decode with dtype + values."""
        sp_in = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0, 3.0]}},
        )
        maybe_pack_inline_steering_for_request(sp_in, torch.float32)
        assert sp_in._effective_prefill_steering_packed is not None

        enc = MsgpackEncoder()
        bufs = enc.encode(sp_in)
        dec = MsgpackDecoder(SamplingParams)
        sp_out = dec.decode(bufs)

        assert sp_out._effective_prefill_steering_packed is not None
        out_arr = sp_out._effective_prefill_steering_packed["post_mlp"][0]
        in_arr = sp_in._effective_prefill_steering_packed["post_mlp"][0]
        assert isinstance(out_arr, np.ndarray)
        assert out_arr.dtype == in_arr.dtype
        assert np.array_equal(out_arr, in_arr)

    def test_packed_payload_smaller_than_unpacked(self):
        """Sanity: the packed wire form is smaller than the unpacked one."""
        vectors = {"post_mlp": {i: [float(j) for j in range(2560)] for i in range(34)}}
        sp_unpacked = SamplingParams(max_tokens=1, steering_vectors=vectors)
        sp_packed = SamplingParams(max_tokens=1, steering_vectors=vectors)
        maybe_pack_inline_steering_for_request(sp_packed, torch.float32)

        enc = MsgpackEncoder()
        unpacked_bytes = sum(len(b) for b in enc.encode(sp_unpacked))
        packed_bytes = sum(len(b) for b in enc.encode(sp_packed))
        # 2560 floats × 34 layers × ~9 B/float ≈ 783 K msgpack;
        # 2560 floats × 34 layers × 4 B/float ≈ 348 K packed fp32
        # Expect at least 2× reduction.
        assert packed_bytes * 2 < unpacked_bytes, (
            f"packed={packed_bytes} not < unpacked={unpacked_bytes} / 2"
        )
