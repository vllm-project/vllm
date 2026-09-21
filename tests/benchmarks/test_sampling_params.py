# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.benchmarks.datasets.utils import get_sampling_params
from vllm.tokenizers import TokenizerLike


class _FakeTokenizer(TokenizerLike):
    """Minimal tokenizer implementing the TokenizerLike protocol
    for testing get_sampling_params."""

    def __init__(self, vocab_size: int = 1000, num_special_tokens: int = 0) -> None:
        self._vocab_size = vocab_size
        self._num_special_tokens = num_special_tokens

    # -- Properties required by TokenizerLike --

    @classmethod
    def from_pretrained(cls, path_or_repo_id, *a, **kw):  # type: ignore[override]
        return cls()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def all_special_tokens(self) -> list[str]:
        return []

    @property
    def all_special_ids(self) -> list[int]:
        return []

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def pad_token_id(self) -> int:
        return 2

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def max_token_id(self) -> int:
        return self._vocab_size - 1

    @property
    def max_chars_per_token(self) -> int:
        return 4

    @property
    def truncation_side(self) -> str:
        return "right"

    def num_special_tokens_to_add(self) -> int:
        return self._num_special_tokens

    def __call__(self, text, text_pair=None, **kw):  # type: ignore[override]
        raise NotImplementedError

    def get_vocab(self) -> dict[str, int]:
        return {}

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text, **kw) -> list[int]:  # type: ignore[override]
        raise NotImplementedError

    def apply_chat_template(self, messages, **kw):  # type: ignore[override]
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):  # type: ignore[override]
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def decode(self, ids, skip_special_tokens: bool = False) -> str:  # type: ignore[override]
        raise NotImplementedError

    def convert_ids_to_tokens(  # type: ignore[override]
        self, ids, skip_special_tokens: bool = False
    ) -> list[str]:
        raise NotImplementedError


class TestGetSamplingParams:
    """Tests for ``get_sampling_params`` in ``vllm.benchmarks.datasets.shared``."""

    # -- helpers --

    @staticmethod
    def _tok(vocab_size: int = 1000, num_special: int = 0) -> _FakeTokenizer:
        return _FakeTokenizer(vocab_size=vocab_size, num_special_tokens=num_special)

    # -- return shape / dtype --

    def test_returns_three_arrays(self):
        rng = np.random.default_rng(0)
        result = get_sampling_params(rng, 5, 0.0, 100, 50, self._tok())
        assert len(result) == 3
        for arr in result:
            assert isinstance(arr, np.ndarray)

    @pytest.mark.parametrize("n", [1, 10, 100])
    def test_output_length_matches_num_requests(self, n: int):
        rng = np.random.default_rng(42)
        input_lens, output_lens, offsets = get_sampling_params(
            rng, n, 0.0, 64, 32, self._tok()
        )
        assert input_lens.shape == (n,)
        assert output_lens.shape == (n,)
        assert offsets.shape == (n,)

    # -- fixed lengths (range_ratio = 0) --

    def test_zero_range_ratio_gives_constant_lengths(self):
        rng = np.random.default_rng(7)
        input_lens, output_lens, _ = get_sampling_params(
            rng, 20, 0.0, 128, 64, self._tok()
        )
        assert np.all(input_lens == 128)
        assert np.all(output_lens == 64)

    def test_special_tokens_subtracted_from_input_only(self):
        rng = np.random.default_rng(7)
        input_lens, output_lens, _ = get_sampling_params(
            rng, 10, 0.0, 100, 50, self._tok(num_special=4)
        )
        # real_input_len = 100 - 4 = 96, range_ratio 0 → all 96
        assert np.all(input_lens == 96)
        # special tokens are not subtracted from output length
        assert np.all(output_lens == 50)

    # -- range ratios --

    def test_input_range_bounds(self):
        rng = np.random.default_rng(0)
        ratio = 0.5
        base = 200
        input_lens, _, _ = get_sampling_params(
            rng, 500, {"input": ratio, "output": 0.0}, base, 50, self._tok()
        )
        lo = int(np.floor(base * (1 - ratio)))
        hi = int(np.ceil(base * (1 + ratio)))
        assert np.all(input_lens >= lo)
        assert np.all(input_lens <= hi)

    def test_output_range_bounds(self):
        rng = np.random.default_rng(0)
        ratio = 0.3
        base = 100
        _, output_lens, _ = get_sampling_params(
            rng, 500, {"input": 0.0, "output": ratio}, 50, base, self._tok()
        )
        lo = max(1, int(np.floor(base * (1 - ratio))))
        hi = int(np.ceil(base * (1 + ratio)))
        assert np.all(output_lens >= lo)
        assert np.all(output_lens <= hi)

    def test_output_low_clamped_to_one(self):
        """Even with a high ratio that would push output_low to 0,
        the function clamps it to 1."""
        rng = np.random.default_rng(0)
        # output_len=1, ratio=0.99 → floor(1*0.01)=0, should clamp to 1
        _, output_lens, _ = get_sampling_params(
            rng, 50, {"input": 0.0, "output": 0.99}, 100, 1, self._tok()
        )
        assert np.all(output_lens >= 1)

    # -- offsets bounded by vocab_size --

    @pytest.mark.parametrize("vocab", [100, 32000, 128256])
    def test_offsets_within_vocab(self, vocab: int):
        rng = np.random.default_rng(0)
        _, _, offsets = get_sampling_params(
            rng, 200, 0.0, 64, 32, self._tok(vocab_size=vocab)
        )
        assert np.all(offsets >= 0)
        assert np.all(offsets < vocab)

    # -- reproducibility --

    def test_same_seed_same_results(self):
        tok = self._tok()
        rr = {"input": 0.3, "output": 0.2}
        a = get_sampling_params(np.random.default_rng(42), 50, rr, 256, 64, tok)
        b = get_sampling_params(np.random.default_rng(42), 50, rr, 256, 64, tok)
        for arr_a, arr_b in zip(a, b):
            np.testing.assert_array_equal(arr_a, arr_b)

    def test_different_seed_different_results(self):
        tok = self._tok()
        rr = {"input": 0.3, "output": 0.2}
        a = get_sampling_params(np.random.default_rng(0), 50, rr, 256, 64, tok)
        b = get_sampling_params(np.random.default_rng(1), 50, rr, 256, 64, tok)
        # Extremely unlikely all three arrays match with different seeds
        assert not all(np.array_equal(arr_a, arr_b) for arr_a, arr_b in zip(a, b))

    # -- validation / error paths --

    @pytest.mark.parametrize("bad_ratio", [-0.1, 1.0, 1.5])
    def test_invalid_input_range_ratio(self, bad_ratio: float):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="input_range_ratio"):
            get_sampling_params(
                rng, 10, {"input": bad_ratio, "output": 0.0}, 100, 50, self._tok()
            )

    @pytest.mark.parametrize("bad_ratio", [-0.1, 1.0, 1.5])
    def test_invalid_output_range_ratio(self, bad_ratio: float):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="output_range_ratio"):
            get_sampling_params(
                rng, 10, {"input": 0.0, "output": bad_ratio}, 100, 50, self._tok()
            )

    def test_invalid_dict_missing_keys(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="input.*output"):
            get_sampling_params(rng, 10, {"input": 0.1}, 100, 50, self._tok())

    def test_input_len_zero_with_special_tokens(self):
        """input_len < num_special_tokens → real_input_len = 0, which is fine
        (range [0, 0])."""
        rng = np.random.default_rng(0)
        input_lens, _, _ = get_sampling_params(
            rng, 5, 0.0, 5, 50, self._tok(num_special=10)
        )
        # real_input_len = max(0, 5 - 10) = 0
        assert np.all(input_lens == 0)

    # -- edge cases --

    def test_single_request(self):
        rng = np.random.default_rng(0)
        i, o, off = get_sampling_params(rng, 1, 0.0, 100, 50, self._tok())
        assert i.shape == (1,)
        assert o.shape == (1,)
        assert off.shape == (1,)

    def test_large_num_requests(self):
        rng = np.random.default_rng(0)
        i, o, off = get_sampling_params(rng, 10_000, 0.5, 512, 128, self._tok())
        assert i.shape == (10_000,)
        assert o.shape == (10_000,)
        assert off.shape == (10_000,)
