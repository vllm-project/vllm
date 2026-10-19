# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU check of the Whisper log-mel formula against Hugging Face.

Loads the formula module by path so this file does not import vLLM.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperFeatureExtractor


def _load_whisper_log_mel():
    path = (
        Path(__file__).resolve().parents[4]
        / "vllm/model_executor/models/whisper_logmel.py"
    )
    if not path.is_file():
        from vllm.model_executor.models.whisper_logmel import (
            whisper_log_mel_spectrogram,
        )

        return whisper_log_mel_spectrogram

    spec = importlib.util.spec_from_file_location("whisper_logmel_formula", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.whisper_log_mel_spectrogram


def _extract(fn, waveform, extractor: WhisperFeatureExtractor, *, device=None):
    return fn(
        waveform,
        n_fft=extractor.n_fft,
        hop_length=extractor.hop_length,
        mel_filters=torch.from_numpy(np.asarray(extractor.mel_filters)),
        n_samples=extractor.n_samples,
        padding_value=float(extractor.padding_value),
        dither=float(extractor.dither),
        device=device,
    )


def test_log_mel_matches_cpu_extractor() -> None:
    """Short, exact, and over-long clips agree with HF to about 1e-5.

    Two clips in one batch each use their own max. Nonzero dither uses one
    shared noise draw. Comparison is a tolerance, not bit equality: GPU
    ``torch.stft`` is not bit-identical to CPU.
    """
    fn = _load_whisper_log_mel()
    extractor = WhisperFeatureExtractor()
    sr = extractor.sampling_rate
    rng = np.random.default_rng(0)
    frames = extractor.n_samples // extractor.hop_length

    for n_samples in (sr, extractor.n_samples, extractor.n_samples + sr):
        wave = rng.standard_normal(n_samples).astype(np.float32)
        ref = extractor(wave, sampling_rate=sr, return_tensors="pt")["input_features"]
        got = _extract(fn, torch.from_numpy(wave), extractor)
        assert got.shape[-1] == frames
        assert got.shape[-2] == extractor.feature_size
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)

    prefix = rng.standard_normal(extractor.n_samples).astype(np.float32)
    long_a = np.concatenate([prefix, rng.standard_normal(128).astype(np.float32)])
    long_b = np.concatenate([prefix, np.ones(64, dtype=np.float32)])
    torch.testing.assert_close(
        _extract(fn, torch.from_numpy(long_a), extractor),
        _extract(fn, torch.from_numpy(long_b), extractor),
        atol=1e-5,
        rtol=1e-5,
    )

    waves = [
        rng.standard_normal(sr // 2).astype(np.float32),
        (rng.standard_normal(3 * sr) * 4).astype(np.float32),
    ]
    ref = extractor(waves, sampling_rate=sr, return_tensors="pt")["input_features"]
    got = _extract(fn, [torch.from_numpy(wave) for wave in waves], extractor)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    for index, wave in enumerate(waves):
        alone = extractor(wave, sampling_rate=sr, return_tensors="pt")["input_features"]
        torch.testing.assert_close(got[index], alone[0], atol=1e-5, rtol=1e-5)

    padded = np.stack(
        [
            wave
            if len(wave) == extractor.n_samples
            else (
                wave[: extractor.n_samples]
                if len(wave) > extractor.n_samples
                else np.pad(wave, (0, extractor.n_samples - len(wave)))
            )
            for wave in waves
        ]
    ).astype(np.float32)
    direct = torch.from_numpy(
        np.asarray(extractor._torch_extract_fbank_features(padded, device="cpu"))
    )
    torch.testing.assert_close(got, direct, atol=1e-5, rtol=1e-5)
    # GPU compare uses this undithered batch. Dither is restored before it.
    cpu_ref = got.clone()

    extractor.dither = 1e-3
    real_randn = torch.randn

    def _shared_noise(shape, dtype=None, device=None, **kwargs):
        return torch.full(shape, 0.25, dtype=dtype, device=device)

    torch.randn = _shared_noise
    try:
        ref = extractor(waves, sampling_rate=sr, return_tensors="pt")["input_features"]
        got = _extract(fn, [torch.from_numpy(wave) for wave in waves], extractor)
        direct = torch.from_numpy(
            np.asarray(extractor._torch_extract_fbank_features(padded, device="cpu"))
        )
    finally:
        torch.randn = real_randn
        extractor.dither = 0.0
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got, direct, atol=1e-5, rtol=1e-5)

    if torch.cuda.is_available():
        got_gpu = _extract(
            fn,
            [torch.from_numpy(wave) for wave in waves],
            extractor,
            device="cuda",
        )
        torch.testing.assert_close(got_gpu.cpu(), cpu_ref, atol=1e-5, rtol=1e-5)
