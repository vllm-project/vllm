# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path

import torch

from tests.entrypoints.openai.correctness.test_transcription_api_correctness import (
    LONGFORM_NUM_SAMPLES,
    load_longform_dataset,
    load_shortform_eval_dataset,
    run_evaluation,
    run_longform_evaluation,
)
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import RemoteOpenAIServer

ASR_MODEL_NAME = "CohereLabs/cohere-transcribe-03-2026"
ASR_MODEL_DIR_NAME = "cohere-transcribe-03-2026"
ASR_EXPECTED_WER = 11.92
ASR_DATASET_REPO = "D4nt3/esb-datasets-earnings22-validation-tiny-filtered"
# Keep the long-form regression on par with the calibrated short-form gate
# until a dedicated long-form baseline is measured for this model.
ASR_LONGFORM_EXPECTED_WER = 7.5


def _get_server_model() -> str:
    # CI pre-downloads the checkpoint under ENGINES_DIR; use it when available
    # so the test can run offline, otherwise fall back to the HF model id.
    engines_dir = os.environ.get("ENGINES_DIR", "/root/engines")
    local_model_dir = Path(engines_dir) / ASR_MODEL_DIR_NAME
    if local_model_dir.is_dir():
        return str(local_model_dir)
    return ASR_MODEL_NAME


def test_cohere_transcribe_wer_correctness():
    model_info = HF_EXAMPLE_MODELS.find_hf_info(ASR_MODEL_NAME)
    server_model = _get_server_model()
    server_args = [
        f"--served-model-name={ASR_MODEL_NAME}",
    ]
    if model_info.trust_remote_code:
        server_args.append("--trust-remote-code")

    with RemoteOpenAIServer(server_model, server_args) as remote_server:
        dataset = load_shortform_eval_dataset(ASR_DATASET_REPO)
        client = remote_server.get_async_client()
        wer = run_evaluation(
            ASR_MODEL_NAME,
            client,
            dataset,
            max_concurrent_reqs=len(dataset),
        )

    print(f"Expected WER: {ASR_EXPECTED_WER}, Actual WER: {wer}")
    torch.testing.assert_close(wer, ASR_EXPECTED_WER, atol=1e-1, rtol=1e-2)


def test_cohere_transcribe_long_audio_wer_correctness():
    model_info = HF_EXAMPLE_MODELS.find_hf_info(ASR_MODEL_NAME)
    server_model = _get_server_model()
    server_args = [
        f"--served-model-name={ASR_MODEL_NAME}",
    ]
    if model_info.trust_remote_code:
        server_args.append("--trust-remote-code")

    # add this after next upgrade of vllm
    # env_dict = {
    #     "VLLM_MAX_AUDIO_DECODE_DURATION_S": "1800",
    # }

    with RemoteOpenAIServer(
        server_model,
        server_args,
        # env_dict=env_dict,
    ) as remote_server:
        dataset = load_longform_dataset()
        client = remote_server.get_async_client()
        wer = run_longform_evaluation(
            model=ASR_MODEL_NAME,
            client=client,
            dataset=dataset,
            max_concurrent_reqs=LONGFORM_NUM_SAMPLES,
        )

    print(f"Expected WER: {ASR_LONGFORM_EXPECTED_WER}, Actual WER: {wer}")
    torch.testing.assert_close(wer, ASR_LONGFORM_EXPECTED_WER, atol=1e-1, rtol=1e-2)
