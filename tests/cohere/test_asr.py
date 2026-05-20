# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path

import torch

from tests.entrypoints.openai.correctness.test_transcription_api_correctness import (
    load_hf_dataset,
    run_evaluation,
)
from tests.models.registry import HF_EXAMPLE_MODELS
from tests.utils import RemoteOpenAIServer

ASR_MODEL_NAME = "CohereLabs/cohere-transcribe-03-2026"
ASR_MODEL_DIR_NAME = "cohere-transcribe-03-2026"
ASR_EXPECTED_WER = 11.92
ASR_DATASET_REPO = "D4nt3/esb-datasets-earnings22-validation-tiny-filtered"


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
        # "--enforce-eager",
        # f"--tokenizer_mode={model_info.tokenizer_mode}",
        f"--served-model-name={ASR_MODEL_NAME}",
    ]
    if model_info.trust_remote_code:
        server_args.append("--trust-remote-code")

    with RemoteOpenAIServer(server_model, server_args) as remote_server:
        dataset = load_hf_dataset(ASR_DATASET_REPO)
        client = remote_server.get_async_client()
        wer = run_evaluation(
            ASR_MODEL_NAME,
            client,
            dataset,
            max_concurrent_reqs=len(dataset),
        )

    print(f"Expected WER: {ASR_EXPECTED_WER}, Actual WER: {wer}")
    torch.testing.assert_close(wer, ASR_EXPECTED_WER, atol=1e-1, rtol=1e-2)
