# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from huggingface_hub import snapshot_download
from runai_model_streamer.safetensors_streamer.streamer_mock import StreamerPatcher

from vllm import SamplingParams

load_format = "runai_streamer"
test_model = "openai-community/gpt2"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)


def test_runai_model_loader_download_files_s3_mocked_with_patch(
    vllm_runner,
    tmp_path: Path,
    monkeypatch,
):
    patcher = StreamerPatcher(str(tmp_path))

    test_mock_s3_model = "s3://my-mock-bucket/gpt2/"

    # Download model from HF
    mock_model_dir = f"{tmp_path}/gpt2"
    snapshot_download(repo_id=test_model, local_dir=mock_model_dir)

    class MockFilesModule:
        def glob(self, path: str, allow_pattern=None, credentials=None):
            return patcher.shim_list_safetensors(path, s3_credentials=credentials)

        def pull_files(
            self,
            model_path,
            dst,
            allow_pattern=None,
            ignore_pattern=None,
            credentials=None,
        ):
            return patcher.shim_pull_files(
                model_path,
                dst,
                allow_pattern,
                ignore_pattern,
                s3_credentials=credentials,
            )
    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_list_safetensors",
        patcher.shim_list_safetensors,
    )
    def mock_get_s3_files_module():
        return MockFilesModule()

    monkeypatch.setattr(
        "runai_model_streamer.s3_utils.s3_utils.get_s3_files_module",
        mock_get_s3_files_module,
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.runai_utils.runai_pull_safetensors",
        patcher.shim_pull_safetensors,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.runai_streamer_loader.list_safetensors",
        patcher.shim_list_safetensors,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.weight_utils.SafetensorsStreamer",
        patcher.create_mock_streamer,
    )

    with vllm_runner(test_mock_s3_model, load_format=load_format) as llm:
        deserialized_outputs = llm.generate(prompts, sampling_params)
        assert deserialized_outputs
