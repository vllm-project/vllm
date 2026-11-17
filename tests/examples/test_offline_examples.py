# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the offline example scripts, one test per script invocation.

Each test runs one example in its own process from the examples directory,
with the arguments the Examples CI step has always used. As one test per
example, CI can time each one and spread them over several jobs.
"""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

# The examples run in their own processes; this process starts no engine.
pytestmark = pytest.mark.skip_global_cleanup

EXAMPLES_DIR = Path(__file__).parents[2] / "examples"

TENSORIZER_DIR = "/tmp/"
TENSORIZER_TENSORS = "/tmp/vllm/facebook/opt-125m/v1/model.tensors"

# Each case is the list of commands one test runs, in order.
EXAMPLES = [
    pytest.param(["basic/offline_inference/chat.py"], id="basic-chat"),
    pytest.param(
        ["basic/offline_inference/generate.py --model facebook/opt-125m"],
        id="basic-generate-opt-125m",
    ),
    pytest.param(
        [
            (
                "basic/offline_inference/generate.py"
                " --model meta-llama/Llama-2-13b-chat-hf --cpu-offload-gb 10"
            )
        ],
        id="basic-generate-llama2-13b-cpu-offload",
    ),
    pytest.param(["basic/offline_inference/classify.py"], id="basic-classify"),
    pytest.param(["basic/offline_inference/embed.py"], id="basic-embed"),
    pytest.param(["basic/offline_inference/score.py"], id="basic-score"),
    # Multi-modal models.
    pytest.param(
        ["generate/multimodal/audio_language_offline.py --seed 0"],
        id="multimodal-audio-language",
    ),
    pytest.param(
        ["generate/multimodal/vision_language_offline.py --seed 0"],
        id="multimodal-vision-language",
    ),
    pytest.param(
        ["generate/multimodal/vision_language_multi_image_offline.py --seed 0"],
        id="multimodal-vision-language-multi-image",
    ),
    pytest.param(
        [
            (
                "generate/multimodal/encoder_decoder_multimodal_offline.py"
                " --model-type whisper --seed 0"
            )
        ],
        id="multimodal-encoder-decoder-whisper",
    ),
    # Pooling models.
    pytest.param(
        ["pooling/embed/vision_embedding_offline.py --seed 0"],
        id="pooling-vision-embedding",
    ),
    # Feature demos.
    pytest.param(
        ["features/automatic_prefix_caching/prefix_caching_offline.py"],
        id="features-prefix-caching",
    ),
    pytest.param(["deployment/llm_engine_example.py"], id="deployment-llm-engine"),
    # Serialize then deserialize, so both commands stay in one test.
    pytest.param(
        [
            (
                "features/tensorize_vllm_model.py --model facebook/opt-125m serialize"
                f" --serialized-directory {TENSORIZER_DIR} --suffix v1"
            ),
            (
                "features/tensorize_vllm_model.py --model facebook/opt-125m"
                f" deserialize --path-to-tensors {TENSORIZER_TENSORS}"
            ),
        ],
        id="features-tensorize-opt-125m",
    ),
    pytest.param(
        [
            (
                "features/speculative_decoding/spec_decode_offline.py --test"
                " --method eagle --num_spec_tokens 3 --dataset-name hf"
                " --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0"
                " --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill"
                " --max-model-len 2048"
            )
        ],
        id="features-spec-decode-eagle",
    ),
    # vllm#26682 uses slightly more memory in PyTorch 2.9+, which made this
    # test OOM on a 1xL4 GPU.
    pytest.param(
        [
            (
                "features/speculative_decoding/spec_decode_offline.py --test"
                " --method eagle3 --num_spec_tokens 3 --dataset-name hf"
                " --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0"
                " --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill"
                " --max-model-len 1536"
            )
        ],
        id="features-spec-decode-eagle3",
    ),
]


@pytest.mark.parametrize("commands", EXAMPLES)
def test_offline_example(commands: list[str]):
    """Run one example's commands from the examples directory.

    Args:
        commands: Each one is an example script path, relative to the
            examples directory, followed by its arguments.

    """
    for command in commands:
        subprocess.run(
            [sys.executable, *shlex.split(command)],
            cwd=EXAMPLES_DIR,
            check=True,
        )
