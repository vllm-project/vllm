# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import subprocess
import sys

# Some models don't have chat template or cannot be accessed
MODELS = [
    "rhymes-ai/Aria",
    "CohereForAI/aya-vision-8b",
    "Salesforce/blip2-opt-6.7b",
    "facebook/chameleon-7b",
    "deepseek-ai/deepseek-vl2-tiny",
    "microsoft/Florence-2-base",
    "adept/fuyu-8b",
    "google/gemma-3-4b-it",
    "THUDM/glm-4v-9b",
    "ibm-granite/granite-speech-3.3-8b",
    "h2oai/h2ovl-mississippi-800m",
    "OpenGVLab/InternVL2-1B",
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "moonshotai/Kimi-VL-A3B-Instruct",
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "TIGER-Lab/Mantis-8B-siglip-llama3",
    "openbmb/MiniCPM-Llama3-V-2_5",
    "openbmb/MiniCPM-o-2_6",
    "openbmb/MiniCPM-V-2_6",
    # "MiniMaxAI/MiniMax-VL-01",
    # "allenai/Molmo-7B-D-0924",
    # "allenai/Molmo-7B-O-0924",
    "nvidia/NVLM-D-72B",
    "AIDC-AI/Ovis2-1B",
    "google/paligemma-3b-mix-224",
    "google/paligemma2-3b-ft-docci-448",
    "microsoft/Phi-3.5-vision-instruct",
    "microsoft/Phi-4-multimodal-instruct",
    # "mistralai/Pixtral-12B-2409",
    "mistral-community/pixtral-12b",
    "Qwen/Qwen-VL-Chat",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-Audio-7B-Instruct",
    "Qwen/Qwen2.5-Omni-7B",
    "Skywork/Skywork-R1V-38B",
    "fixie-ai/ultravox-v0_5-llama-3_2-1b",
    # "openai/whisper-large-v3",
]


def main(output_dir: str, sync: bool):
    for model in MODELS:
        for parallel_backend in ("uni", "mt", "mp"):
            args = [
                sys.executable,
                "benchmark_processing.py",
                "-m",
                model,
                "-p",
                parallel_backend,
                "-o",
                output_dir,
            ]
            if sync:
                args.extend(["--async"])
            args.extend(["--append"])

            try:
                subprocess.run(args, timeout=5 * 60, check=True)
            except Exception as e:
                print(f"Failed to benchmark {model}:\n{e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark parallel multi-modal processing for all models")
    parser.add_argument("-o",
                        "--output-dir",
                        type=str,
                        required=True,
                        help="Directory to save the results")
    parser.add_argument("--sync",
                        action="store_true",
                        help="Test the sync engine instead of async engine")

    args = parser.parse_args()

    main(args.output_dir, args.sync)
