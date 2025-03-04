# SPDX-License-Identifier: Apache-2.0
# Implements a simple offline inference script for the Phi 3.5 Speech model.
# Code implemented by Jacob Platin (jacobplatin@microsoft.com)

import soundfile

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser


def main_pure_text(args: dict) -> None:
    """
    Main function for the offline inference script.
    """
    llm = LLM(model=args.model_path,
              trust_remote_code=True,
              enforce_eager=True)
    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    prompt_suffix = '<|end|>\n'
    prompt = f'{user_prompt}what is the answer for 1+1? Explain'\
             f' it.{prompt_suffix}{assistant_prompt}'
    print(f'>>> Prompt\n{prompt}')
    # NOTE: soundfile.read will return the audio feature and the sampling rate
    generate_args = {"prompt": prompt}
    # NOTE: you should use the following settings to ensure parity in HF
    # generate_ids = model.generate(
    #     **inputs,
    #     top_p=1,
    #     max_new_tokens=1200,
    #     temperature=0,
    #     use_cache=False,
    #     min_p=0,
    #     top_k=-1,
    # )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1200,
    )

    outputs = llm.generate(generate_args, sampling_params=sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}\n\n")


def main_with_lora_speech(args: dict, activate_lora_request=True) -> None:
    """
    Main function for the offline inference script.
    """
    wav_paths = [args.wav_path]
    llm = LLM(model=args.model_path,
              trust_remote_code=True,
              enable_lora=activate_lora_request,
              enforce_eager=True,
              max_lora_rank=512,
              lora_extra_vocab_size=0,
              limit_mm_per_prompt={"audio": len(wav_paths)},
              max_loras=5)

    # assert len(wav_paths) == 1, "Only support single audio files for now!"

    prompt = "Generate a comprehensive text transcription of the "\
        "spoken content."
    placeholders = "\n".join(f"<|audio_{i}|>"
                             for i in range(1,
                                            len(wav_paths) + 1))
    prompt = f"<|user|>\n{placeholders}\n{prompt}<|end|>\n<|assistant|>\n"

    # NOTE: soundfile.read will return the audio feature and the sampling rate
    generate_args = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": [soundfile.read(wav_path) for wav_path in wav_paths]
        }
    }
    # NOTE: you should use the following settings to ensure parity in HF
    # generate_ids = model.generate(
    #     **inputs,
    #     top_p=1,
    #     max_new_tokens=1200,
    #     temperature=0,
    #     use_cache=False,
    #     min_p=0,
    #     top_k=-1,
    # )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=200,
    )

    outputs = llm.generate(
        generate_args,
        sampling_params=sampling_params,
        lora_request=[LoRARequest("speech_adapter", 3, args.speech_lora_path)]
        if activate_lora_request else None)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}\n\n")


def main_with_lora_speech_batch(args: dict,
                                activate_lora_request=True) -> None:
    """
    Main function for the offline inference script.
    """
    wav_paths = [args.wav_path, args.wav_path]

    llm = LLM(model=args.model_path,
              trust_remote_code=True,
              enable_lora=activate_lora_request,
              enforce_eager=True,
              max_lora_rank=512,
              lora_extra_vocab_size=0,
              limit_mm_per_prompt={"audio": len(wav_paths)},
              max_loras=5)

    # assert len(wav_paths) == 1, "Only support single audio files for now!"

    prompt = "Based on the attached audio, generate a comprehensive text "\
        "transcription of the spoken content."
    placeholders = "\n".join(f"<|audio_{i}|>"
                             for i in range(1,
                                            len(wav_paths) + 1))
    prompt = f"<|user|>\n{placeholders}\n{prompt}<|end|>\n<|assistant|>\n"

    # NOTE: soundfile.read will return the audio feature and the sampling rate
    generate_args = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [soundfile.read(wav_path) for wav_path in wav_paths]
            }
        },
        {
            "prompt": prompt,
            "multi_modal_data": {
                "audio": [soundfile.read(wav_path) for wav_path in wav_paths]
            }
        },
    ]
    # NOTE: you should use the following settings to ensure parity in HF
    # generate_ids = model.generate(
    #     **inputs,
    #     top_p=1,
    #     max_new_tokens=1200,
    #     temperature=0,
    #     use_cache=False,
    #     min_p=0,
    #     top_k=-1,
    # )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1200,
    )

    outputs = llm.generate(
        generate_args,
        sampling_params=sampling_params,
        lora_request=LoRARequest("speech_adapter", 3, args.speech_lora_path)
        if activate_lora_request else None)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}\n\n")


def main_with_lora_vision(args: dict, activate_lora_request=True) -> None:
    """
    Main function for the offline inference script.
    """
    image_urls = [args.image_url]
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        enable_lora=activate_lora_request,
        enforce_eager=True,
        max_lora_rank=512,
        lora_extra_vocab_size=0,
        max_loras=5,
        # max_model_len=4096,
        # max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    # prompt = "what's the traffic sign in the image"
    prompt = "What is shown in this image?"

    placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>\n{placeholders}\n{prompt}<|end|>\n<|assistant|>\n"

    image_data = [fetch_image(url) for url in image_urls]

    # NOTE: soundfile.read will return the audio feature and the sampling rate
    generate_args = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image_data,
        },
    }
    # NOTE: you should use the following settings to ensure parity in HF
    # generate_ids = model.generate(
    #     **inputs,
    #     top_p=1,
    #     max_new_tokens=1200,
    #     temperature=0,
    #     use_cache=False,
    #     min_p=0,
    #     top_k=-1,
    # )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1200,
    )

    outputs = llm.generate(
        generate_args,
        sampling_params=sampling_params,
        lora_request=[LoRARequest("vision_adapter", 3, args.vision_lora_path)]
        if activate_lora_request else None)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}\n\n")


def main_with_lora_vision_batch(args: dict,
                                activate_lora_request=True) -> None:
    """
    Main function for the offline inference script.
    """
    image_urls = [
        args.image_url,
        "https://alinasayre.com/wp-content/uploads/2013/10/d67cd-dsc01646.jpg"
    ]
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        enable_lora=activate_lora_request,
        enforce_eager=True,
        max_lora_rank=512,
        lora_extra_vocab_size=0,
        max_loras=5,

        # max_model_len=4096,
        # max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
    )
    # prompt = "what's the traffic sign in the image"
    prompt = "What is shown in this image?"

    placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>\n{placeholders}\n{prompt}<|end|>\n<|assistant|>\n"

    # image_data=[fetch_image(url) for url in image_urls]

    # NOTE: soundfile.read will return the audio feature and the sampling rate
    generate_args = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [
                    fetch_image(url) for url in [
                        "https://www.ilankelman.org/stopsigns/australia.jpg",
                        "https://alinasayre.com/wp-content/uploads/2013/10/"\
                            "d67cd-dsc01646.jpg"
                    ]
                ],
            },
        },
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [fetch_image(url) for url in image_urls],
            },
        },
    ]
    # NOTE: you should use the following settings to ensure parity in HF
    # generate_ids = model.generate(
    #     **inputs,
    #     top_p=1,
    #     max_new_tokens=1200,
    #     temperature=0,
    #     use_cache=False,
    #     min_p=0,
    #     top_k=-1,
    # )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1200,
    )

    outputs = llm.generate(
        generate_args,
        sampling_params=sampling_params,
        lora_request=LoRARequest("vision_adapter", 3, args.vision_lora_path)
        if activate_lora_request else None)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}\n\n")


def main_with_lora_vision_speech(args: dict,
                                 activate_lora_request=True) -> None:
    """
    Main function for the offline inference script.
    """
    image_urls = [args.image_url]
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        enable_lora=activate_lora_request,
        enforce_eager=True,
        max_lora_rank=512,
        lora_extra_vocab_size=0,
        max_loras=5,

        # max_model_len=4096,
        # max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

    prompt = ""

    placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>\n{placeholders}\n<|audio_1|>\n{prompt}<|end|>"\
        "\n<|assistant|>\n"

    image_data = [fetch_image(url) for url in image_urls]

    wav_paths = [
        "/scratch/turing_westus3_prm_data/users/congcongchen/MoE_2/hf-models"\
            "/phio/examples/what_is_the_traffic_sign_in_the_image.wav"
    ]
    # NOTE: soundfile.read will return the audio feature and the sampling rate
    generate_args = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image_data,
            "audio": [soundfile.read(wav_path) for wav_path in wav_paths],
        },
    }
    # NOTE: you should use the following settings to ensure parity in HF
    # generate_ids = model.generate(
    #     **inputs,
    #     top_p=1,
    #     max_new_tokens=1200,
    #     temperature=0,
    #     use_cache=False,
    #     min_p=0,
    #     top_k=-1,
    # )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1200,
    )

    outputs = llm.generate(
        generate_args,
        sampling_params=sampling_params,
        lora_request=[LoRARequest("vision_adapter", 3, args.vision_lora_path)]
        if activate_lora_request else None)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}\n\n")


def main_with_lora_vision_speech_batch(args: dict,
                                       activate_lora_request=True) -> None:
    """
    Main function for the offline inference script.
    """
    image_urls = [
        args.image_url,
        "https://alinasayre.com/wp-content/uploads/2013/10/d67cd-dsc01646.jpg"
    ]
    wav_paths = [args.wav_path]
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        enable_lora=activate_lora_request,
        enforce_eager=True,
        max_lora_rank=512,
        lora_extra_vocab_size=0,
        max_loras=5,

        # max_model_len=40960,
        # max_num_seqs=5,
        limit_mm_per_prompt={
            "image": len(image_urls),
            "audio": len(wav_paths)
        },
    )

    prompt = "try your best to answer the question"

    placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|user|>\n{placeholders}\n<|audio_1|>\n{prompt}"\
        "<|end|>\n<|assistant|>\n"

    # image_data=[fetch_image(url) for url in image_urls]

    # NOTE: soundfile.read will return the audio feature and the sampling rate
    generate_args = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [fetch_image(url) for url in image_urls],
                "audio": [soundfile.read(wav_path) for wav_path in wav_paths],
            },
        },
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [
                    fetch_image(url) for url in [
                        "https://alinasayre.com/wp-content/uploads/"\
                            "2013/10/d67cd-dsc01646.jpg",
                        "https://alinasayre.com/wp-content/uploads/"\
                            "2012/01/c3a7c-dsc01668.jpg"
                    ]
                ],
                "audio": [soundfile.read(wav_path) for wav_path in wav_paths],
            },
        },
    ]
    # NOTE: you should use the following settings to ensure parity in HF
    # generate_ids = model.generate(
    #     **inputs,
    #     top_p=1,
    #     max_new_tokens=1200,
    #     temperature=0,
    #     use_cache=False,
    #     min_p=0,
    #     top_k=-1,
    # )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1200,
    )

    outputs = llm.generate(
        generate_args,
        sampling_params=sampling_params,
        lora_request=LoRARequest("vision_adapter", 3, args.vision_lora_path)
        if activate_lora_request else None)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}\n\n")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models that support multi-image input")
    parser.add_argument(
        "--model-path",
        "-p",
        type=str,
        default=
        "/scratch/turing_westus3_prm_data/users/congcongchen/phi4-mini-mm",
        help="Path to the (HuggingFace) model checkpoint.",
    )

    parser.add_argument(
        "--vision-lora-path",
        "-v",
        type=str,
        default=
        "/scratch/turing_westus3_prm_data/users/congcongchen/phi4-mini-mm/vision-lora",
        help="Path to the (HuggingFace) vision lora model checkpoint.",
    )

    parser.add_argument(
        "--speech-lora-path",
        "-s",
        type=str,
        default=
        "/scratch/turing_westus3_prm_data/users/congcongchen/phi4-mini-mm/speech-lora",
        help="Path to the (HuggingFace) speech lora model checkpoint.",
    )

    parser.add_argument(
        "--wav-path",
        "-w",
        type=str,
        default=
        "/scratch/turing_westus3_prm_data/users/congcongchen/30s_test_6.wav",
        help="Path to the audio file.",
    )

    parser.add_argument(
        "--image-url",
        "-i",
        type=str,
        default=
        "https://alinasayre.com/wp-content/uploads/2013/10/d67cd-dsc01646.jpg",
    )

    parser.add_argument(
        "--test-type",
        "-t",
        type=str,
        default="speech_language_with_lora",
    )

    args = parser.parse_args()
    ##### Language Only #####
    test_type = args.test_type
    if test_type == "language_only":
        main_pure_text(args)
    ##### Speech + Language #####
    elif test_type == "speech_language_with_lora":
        main_with_lora_speech(args)
    elif test_type == "speech_language_with_lora_batch":
        main_with_lora_speech_batch(args)
    elif test_type == "speech_language_without_lora":
        main_with_lora_speech(args, activate_lora_request=False)
    ##### Vision + Language #####
    elif test_type == "vision_language_with_lora":
        main_with_lora_vision(args)
    elif test_type == "vision_language_with_lora_batch":
        main_with_lora_vision_batch(args)
    elif test_type == "vision_language_without_lora":
        main_with_lora_vision(args, activate_lora_request=False)
    ##### Vision + Speech + Language #####
    elif test_type == "vision_speech_language_with_lora":
        main_with_lora_vision_speech(args)
    elif test_type == "vision_speech_language_with_lora_batch":
        main_with_lora_vision_speech_batch(args)
    elif test_type == "vision_speech_language_without_lora":
        main_with_lora_vision_speech(args, activate_lora_request=False)
