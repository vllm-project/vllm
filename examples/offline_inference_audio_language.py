"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on audio language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.utils import FlexibleArgumentParser

audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]
question_per_audio_count = [
    "What is recited in the audio?",
    "What sport and what nursery rhyme are referenced?"
]


# Ultravox 0.3
def run_ultravox(question, audio_count):
    model_name = "fixie-ai/ultravox-v0_3"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{
        'role':
        'user',
        'content':
        "<|reserved_special_token_0|>\n" * audio_count + question
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    llm = LLM(model=model_name,
              enforce_eager=True,
              enable_chunked_prefill=False,
              max_model_len=8192,
              limit_mm_per_prompt={"audio": audio_count})
    stop_token_ids = None
    return llm, prompt, stop_token_ids


model_example_map = {
    "ultravox": run_ultravox,
}


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    audio_count = args.num_audios
    llm, prompt, stop_token_ids = model_example_map[model](
        question_per_audio_count[audio_count - 1], audio_count)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=stop_token_ids)

    assert args.num_prompts > 0
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": [
                asset.audio_and_sample_rate
                for asset in audio_assets[:audio_count]
            ]
        },
    }
    if args.num_prompts > 1:
        # Batch inference
        inputs = [inputs] * args.num_prompts

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'audio language models')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="ultravox",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument("--num-audios",
                        type=int,
                        default=1,
                        choices=[1, 2],
                        help="Number of audio items per prompt.")

    args = parser.parse_args()
    main(args)
