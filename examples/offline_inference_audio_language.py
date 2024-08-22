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

# Input audio and question
audio_and_sample_rate = AudioAsset("mary_had_lamb").audio_and_sample_rate
question = "What is recited in the audio?"


# Ultravox 0.3
def run_ultravox(question):
    model_name = "fixie-ai/ultravox-v0_3"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{
        'role': 'user',
        'content': f"<|reserved_special_token_0|>\n{question}"
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    llm = LLM(model=model_name)
    stop_token_ids = None
    return llm, prompt, stop_token_ids


model_example_map = {
    "ultravox": run_ultravox,
}


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    llm, prompt, stop_token_ids = model_example_map[model](question)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=stop_token_ids)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_and_sample_rate
            },
        }

    else:
        # Batch inference
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_and_sample_rate
            },
        } for _ in range(args.num_prompts)]

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

    args = parser.parse_args()
    main(args)
