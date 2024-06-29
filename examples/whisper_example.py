import vllm
import torch
import requests
from vllm import LLM
from vllm.multimodal.audio import AudioData
from datasets import Audio


def main():
    sr = 16000
    audio = Audio(sampling_rate=sr)
    llm = LLM(
        model="openai/whisper-large-v3",
        max_num_seqs = 1,
        max_seq_len_to_capture = 448,
        max_model_len = 448,
        gpu_memory_utilization = 0.4
    )

    r = requests.get('https://github.com/mesolitica/malaya-speech/raw/master/speech/singlish/singlish0.wav')
    y = audio.decode_example(audio.encode_example(r.content))['array']

    output_lang = llm.generate({
        "prompt_token_ids": [50258],
        "multi_modal_data": AudioData(y),
    }, sampling_params = SamplingParams(max_tokens = 1, temperature = 0))

    outputs = llm.generate({
        "prompt_token_ids": [50258, output_lang[0].outputs[0].token_ids[0], 50360],
        "multi_modal_data": AudioData(y),
    }, sampling_params = SamplingParams(max_tokens = 10, temperature = 0))

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
