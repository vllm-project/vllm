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
    prompt = '<|startoftranscript|><|en|><|transcribe|>'
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": AudioData(y),
    })
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
