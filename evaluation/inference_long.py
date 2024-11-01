'''
 Copyright (c) ByteDance Inc.
 Authors:
  - Tongping Liu (tongping.liu@bytedance.com)
'''

from transformers import pipeline, set_seed
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    " I want you to act as a storyteller. You will come up with entertaining stories that are engaging, imaginative and captivating for the audience. It can be fairy tales, educational stories or any other type of stories which has the potential to capture people’s attention and imagination. Depending on the target audience, you may choose specific themes or topics for your storytelling session e.g., if it’s children then you can talk about animals; If it’s adults then history-based tales might engage them better etc. My first request is “I need an interesting story on perseverance.",
    " I want you to act as an advertiser. You will create a campaign to promote a product or service of your choice. You will choose a target audience, develop key messages and slogans, select the media channels for promotion, and decide on any additional activities needed to reach your goals. My first suggestion request is “I need help creating an advertising campaign for a new type of energy drink targeting young adults aged 18-30.”",
    "I want you to act as a travel guide. I will write you my location and you will suggest a place to visit near my location. In some cases, I will also give you the type of places I will visit. You will also suggest me places of similar type that are close to my first location. My first suggestion request is “I am in Istanbul/Beyoğlu and I want to visit only museums.”",
    "Please summarize the following paragraphs in less than 20 words: Large language models trained on massive text collections have shown surprising emergent capabilities to generate text and perform zero- and few-shot learning. While in some cases the public can interact with these models through paid APIs, full model access is currently limited to only a few highly resourced labs. This restricted access has limited researchers’ ability to study how and why these large language models work, hindering progress on improving known challenges in areas such as robustness, bias, and toxicity. We present Open Pretrained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We train the OPT models to roughly match the performance and sizes of the GPT-3 class of models, while also applying the latest best practices in data collection and efficient training. Our aim in developing this suite of OPT models is to enable reproducible and responsible research at scale, and to bring more voices to the table in studying the impact of these LLMs. Definitions of risk, harm, bias, and toxicity, etc., should be articulated by the collective research community as a whole, which is only possible when models are available for study.",
]

prompts = prompts * 10
set_seed(32)

import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

# Create a sampling params object.
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8192, ignore_eos=True)
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8192, ignore_eos=True)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
#sampling_params = SamplingParams(temperature=0, top_p=1, top_k=1,max_tokens=2048)

# Create an LLM.
llm = LLM(model="facebook/opt-6.7b", use_dattn=True, enforce_eager=True)
#llm = LLM(model="facebook/opt-6.7b", enforce_eager=True)
#llm = LLM(model="facebook/opt-6.7b", enforce_eager=True)
#llm = LLM(model="facebook/opt-1.3b", enforce_eager=True)
#llm = LLM(model="facebook/opt-125m", use_dattn=True, enforce_eager=True)
#llm = LLM(model="facebook/opt-125m", enforce_eager=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
total = 0
for index, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    total += len(generated_text)
    print(f"prompt-{index}, generated text:{len(generated_text)}")
    #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    #print(f"Prompt: {prompt!r}\n, Generated text: {generated_text!r}\n\n")

print(f"generated text with the total length-{total}")
