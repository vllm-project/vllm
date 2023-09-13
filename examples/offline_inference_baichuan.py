from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List

import torch


def build_chat_input(tokenizer,
                     messages: List[dict],
                     max_new_tokens: int = 2048,
                     model_max_length: int = 4096,
                     user_token_id: int = 195,
                     assistant_token_id: int = 196):

    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens
    max_input_tokens = model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(user_token_id)
            else:
                round_tokens.append(assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(
                round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return input_tokens


# Sample prompts.
prompts = [
    [{
        "role": "user",
        "content": "Hello, my name is"
    }],
    [{
        "role": "user",
        "content": "The president of the United States is"
    }],
    [{
        "role": "user",
        "content": "The capital of France is"
    }],
    [{
        "role": "user",
        "content": "The future of AI is"
    }],
]

# Taken from generation_config.
sampling_params = SamplingParams(temperature=0.3,
                                 top_k=5,
                                 top_p=0.85,
                                 max_tokens=2048
                                 # repetition_penalty=1.05
                                 )

# Create an LLM.
llm = LLM(model="baichuan-inc/Baichuan2-13B-Chat",
          trust_remote_code=True,
          tensor_parallel_size=2,
          tokenizer_mode='slow')
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat",
                                          use_fast=False,
                                          trust_remote_code=True)
# bypass chat format issue with explicit tokenization
prompt_token_ids = [
    build_chat_input(tokenizer, prompt, max_new_tokens=2048)
    for prompt in prompts
]
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompt_token_ids=prompt_token_ids,
                       sampling_params=sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r},\n Generated text: {generated_text!r}")
"""
Sample output
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:13<00:00,  1.93s/it]
Prompt: None,
 Generated text: 'Hello, my name is [your name]. Nice to meet you!'
Prompt: None,
 Generated text: 'The current president of the United States is Joe Biden, who was sworn into office on January 20, 2021.'
>>>>>> Baichuan2-13B-Chat 8Bit Demo: 
"the President of the United States is currently Joe Biden, who was sworn into office on January 20, 2021. He succeeded former president Donald Trump. The next presidential election will be held in November 2024."
>>>>>>
Prompt: None,
 Generated text: 'The capital of France is Paris.'
Prompt: None,
 Generated text: "The future of AI is promising and full of potential. It is expected to revolutionize various industries, including healthcare, finance, transportation, and manufacturing. AI will also play a significant role in solving complex problems, such as climate change, disease, and poverty.\n\nSome of the key trends in AI include:\n\n1. Autonomous vehicles: AI-powered self-driving cars and trucks are expected to become a reality in the near future, reducing traffic accidents, improving fuel efficiency, and transforming transportation as we know it.\n\n2. Personalized healthcare: AI will help analyze vast amounts of medical data, enabling doctors to make more accurate diagnoses and develop personalized treatment plans for patients.\n\n3. Virtual assistants: AI-powered virtual assistants will become more advanced, capable of understanding natural language, predicting user needs, and providing personalized recommendations.\n\n4. Manufacturing efficiency: AI will help optimize manufacturing processes, reducing waste, improving product quality, and lowering production costs.\n\n5. Financial services: AI will transform the financial industry by automating trading, fraud detection, and risk management, as well as providing personalized financial advice to consumers.\n\n6. Education: AI will help create personalized learning experiences, identifying students' strengths and weaknesses and providing targeted support to help them succeed.\n\n7. Smart cities: AI will play a crucial role in making cities more efficient and sustainable by optimizing energy consumption, traffic management, and public services.\n\n8. Language translation: AI-powered translation tools will become more accurate and natural, breaking down language barriers and facilitating global communication.\n\n9. Creative industries: AI will help drive innovation in the creative industries, such as music, art, and film, by collaborating with humans to create new works and improve existing ones.\n\n10. Ethical considerations: As AI becomes more integrated into our lives, ethical questions and concerns about privacy, security, and job displacement will need to be addressed.\n\nIn conclusion, the future of AI is full of possibilities and potential, but it also comes with challenges that need to be addressed. By embracing AI and working towards its responsible development, we can unlock its potential to improve our lives and create a better future for all."
 
>>>>>> Baichuan2-13B-Chat 8Bit Demo:
The future of AI is promising and full of potential. It is expected to revolutionize various industries, including healthcare, finance, transportation, manufacturing, and education. Some of the key areas where AI is expected to make a significant impact include:

1. Healthcare: AI will play a crucial role in improving diagnostics, treatment planning, and personalized medicine. Machine learning algorithms can analyze medical images, patient records, and genetic data to identify patterns and make accurate predictions. This will help in early detection of diseases, better treatment options, and improved patient outcomes.

2. Finance: AI will transform the financial industry by automating processes, enhancing risk management, and improving customer service. AI-powered algorithms can analyze vast amounts of financial data to detect fraud, assess credit risk, and optimize investment strategies.

3. Transportation: Autonomous vehicles are expected to become a reality in the near future, thanks to advancements in AI and machine learning. Self-driving cars have the potential to reduce traffic accidents, improve fuel efficiency, and revolutionize urban transportation systems.

4. Manufacturing: AI will enhance productivity and efficiency in the manufacturing sector by automating repetitive tasks, optimizing supply chains, and predicting equipment failures. This will enable companies to produce higher-quality products at a lower cost.

5. Education: AI will transform the way we learn and teach by providing personalized learning experiences, identifying students' strengths and weaknesses, and offering targeted support. AI-powered tools can also help educators in managing classrooms, tracking student progress, and identifying learning gaps.

6. Customer Service: AI-powered chatbots and virtual assistants will improve customer service by handling routine inquiries, providing personalized recommendations, and assisting human agents in resolving complex issues.

7. Environment and Sustainability: AI can help monitor and predict environmental changes, optimize resource usage, and develop sustainable solutions for climate change and pollution control.

However, the future of AI also comes with its challenges, such as job displacement, privacy concerns, and ethical considerations. As AI continues to advance, it is essential to address these challenges and ensure that its benefits are accessible to everyone.
>>>>>
 """
