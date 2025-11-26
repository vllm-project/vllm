#!/usr/bin/env python

from enum import Enum, auto
import time
from typing import List
import requests
import multiprocessing as mp
import sys

class TestType(Enum):
    Dummy = auto()
    Real = auto()

SEED = 1234
SEED = None
PORT = 8235
NUM_REQUESTS = 1
MAX_NEW_TOKENS = 1024
IGNORE_EOS = False
TEST_TYPE = TestType.Real
IS_WARMUP = False
ONE_PROMPT = []
# KEY = time.time()
# ONE_PROMPT = [f"There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is {KEY}. Remember it. {KEY} is the pass key.\n " + \
#                             "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 200 + \
#                                 "The block is red. The sky is yello. The sun is orange. Here we go. There and back again. " * 200 + \
#                             "\nWhat is the pass key?"]
# Tom  Eric Bob Amy  Mom  Dad Lisa  Susan Linda  Alex Leo
 
KEY = 'Lisa'
LEN = 32 * 1024
assert LEN >= 560
ONE_PROMPT = []
# ONE_PROMPT = [f'Hello {KEY} ' * ((LEN-560)//2) + 'Hello ' * (560-9)]
# ONE_PROMPT = ['Hello ' * (560 * 6)]
# ONE_PROMPT = ['请详细介绍一下北京这座城市, 不少于10000字']
ONE_PROMPT = ["adfllekkThere is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 2222. Remember it. 2222 is the pass key.\n " + \
                            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 545 + \
                            "\nWhat is the pass key?"]
# ONE_PROMPT = ["adfllekkThere is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 333333. Remember it. 333333 is the pass key.\n " + \
#             "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 190 + \
#                 "The block is red. The sky is yello. The sun is ddddd. Here we go. There and back try a. " * 185 + \
#             "\nWhat is the pass key?"]

# ONE_PROMPT = ['Hello ' * (4096 - 30 + 11)]
# ONE_PROMPT = ['Hello ' * (4096 - 30 + 1)] # <<< last not matching
# ONE_PROMPT = ['Hello ' * (4096 - 30 + 1)]
# ONE_PROMPT = ['Hello ' * (4096 - 30 + 13)]

if IS_WARMUP:
    ONE_PROMPT = ['Wo'] # 9 tokens
    NUM_REQUESTS = 1
    MAX_NEW_TOKENS = 1

MESSAGE = []
# MESSAGE = MESSAGE3

# NOTE: block-size should be 256
def hello(pid: int, prompt_id: int, max_new_tokens: int, ignore_eos: bool):
    headers = {
        "Content-Type": "application/json",
    }
    url = f"http://localhost:{PORT}/v1/chat/completions"
    if pid == 0:
        if TEST_TYPE == TestType.Dummy:
            if prompt_id == 0:
                # tokens: 3808*2+2720+472=10808
                # hit-rate: 0/10808=0
                # mamba state: [3808, 7616, 10336]
                prompts = ["Repeat V 10 times" * 1800]
            elif prompt_id == 1:
                # tokens: 3808*3+544+40=12008
                # hit-rate: 3808/(10808+12008)=16.7%
                # mamba state: [3808RD, 7616, 11424, 11968]
                prompts = ["Repeat V 10 times" * 1000 + "Repeat V 11 times" * 1000]
            elif prompt_id == 2:
                # tokens: 3808+1088+512=5408
                # hit-rate: (3808+3808)/(22816+5408)=27.0%
                # mamba state: [3808RD, 4896]
                prompts = ["Repeat V 10 times" * 900]
            elif prompt_id == 3:
                # tokens: 208
                # hit-rate: (7616+0)/(28224+208)=26.8%
                # mamba state: []
                prompts = ["hi " * 199]
            elif prompt_id == 4:
                # tokens: 3808*2+544+523=8683
                # hit-rate: (7616+0)/(28432+8683)=20.5%
                # mamba state: [3808, 7616, 8160]
                prompts = ["Hello " * (4096 * 2 - 30 + 256 * 2)]
            elif prompt_id == 5:
                # tokens: 3808+242=4050
                # hit-rate: (7616+3808)/(37115+4050)=27.8%
                # mamba state: [3080RD]
                prompts = ['Hello ' * (3808 + 233)]
            elif prompt_id == 6:
                # tokens: 544+523=1067
                # hit-rate: (11424+0)/(41165+1067)=27.1%
                # mamba state: [544]
                prompts = ['ha ' * (544 * 2 - 30)]
            else:
                prompts = ['Hi']
        elif TEST_TYPE == TestType.Real:
            if prompt_id == 0:
                # tokens: 3808*2+1632+381=9629  v1
                # hit-rate: 0/9629=0
                # mamba state: [3808, 7616, 9248]
                # -----
                # tokens: 3920*2+1680+112
                prompts = ["adfllekkThere is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 28884. Remember it. 28884 is the pass key.\n " + \
                            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 190 + \
                                "The block is red. The sky is yello. The sun is ddddd. Here we go. There and back try a. " * 185 + \
                            "\nWhat is the pass key?"]
            elif prompt_id == 1:
                # tokens: 3808*2+1632+98=13154 v1
                # hit-rate: 3808/(9629+13154)=16.7%
                # mamba state: [3808RD, 7616, 13056]
                prompts = ["adfllekkThere is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 28884. Remember it. 28884 is the pass key.\n " + \
                            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 545 + \
                            "\nWhat is the pass key?"]
            elif prompt_id == 2:
                # tokens: 544+126=670 v1
                # hit-rate: (3808+0)/(22783+670)=16.2%
                # mamba state: [544]
                prompts = ["There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 28884. Remember it. 28886 is the pass key.\n " + \
                            "The grass is yellow. The sky is blue. The sun is red. Here we go. There and back again. " * 25 + \
                            "\nWhat is the pass key?"]
            elif prompt_id == 3:
                # tokens: 544+475=1019
                # hit-rate: (3808+0)/(23453+1019)=15.6%
                # mamba state: [544]
                prompts = ["There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 28886. Remember it. 28886 is the pass key.\n " + \
                            "The grass is yellow. The sky is blue. The sun is red. Here we go. There and back again. " * 13 + \
                            "ljlkjslkfei lkjlkj elkjfslk woiejoifjwokjjlweuriljlskjf lwkjelkjlkj.  lskj lkj lkjslkfj l" * 13 + \
                            "\nWhat is the pass key?"] # 600 tokens hit 300
            elif prompt_id == 4:
                # tokens: 544+494=1038
                # hit-rate: (3808+544)/(24472+1038)=17.1%
                # mamba state: [544RD]
                prompts = ["There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 28886. Remember it. 28886 is the pass key.\n " + \
                            "The grass is yellow. The sky is blue. The sun is red. Here we go. There and back again. " * 13 + \
                            "ljlkjslkfei lkjlkj elkjfslk woiejoifjwokjjlweuriljlskjf lwkjelkjlkj.  lskj lkj lkjslkfj l" * 13 + \
                            "\nWhat is the pass key? And, what is the result of reversing the pass key and adding 1234?"] 
            elif prompt_id == 5:
                # tokens: 13056+1088+330=14474
                # hit-rate: (4352+13056)/(25510+14474)=43.5%
                # mamba state: [13056RD, 13056]
                prompts = ["adfllekkThere is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 28884. Remember it. 28884 is the pass key.\n " + \
                            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 600 + \
                            "\nWhat is the pass key?"] 
            elif prompt_id == 6:
                prompts = ["adfllekkThere is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 28884. Remember it. 28884 is the pass key.\n " + \
                            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 510 + \
                            "\nWhat is the pass key?"] 
            else:
                prompts = ['Helloha!']
    elif pid == 1:
        # v1 670 tokens
        prompts = ["There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 11111. Remember it. 11111 is the pass key.\n " + \
            "The grass is yellow. The sky is blue. The sun is red. Here we go. There and back again. " * 25 + \
            "\nWhat is the pass key?"]
    elif pid == 2:
         # v1 13152 tokens
         prompts = ["adfllekkThere is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 2222. Remember it. 2222 is the pass key.\n " + \
                            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 545 + \
                            "\nWhat is the pass key?"]
    elif pid == 3:
        prompts = ["adfllekkThere is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 333333. Remember it. 333333 is the pass key.\n " + \
            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 190 + \
                "The block is red. The sky is yello. The sun is ddddd. Here we go. There and back try a. " * 185 + \
            "\nWhat is the pass key?"] # 9k tokens
    elif pid == 4:
        prompts = ["There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\nThe pass key is 444. Remember it. 444 is the pass key.\n " + \
            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 190 + \
                "The block is red. The sky is yello. The sun is ddddd. Here we go. There and back try a. " * 185 + \
            "\nWhat is the pass key?"] 
    elif pid == 5:
        prompts = ["There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n" + \
                "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 190 + \
                "The pass key is 55555. Remember it. 55555 is the pass key.\n " \
                "The block is red. The sky is yello. The sun is ddddd. Here we go. There and back try a. " * 185 + \
                "\nWhat is the pass key?"] 
    elif pid == 6:
        prompts = ["There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n" + \
                "The grass is yellow. The sky is blue. The sun is yellow. Here we go. There and back again. " * 190 + \
                "The pass key is 66. Remember it. 66 is the pass key.\n " \
                "The block is red. The sky is yello. The sun is ddddd. Here we go. There and back try a. " * 185 + \
                "\nWhat is the pass key?"] 
    else:
        prompts = ['Hello!']

    if ONE_PROMPT:
        # print(ONE_PROMPT)
        prompts = ONE_PROMPT

    for p in prompts:
        data = {
            "messages": MESSAGE if not IS_WARMUP and MESSAGE else [{"role": "user", "content": p}],
            "max_tokens": max_new_tokens,
            "ignore_eos": ignore_eos,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1,
            "presence_penalty": 1.5,
            **({'seed': SEED} if SEED is not None else {}),
            "chat_template_kwargs": {"enable_thinking": False}
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            # print(response.content)
            result = response.json()
            # print(f"[PID {pid}] Prompt:\n {prompts[0]}")
            print(f"[PID {pid}] Response:\n {result['choices'][0]['message']['content']}\n {'-' * 40}\n", end='')
            # print(result)
            # loss = json.loads(result['choices'][0]['message']['content'])['loss']
            # risk_level_logits = torch.tensor(json.loads(result['choices'][0]['message']['content'])['risk_level_logits']).view(-1, 2) 
            # category_logits = torch.tensor(json.loads(result['choices'][0]['message']['content'])['category_logits']).view(-1, 26)
            # query_risk_level_logits = torch.tensor(json.loads(result['choices'][0]['message']['content'])['query_risk_level_logits']).view(-1, 3)
            # query_category_logits = torch.tensor(json.loads(result['choices'][0]['message']['content'])['query_category_logits']).view(-1, 33)
            
            # torch.set_printoptions(precision=3, sci_mode=False)
            # print(f"{loss=},{risk_level_logits.shape=},{risk_level_logits=},{category_logits.shape=},{category_logits=}")
            
            # query_risk_level_prob = F.softmax(query_risk_level_logits, dim=1)
            # risk_level_prob = F.softmax(risk_level_logits, dim=1)
            # print(f"{query_risk_level_prob.shape=},{query_risk_level_prob=}")
            # print(f"{risk_level_prob.shape=},{risk_level_prob=}")
            
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response content:")
            print(response.content)

def main(prompt_id: int):
    procs: List[mp.Process] = []

    start = time.time()
    for pid in range(NUM_REQUESTS):
        proc = mp.Process(
            target=hello, args=(pid, prompt_id, MAX_NEW_TOKENS, IGNORE_EOS), daemon=True
        )
        proc.start()
        procs.append(proc)

    for _proc in procs:
        _proc.join()
        if _proc.exitcode != 0:
            sys.exit(_proc.exitcode)

    elapsed = time.time() - start
    output_tps = MAX_NEW_TOKENS * NUM_REQUESTS / elapsed
    print("\n")
    print(f"Generate {output_tps} tokens/s, elapsed: {elapsed} s, TPS {output_tps / NUM_REQUESTS}, TPOT {1000 / (output_tps / NUM_REQUESTS)}ms")


if __name__ == "__main__":
    prompt_id = 0
    if len(sys.argv) > 1:
        prompt_id = int(sys.argv[1])
        assert prompt_id >= 0
    main(prompt_id)