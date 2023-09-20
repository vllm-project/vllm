# 声明使用utf8编码
# -*- coding: utf-8 -*-
import json
import requests
from vllm import LLM, SamplingParams


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def vllm_request_template(url, prompt, temperature, max_tokens=4096, use_beam_search=False, num_of_seqencens=1, top_k=1, top_p=0.95, stream=True, stop="$$"):
    """vllms部署llama2访问接口模板(完整版)
    """

    def get_stream_res(response):
        for chunk in response.iter_lines(chunk_size=8192,decode_unicode=False,delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"]
                yield output 

    data = {
        "prompt": prompt,
        "use_beam_search": use_beam_search,
        "n": num_of_seqencens,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop": stop,
        "stream": stream 
    }
    # data = json.dumps(data)
    if stream:
        headers = {"User-Agent": "stream test client"}
    else:
        headers = {'Content-Type': 'application/json'}

    res = requests.post(url, headers=headers, json=data, stream=stream)

    if stream:
        num_printed_lines = 0
        for h in get_stream_res(res):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            is_end = False
            tmp_s = ""
            for i, line in enumerate(h):
                num_printed_lines += 1
                if line.endswith("$$"):     # 检测到padding符号，终止
                    line = line.replace("$$", "").strip()
                    is_end = True
                tmp_s += line
                print(f"Beam candidate {i}: {line!r}", flush=True)
                if is_end:
                    print("最终额输出是:", tmp_s)
                    return 
                print("临时输出:", tmp_s)
    else:
        # ans = json.loads(res.text)["text"][0].strip().split("### Response:")[1].strip()
        try:
            res_flag = "### Response:"
            ans = json.loads(res.text)["text"][0].strip()
            if res_flag in ans: 
                ans = ans.split(res_flag)[1].strip()
        except Exception as e:
            print("检测到异常的返回，res.text: ", res.text)
            print(e)
            ans = ""
        return ans 
    pass

def normal_api_main():
    url = "http://ip:port/generate"
    prompt = "你好，今天天气不错，我想出去玩，请给我推荐一些户外运动？" 
    # prompt = "你好，帮我写一篇500字关于北京的近代史的文章"
    prompt_input_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n {instruction} \n\n### Response:\n\n"
    )
    inp_prompt = prompt_input_template.format(instruction=prompt)
    temperature = 0.1
    stream = True 
    stop = "$$"
    print(vllm_request_template(url, inp_prompt, temperature, max_tokens=4096, use_beam_search=False, num_of_seqencens=1, top_k=-1, top_p=1.0, stream=stream, stop=stop))

if __name__ == "__main__":
    normal_api_main()
    pass
