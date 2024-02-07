"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List
import pdb
import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": [{"role": "user", "content": prompt}],
        "n":n,
        "use_beam_search": False,
        "stream": stream
    }
    print(api_url)
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    print()
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n
    stream = args.stream
    prompt = "你是私有数据库搜索引擎，正在根据用户的问题搜索和总结答案，请认真作答，并给出出处。\n            用户的问题为：\n            哪些文件里面有盐雾的测试条件？\n            你搜索到的相关内容为：\n            HQ WD6931 手表主板DFM 报告_20240102/slide_5 ：文件标题：\nHQ WD6931 手表主板DFM 报告_20240102/slide_5\n:/n文件内容：# WD6931手表主板防护要求\n\n__待防护产品__\n\n__    __ WD6931手表主板尺寸: 36\\.7 x 30\\.8 mm\n\n__UPH __  __要求__\n\nTBD\n\n__防护要求__\n\n需求一：\n\n1\\. 整机测试要求：120H高温高湿（55℃，95%RH）开机存储测试120H检查外观、功能\n\n2\\. 板级测试：裸板IPX4\n\n需求二：\n\n1\\. 整机测试要求：120H高温高湿（55℃，95%RH）开机存储测试120H检查外观、功能\n\n2\\. 板级测试: 裸板通电状态下喷淋盐水10s，放入温箱80度保持10mins后，检查电流是否正常。共重复做20个循环。\n\n\nD手机镀膜 DFM报告/pdf_13 ：文件标题：\nD手机镀膜 DFM报告/pdf_13\n:/n文件内容：IQC摆盘干燥\nOQC 检测 纳米镀膜\n镀膜工序流程\n\n"
    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url, n, stream)

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)
