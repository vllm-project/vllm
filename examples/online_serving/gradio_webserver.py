# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example for starting a Gradio Webserver
Start vLLM API server:
    python -m vllm.entrypoints.api_server \
        --model meta-llama/Llama-2-7b-chat-hf

Start Webserver:
    python examples/online_serving/gradio_webserver.py

Note that `pip install --upgrade gradio` is needed to run this example.
More details: https://github.com/gradio-app/gradio

If your antivirus software blocks the download of frpc for gradio,
you can install it manually by following these steps:

1. Download this file: https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64
2. Rename the downloaded file to: frpc_linux_amd64_v0.3
3. Move the file to this location: /home/user/.cache/huggingface/gradio/frpc
"""

import argparse
import json

import gradio as gr
import requests


def http_bot(prompt):
    headers = {"User-Agent": "vLLM Client"}
    pload = {
        "prompt": prompt,
        "stream": True,
        "max_tokens": 128,
    }
    response = requests.post(args.model_url, headers=headers, json=pload, stream=True)

    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\n"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"][0]
            yield output


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# vLLM text completion demo\n")
        inputbox = gr.Textbox(label="Input", placeholder="Enter text and press ENTER")
        outputbox = gr.Textbox(
            label="Output", placeholder="Generated result from the model"
        )
        inputbox.submit(http_bot, [inputbox], [outputbox])
    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--model-url", type=str, default="http://localhost:8000/generate"
    )
    return parser.parse_args()


def main(args):
    demo = build_demo()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
