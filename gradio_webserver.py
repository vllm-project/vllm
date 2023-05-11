import argparse
import json
import time

import gradio as gr
import requests


def http_bot(prompt):
    headers = {"User-Agent": "Cacheflow Client"}
    pload = {
        "prompt": prompt,
        "max_tokens": 128,
    }
    response = requests.post(args.model_url, headers=headers, json=pload, stream=True)

    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"][0]
            yield output


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Cacheflow demo\n"
        )
        inputbox = gr.Textbox(label="Input", placeholder="Enter text and press ENTER")# .style(container=False)
        outputbox = gr.Textbox(label="Output", placeholder="Generated result from the model")
        inputbox.submit(http_bot, [inputbox], [outputbox])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10003)
    parser.add_argument("--model-url", type=str, default="http://localhost:10002/generate")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(concurrency_count=100).launch(server_name=args.host, server_port=args.port)