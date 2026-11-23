# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert document pages with Granite Docling, choosing the connector route of
each page with the checkpoint's density router.

Launch the vLLM server with the following command:

vllm serve ibm-granite/granite-for-docling-500m \
    --trust-request-mm-kwargs --api-server-count 8 --gpu-memory-utilization 0.8

Then run this script with the pages to convert:

python granite_docling_density_router_online.py page1.png page2.png
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from openai import OpenAI
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vllm.multimodal.utils import encode_image_url


class DensityRouter:
    """The density router decides whether a page takes the fine connector route
    (four times as many image tokens per tile).

    It reads the vision encoder output, while vLLM fixes the number of image tokens
    before the encoder runs, so the route is chosen here and sent with the request.
    """

    def __init__(self, model: str, device: str):
        self.image_processor = AutoProcessor.from_pretrained(model).image_processor
        model = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16)
        # Only the vision encoder and the router are needed
        self.model = model.model
        self.model.text_model = None
        self.model.to(device).eval()
        self.device = device
        self.lock = threading.Lock()

    def __call__(self, image: Image.Image) -> bool:
        # Preprocessing a page takes far longer than routing it, so it runs in the
        # calling thread and only the forward pass is serialized. Routing page by
        # page limits the client to about a dozen pages per second.
        inputs = self.image_processor(images=[[image]], return_tensors="pt")
        with self.lock, torch.inference_mode():
            fine_route = self.model.predict_fine_route(
                inputs["pixel_values"].to(self.device),
                inputs["pixel_attention_mask"].to(self.device),
            )
        return bool(fine_route.item())


def convert_page(client: OpenAI, model: str, router: DensityRouter, path: str) -> str:
    image = Image.open(path).convert("RGB")
    fine_route = router(image)
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_url(image)},
                    },
                    {"type": "text", "text": "<doclang>"},
                ],
            }
        ],
        temperature=0.0,
        max_completion_tokens=16384,
        extra_body={
            "skip_special_tokens": False,
            "mm_processor_kwargs": {"fine_route": fine_route},
        },
    )
    return chat_completion.choices[0].message.content


def parse_args():
    parser = argparse.ArgumentParser(
        description="Granite Docling client that routes each page with the "
        "density router"
    )
    parser.add_argument("pages", nargs="+", help="Page images to convert")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--device", default="cuda", help="Device of the router")
    parser.add_argument(
        "--num-workers", type=int, default=64, help="Number of concurrent requests"
    )
    return parser.parse_args()


def main(args):
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)
    model = client.models.list().data[0].id
    router = DensityRouter(model, args.device)

    with ThreadPoolExecutor(args.num_workers) as executor:
        outputs = executor.map(
            lambda path: convert_page(client, model, router, path), args.pages
        )
        for path, output in zip(args.pages, outputs):
            print(f"{path}:\n{output}\n")


if __name__ == "__main__":
    main(parse_args())
