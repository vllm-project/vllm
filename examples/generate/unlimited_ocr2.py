# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse page images with decode CUDA graphs and bounded document windows."""

import argparse
from collections.abc import Iterator, Sequence

from PIL import Image

from vllm import LLM, SamplingParams


def page_windows(
    paths: Sequence[str], window_size: int, overlap: int
) -> Iterator[tuple[Sequence[str], int]]:
    """Yield bounded windows and the number of already emitted context pages."""
    if window_size == 0:
        yield paths, 0
        return
    if window_size < 1 or not 0 <= overlap < window_size:
        raise ValueError("Require page_window > overlap_pages >= 0")
    start = 0
    while start < len(paths):
        context = min(start, overlap)
        end = min(len(paths), start + window_size - context)
        yield paths[start - context : end], context
        start = end


def document_pages(text: str, expected: int, context: int) -> list[str]:
    """Discard overlapping context and repeated tail pages from a window."""
    pages = text.split("<PAGE>")[1:]
    if len(pages) < expected:
        raise RuntimeError(
            f"Received {len(pages)} page markers for {expected} images; "
            "increase max_tokens or reduce page_window"
        )
    return ["<PAGE>" + page for page in pages[context:expected]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Page images in document order")
    parser.add_argument("--model", default="baidu/Unlimited-OCR-2")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument(
        "--page-window", type=int, default=10, help="0 disables windows"
    )
    parser.add_argument("--overlap-pages", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"],
        default="FLASH_ATTN",
    )
    args = parser.parse_args()
    # Validate before loading the model or requesting GPU memory.
    first_paths, _ = next(
        page_windows(args.images, args.page_window, args.overlap_pages)
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        enforce_eager=args.enforce_eager,
        attention_config={"backend": args.attention_backend},
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(first_paths)},
        mm_processor_kwargs={"crop_mode": False},
        mm_encoder_attn_backend="TORCH_SDPA",
        compilation_config={"cudagraph_capture_sizes": [1, 2]},
        kernel_config={"enable_jit_warmup": False},
    )
    sampling = SamplingParams(
        temperature=0, max_tokens=args.max_tokens, skip_special_tokens=False
    )
    for paths, context in page_windows(
        args.images, args.page_window, args.overlap_pages
    ):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        output = (
            llm.generate(
                {
                    "prompt": "<PAGE><image>" * len(images) + "Multi page parsing.",
                    "multi_modal_data": {"image": images},
                },
                sampling,
            )[0]
            .outputs[0]
            .text
        )
        for page in document_pages(output, len(images), context):
            print(page, flush=True)


if __name__ == "__main__":
    main()
