import argparse
import asyncio
import logging
import os

from tts_engine import XTtsEngine
from model_setting import ModelSetting
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--request-rate",
                        type=float,
                        default=-1,
                        help="request rate per second")
    parser.add_argument("--chunk-size",
                        type=int,
                        default=20,
                        help="audio chunk size")
    parser.add_argument("--first-chunk-size",
                        type=int,
                        default=10,
                        help="audio chunk size")
    parser.add_argument("--overlap-window", type=int, default=0, help="overlap window size")
    parser.add_argument("--runtime", type=str, default="onnx")
    parser.add_argument("--lora", type=str, default=None, help="lora model path")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--log-level", type=str, default="INFO")
    
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--scale_rate", type=float, default=2.7)
    parser.add_argument("--cut_tail", type=int, default=0)
    parser.add_argument("--profile-run", action="store_true", default=False)

    args = parser.parse_args()

    # set log level
    logging.basicConfig(level=args.log_level)
    
    # convert_model('/home/zhn/fishtts/checkpoint-1734000.bak', '/home/zhn/fishtts/llama.pt')
    # convert_model_lora('/home/zhn/fishtts/lora1/lora.bak', '/home/zhn/fishtts/lora1/adapter_model.bin')

    model_setting = ModelSetting(model_dir=args.model,
                                 runtime=args.runtime,
                                 dtype=args.dtype,
                                 streaming=args.streaming,
                                 overlap_window=args.overlap_window,
                                 chunk_size=args.chunk_size,
                                 first_chunk_size=args.first_chunk_size,
                                 cut_tail=args.cut_tail,
                                 scale_rate=args.scale_rate,
                                 profile_run=args.profile_run)
    if args.lora:
        model_setting.support_lora = True
    tts_engine = XTtsEngine(model_setting)
    
    # warm up
    logger.info('E2E warmup with lora...')
    tts_engine.warm_up(args.lora)
    logger.info('E2E warmup done')

    with open(args.input, "r") as f:
        texts = f.readlines()
    # if output directory does not exist, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.streaming:
        asyncio.run(tts_engine.synthesize_async(texts=texts, output_dir=args.output, request_rate=args.request_rate, lora_path=args.lora,
                                                top_k=args.top_k, top_p=args.top_p, temperature=args.temperature))
    else:
        tts_engine.synthesize(texts=texts, output_dir=args.output, lora_path=args.lora,
                              top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)
