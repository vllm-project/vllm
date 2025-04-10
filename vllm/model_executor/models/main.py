import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, QuantoConfig, GenerationConfig
import torch
import safetensors
import argparse
import os
import json
from PIL import Image

"""
 usage:
    export SAFETENSORS_FAST_GPU=1
    python main.py --quant_type int8 --world_size 8 --model_id <model_path> --image_path <image_path>
"""

def generate_quanto_config(hf_config: AutoConfig, quant_type: str):
    QUANT_TYPE_MAP = {
        "default": None,
        "int8": QuantoConfig(
            weights="int8",
            modules_to_not_convert=[
                "vision_tower",
                "image_newline",
                "multi_modal_projector",
                "lm_head",
                "embed_tokens",
            ] + [f"model.layers.{i}.coefficient" for i in range(hf_config.text_config.num_hidden_layers)]
            + [f"model.layers.{i}.block_sparse_moe.gate" for i in range(hf_config.text_config.num_hidden_layers)]
        ),
    }
    return QUANT_TYPE_MAP[quant_type]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_type", type=str, default="default", choices=["default", "int8"])
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    return parser.parse_args()

def check_params(args, hf_config: AutoConfig):
    if args.quant_type == "int8":
        assert args.world_size >= 8, "int8 weight-only quantization requires at least 8 GPUs"

    assert hf_config.text_config.num_hidden_layers % args.world_size == 0, f"num_hidden_layers({hf_config.text_config.num_hidden_layers}) must be divisible by world_size({args.world_size})"

@torch.no_grad()
def main():
    args = parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    model_id = args.model_id

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    quantization_config = generate_quanto_config(hf_config, args.quant_type)

    check_params(args, hf_config)

    model_safetensors_index_path = os.path.join(model_id, "model.safetensors.index.json")
    with open(model_safetensors_index_path, "r") as f:
        model_safetensors_index = json.load(f)
    weight_map = model_safetensors_index['weight_map']
    vision_map = {}
    for key, value in weight_map.items():
        if 'vision_tower' in key or 'image_newline' in key or 'multi_modal_projector' in key:
            new_key = key.replace('.weight','').replace('.bias','')
            if new_key not in vision_map:
                vision_map[new_key] = value
    device_map = {
        'language_model.model.embed_tokens': 'cuda:0',
        'language_model.model.norm': f'cuda:{args.world_size - 1}',
        'language_model.lm_head': f'cuda:{args.world_size - 1}'
    }
    for key, value in vision_map.items():
        device_map[key] = f'cuda:0'
    device_map['vision_tower.vision_model.post_layernorm'] = f'cuda:0'
    layers_per_device = hf_config.text_config.num_hidden_layers // args.world_size
    for i in range(args.world_size):
        for j in range(layers_per_device):
            device_map[f'language_model.model.layers.{i * layers_per_device + j}'] = f'cuda:{i}'

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant created by Minimax based on MiniMax-VL-01 model."}]},
        {"role": "user", "content": [{"type": "image", "image": "placeholder"},{"type": "text", "text": "Describe this image."}]},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"prompt: \n{prompt}")
    raw_image = Image.open(args.image_path)
    model_inputs = processor(images=[raw_image], text=prompt, return_tensors='pt').to('cuda').to(torch.bfloat16)

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
        offload_buffers=True,
    )
    generation_config = GenerationConfig(
        max_new_tokens=100,
        eos_token_id=200020,
        use_cache=True,
    )
    generated_ids = quantized_model.generate(**model_inputs, generation_config=generation_config)
    print(f"generated_ids: {generated_ids}")
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    # The image depicts a single, whole apple with a rich, red color. The apple appears to be fresh, with a smooth, glossy skin that reflects light, indicating its juiciness. The surface of the apple is dotted with small, light-colored

def query_safetensors(path):
    safetensor = safetensors.torch.load_file(path)
    for key in safetensor.keys():
        print(key, safetensor[key].shape)
if __name__ == "__main__":
    main()