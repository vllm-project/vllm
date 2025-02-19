# SPDX-License-Identifier: Apache-2.0
import boto3


def restructure_model_path(model_path):
    # Initialize S3 client
    s3 = boto3.client('s3')
    bucket_name = 'vllm-ci-model-weights'

    # Split the path into repo and model name
    repo, model_name = model_path.split('/', 1)

    print(f"Processing: {model_path}")
    print(f"Repo: {repo}")
    print(f"Model: {model_name}")

    try:
        # List objects in the source path
        paginator = s3.get_paginator('list_objects_v2')
        source_prefix = model_name + '/'

        # First verify the source exists
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=source_prefix)
        if 'Contents' not in response:
            print(f"Source path not found: {source_prefix}")
            return False

        # Copy objects to new location
        for page in paginator.paginate(Bucket=bucket_name,
                                       Prefix=source_prefix):
            for obj in page.get('Contents', []):
                source_key = obj['Key']
                target_key = f"{repo}/{source_key}"

                print(f"Moving: {source_key} -> {target_key}")

                # Copy object to new location
                s3.copy_object(Bucket=bucket_name,
                               CopySource={
                                   'Bucket': bucket_name,
                                   'Key': source_key
                               },
                               Key=target_key)

                # Delete original object
                s3.delete_object(Bucket=bucket_name, Key=source_key)

        print("Migration completed successfully!")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    model_paths = [
        "distilbert/distilgpt2",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "openai-community/gpt2",
        "ArthurZ/Ilama-3.2-1B",
        "llava-hf/llava-1.5-7b-hf",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "JackFram/llama-160m",
        "ai21labs/Jamba-tiny-random",
        "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
        "nm-testing/Phi-3-mini-128k-instruct-FP8",
        "nm-testing/Qwen2-0.5B-Instruct-FP8-SkipQKV",
        "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
        "nm-testing/Qwen2-1.5B-Instruct-FP8-K-V",
        "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symTrue",
        "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symFalse",
        "AMead10/Llama-3.2-1B-Instruct-AWQ",
        "shuyuej/Llama-3.2-1B-Instruct-GPTQ",
        "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head",
        "ModelCloud/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit-10-25-2024",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
        "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
        "amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test",
        "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change",
        "nm-testing/tinyllama-oneshot-w8-channel-a8-tensor",
        "nm-testing/asym-w8w8-int8-static-per-tensor-tiny-llama",
        "neuralmagic/Llama-3.2-1B-quantized.w8a8",
        "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Asym",
        "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Static-Per-Tensor-Sym",
        "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Static-Per-Tensor-Asym",
        "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change",
        "nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2",
        "nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2-asym",
        "nm-testing/tinyllama-oneshot-w8a8-channel-dynamic-token-v2",
        "nm-testing/tinyllama-oneshot-w8a8-channel-dynamic-token-v2-asym",
        "nm-testing/tinyllama-oneshot-w4a16-channel-v2",
        "nm-testing/tinyllama-oneshot-w4a16-group128-v2",
        "nm-testing/tinyllama-oneshot-w8a16-per-channel",
        "nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t",
        "nm-testing/Meta-Llama-3-8B-FP8-compressed-tensors-test",
        "nm-testing/TinyLlama-1.1B-compressed-tensors-kv-cache-scheme",
        "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Dynamic-2of4-testing",
        "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Static-Per-Tensor-testing",
        "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Static-testing",
        "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Dynamic-IA-Per-Tensor-Weight-testing",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_per_tok_dyn_act_fp8-BitM",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_tensor_act_fp8-BitM",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_per_tok_dyn_act_fp8-BitM",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_tensor_act_fp8-BitM",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_per_tok_dyn_act_int8-BitM",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_tensor_act_int8-BitM",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_per_tok_dyn_act_int8-BitM",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_tensor_act_int8-BitM",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-INT8-Dynamic-IA-Per-Channel-Weight-testing",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-INT8-Static-testing",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-INT8-Dynamic-IA-Per-Tensor-Weight-testing",
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-2of4-Sparse-Dense-Compressor",
    ]
    for model_path in model_paths:
        restructure_model_path(model_path)
