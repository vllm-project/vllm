from vllm.config.model import ModelConfig, _find_dtype
import json

def main():
    trust_remote_code_models = [
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        "XiaomiMiMo/MiMo-7B-RL",
        # "FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1", # is not available online right now
        "meituan-longcat/LongCat-Flash-Chat",
    ]
    models_to_test = [
        "Zyphra/Zamba2-7B-instruct",
        "mosaicml/mpt-7b",
        "databricks/dbrx-instruct",
        "tiiuae/falcon-7b",
        "tiiuae/falcon-40b",
        "luccafong/deepseek_mtp_main_random",
        "luccafong/deepseek_mtp_draft_random",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "tiny-random/qwen3-next-moe",
        "zai-org/GLM-4.5",
        "baidu/ERNIE-4.5-21B-A3B-PT",
        # Select some models using base convertor for testing
        "lmsys/gpt-oss-20b-bf16",
        "deepseek-ai/DeepSeek-V3.2-Exp",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ] + trust_remote_code_models
    all_res = {}
    for model in models_to_test:
        print(f"testing {model=}")
        model_config = ModelConfig(model, trust_remote_code=model in trust_remote_code_models)
        res = {}
        hf_config = model_config.hf_config
        hf_text_config = model_config.hf_text_config
        res["architectures"] = model_config.architectures
        res["model_type"] = hf_config.model_type
        res["text_model_type"] = getattr(hf_text_config, "model_type", None)
        res["hidden_size"] = model_config.get_hidden_size()
        res["total_num_hidden_layers"] = model_config.get_total_num_hidden_layers()
        res["total_num_attention_heads"] = getattr(hf_text_config, "num_attention_heads", 0)
        res["head_size"] = model_config.get_head_size()
        res["vocab_size"] = model_config.get_vocab_size()
        res["total_num_kv_heads"] = model_config.get_total_num_kv_heads()
        res["num_experts"] = model_config.get_num_experts()

        res["is_deepseek_mla"] = model_config.is_deepseek_mla
        res["is_multimodal_model"] = model_config.is_multimodal_model
        dtype = _find_dtype(model, hf_config, revision=model_config.revision)
        res["dtype"] = str(dtype)
        res["is_dtype_str"] = isinstance(dtype, str)
        all_res[model] = res
    
    with open("model_arch_groundtruth.json", "w") as f:
        json.dump(all_res, f, indent=4)


if __name__ == "__main__":
    main()