from vllm import LLM, SamplingParams

import argparse
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = "/data/models/DeepSeek-R1/"
#model_path = "/mnt/workdisk/dohayon/Projects/R1/DeepSeek-R1-fp8/"
# model_path = "deepseek-ai/DeepSeek-V2-Lite"
model_path = "/mnt/disk5/hf_models/DeepSeek-R1-BF16"
# model_path = "/mnt/disk2/hf_models/DeepSeek-R1-G2/"
# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=model_path, help="The model path.")
parser.add_argument("--task", type=str, default="gsm8k", help="The model path.")
parser.add_argument("--tokenizer", type=str, default=model_path, help="The model path.")
parser.add_argument("--tp_size", type=int, default=8, help="Tensor Parallelism size.")
parser.add_argument("--ep_size", type=int, default=8, help="Expert Parallelism size.")
parser.add_argument("-l", "--limit", type=int, default=None, help="test request counts.")
args = parser.parse_args()

os.environ["VLLM_EP_SIZE"] = f"{args.ep_size}"
os.environ["VLLM_TP_SIZE"] = f"{args.tp_size}"
os.environ["VLLM_SKIP_WARMUP"] = "true"

# os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
# os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
# if args.ep_size > 1:
#     os.environ["VLLM_MOE_N_SLICE"] = "1"
#     os.environ["VLLM_EP_SIZE"] = f"{args.ep_size}"
# else:
#     os.environ["VLLM_MOE_N_SLICE"] = "4"
#     os.environ["VLLM_EP_SIZE"] = "1"

# os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"
# os.environ["PT_HPU_WEIGHT_SHARING"] = "0"

if __name__ == "__main__":

    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval import simple_evaluate

    model = args.model
    if args.tp_size == 1:
        llm = VLLM(
            pretrained=model, 
            tokenizer=args.tokenizer,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.8,
        )
    else:
        llm = VLLM(
            pretrained=model, 
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tp_size,
            distributed_executor_backend='mp',
            quantization="inc_q",
            weights_load_device="cpu",
            trust_remote_code=True,
            max_model_len=4096,
            dtype="bfloat16",
            max_num_seqs=1,
            gpu_memory_utilization=0.96,
        )

    
    # Run the evaluation; you can adjust num_fewshot and batch_size as needed.
    if args.task == "gsm8k":
        print("============ Start Evaluation ============")
        results = simple_evaluate(
            model=llm,
            tasks=["gsm8k"],
            num_fewshot=5,
            batch_size=1,
            limit=args.limit,
        )
        # save as json
        with open(f"gsm8k_ep{args.ep_size}_result_samples.jsonl", "w") as f:
            json.dump(results['results'], f)
            f.write("\n")
            for sample in results['samples']['gsm8k']:
                json.dump(sample, f)
                f.write("\n")
    elif args.task == "hellaswag":
        results = simple_evaluate(
            model=llm,
            tasks=["hellaswag"],
            num_fewshot=0,
            batch_size=8,
            limit=args.limit,
        )
        with open(f"hellaswag_ep{args.ep_size}_result_samples.jsonl", "w") as f:
            json.dump(results['results'], f)
            f.write("\n")
            for sample in results['samples']['hellaswag']:
                json.dump(sample, f)
                f.write("\n")
    elif args.task == "drop":
        results = simple_evaluate(
            model=llm,
            tasks=["drop"],
            num_fewshot=3,
            batch_size=1,
            limit=args.limit,
        )
        with open(f"drop_ep{args.ep_size}_result_samples.jsonl", "w") as f:
            json.dump(results['results'], f)
            f.write("\n")
            for sample in results['samples']['drop']:
                json.dump(sample, f)
                f.write("\n")
    else:
        tasks = args.task.split(",")
        assert isinstance(tasks, list), f"tasks must be a list of strings, got {type(tasks)}"
        tasks_name = "_".join(tasks)
        print(f"============ Start Evaluation {tasks}============")
        results = simple_evaluate(
            model=llm,
            tasks=tasks,
            num_fewshot=0,
            batch_size=1,
            limit=args.limit,
        )
        print("Evaluation Results Table: ")
        from lm_eval.utils import make_table
        print(make_table(results))
        print(f"============ Completed {tasks} ============")
        with open(f"{tasks_name}_ep{args.ep_size}_result_samples.jsonl", "w") as f:
            json.dump(results['results'], f)
            f.write("\n")
            for sample in results['samples'][tasks_name]:
                json.dump(sample, f)
                f.write("\n")
            
    
    del llm
    print("============ Completed ============")
    
    # Print out the results.
    print("Evaluation Results:")
    for task, metrics in results['results'].items():
        print(f"{task}: {metrics}")

