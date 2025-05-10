import time
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ---- Extract boxed answer ----
import re
def extract_final_answer(text: str):
    match = re.search(r"([^\s{}\\]+)}\s*\\\]", text)
    if match:
        return match.group(1).strip()
    return None


NUM_QUESTIONS = 15
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=2048,
    skip_special_tokens=True,
)

scheduler_settings = [
    {"policy": 0, "speculative": None, "early_exit": False},
    {"policy": 0, "speculative": None, "early_exit": True},
    {"policy": 1, "speculative": None, "early_exit": False},
    {"policy": 1, "speculative": None, "early_exit": True},
    {"policy": 0, "speculative": {
        "num_speculative_tokens": 5  
    }, "early_exit": False},
    {"policy": 0, "speculative": {
        "num_speculative_tokens": 5  
    }, "early_exit": True},
    {"policy": 1, "speculative": {
        "num_speculative_tokens": 5  
    }, "early_exit": False},
    {"policy": 1, "speculative": {
        "num_speculative_tokens": 5  
    }, "early_exit": True},
]

# ---- Load dataset ----
ds = load_dataset("openai/gsm8k", "main", split="test")
sample_ds = ds.select(range(NUM_QUESTIONS))
questions_df = pd.DataFrame({"qid": list(range(NUM_QUESTIONS)), "question": sample_ds["question"]})

# ---- Run experiments ----
all_results = []

for setting in scheduler_settings:
    for early_exit in [False, True]:
        llm = LLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            tensor_parallel_size=1,
            enable_prefix_caching=True,
            trust_remote_code=True,
            max_model_len=2048,
            scheduling_policy =setting['policy'],
            speculative_config=setting["speculative"],
            enable_early_exit_reasoning_model=setting["early_exit"],
        )


        for row in questions_df.itertuples():
            prompt = f"""You are a helpful and accurate math tutor. Provide step-by-step reasoning and the final answer.

Question: {row.question}
Answer:"""

            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
            duration = time.time() - start_time

            output = outputs[0].outputs[0]
            final_text = output.text.strip()
            boxed = extract_final_answer(final_text)

            all_results.append({
                "qid": row.qid,
                "scheduler_policy": setting["policy"],
                "speculative": setting["speculative"],
                "early_exit": setting["early_exit"],
                "duration": duration,
                "num_input_tokens": len(outputs[0].prompt_token_ids),
                "num_decoded_tokens": len(output.token_ids),
                "boxed_answer": boxed,
                "response": final_text,
            })

# ---- Save results ----
results_df = pd.DataFrame(all_results)
results_df.to_csv("gsm8k_scheduler_eval.csv", index=False)
print("Saved results to gsm8k_scheduler_eval.csv")

# ---- Plotting ----
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x="scheduler_policy", y="num_decoded_tokens", hue="early_exit")
plt.title("Decoded Token Count by Scheduler Policy (Early Exit vs Not)")
plt.ylabel("# of Decoded Tokens")
plt.xlabel("Scheduler Policy")
plt.grid(True)
plt.tight_layout()
plt.savefig("token_count_boxplot.png")
print("Saved chart to token_count_boxplot.png")

plt.figure(figsize=(10, 6))
sns.histplot(data=results_df[results_df.early_exit == True],
             x="boxed_answer", hue="scheduler_policy", multiple="stack", shrink=0.8)
plt.title("Final Boxed Answers Distribution (Early Exit Only)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxed_answer_histogram.png")
print("Saved chart to boxed_answer_histogram.png")
