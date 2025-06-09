import base64
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
model_name = "/model/glm-4v-9b-0603"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def run_chat():
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"images": 2,"videos": 1},
        gpu_memory_utilization=0.7,
        allowed_local_media_path='/',
        enforce_eager=True,
        enable_prefix_caching=False,
        dtype="bfloat16",
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        stop_token_ids=[151329, 151348, 151336],
        n=1,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        max_tokens=8192,
        prompt_logprobs=True,
        seed=42,
    )

    messages = [
        # {
        #     "role": "user",
        #     "content": [
        #         {
        #             "type": "image_url",
        #             "image_url": {
        #                 "url": f"data:image/jpeg;base64,{base64.b64encode(open('test_v2.png', 'rb').read()).decode('utf-8')}"
        #             },
        #         },
        #         {
        #             "type": "text",
        #             "text": "图中题目的答案是_____。",
        #         },
        #     ],
        # }
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": 'file:///mnt/vllm/test.mp4'
                    },
                },
                {
                    "type": "text",
                    "text": "详细描述一下这个视频",
                }
            ],
        }
    ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": "你是谁"
    #     }
    # ]
    outputs = llm.chat(
        messages,
        sampling_params=sampling_params,
    )
    print('=' * 20)
    for o in outputs[0].outputs:
        generated_text = o.text
        print(generated_text)
        print('=' * 20)
    print('=' * 20)

if __name__ == "__main__":
    run_chat()