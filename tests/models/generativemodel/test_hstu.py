import os
import random

# Edit these three values to change the test input.
BATCH_SIZE = 4
HIS_SEQ_LEN = 50
CANDIDATE_NUM = 10

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


def make_prompt(uid: int) -> TokensPrompt:
    if HIS_SEQ_LEN < 2 or HIS_SEQ_LEN % 2:
        raise ValueError("HIS_SEQ_LEN must be a positive even number")

    rng = random.Random(42 + uid)
    history = []
    for _ in range(HIS_SEQ_LEN // 2):
        history.extend(
            [rng.randrange(100000), 100000 + rng.randrange(1, 1024),]
        )
    candidates = [rng.randrange(100000) for _ in range(CANDIDATE_NUM)]

    return TokensPrompt(
        prompt_token_ids=history + candidates,
        additional_information={
            "uid": [uid],
            "request_stage": 2,
            "candidate_num": [CANDIDATE_NUM],
        },
    )


def main() -> None:
    prompts = [make_prompt(uid) for uid in range(BATCH_SIZE)]
    sampling_params = [
        SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=1,
            extra_args={"gr_params": prompt["additional_information"]},
        )
        for prompt in prompts
    ]

    llm = LLM(
        # Replace this path with the HSTU model directory used on the target machine.
        model="Lingqu-Rec/HSTU-0.2B",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        max_num_batched_tokens=20480,
        compilation_config={ 
            'cudagraph_mode': 'FULL_DECODE_ONLY',
            'cudagraph_capture_sizes': [(j + 1) * 128 for j in range(32)] + [2, 4, 8, 16, 32, 64]
        },
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for req in outputs:
        print(f"Request {req.request_id}:")
        for lp_dict in req.prompt_logprobs[1:]:
            for token_id, logprob_obj in lp_dict.items():
                print(f"  {token_id}:{logprob_obj.logprob:.4f}")
        print()

if __name__ == "__main__":
    main()