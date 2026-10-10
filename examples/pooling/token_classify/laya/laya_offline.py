# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Answer typed questions with a Laya checkpoint converted by
`convert_checkpoint.py`.

    python laya_offline.py --model ./laya-vllm
"""

import argparse
import json

from laya_prompts import (
    build_prompt,
    check_question,
    decode_answer,
    normalize_question,
)

from vllm import LLM

STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate "
    "today or we will cancel our plan.",
}
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./laya-vllm")
    args = parser.parse_args()

    llm = LLM(model=args.model, runner="pooling")
    tokenizer = llm.get_tokenizer()
    laya_config = llm.llm_engine.model_config.hf_config.laya_config

    for qid, qdef in QUESTIONS.items():
        check_question(qid, qdef)
    questions = {qid: normalize_question(q) for qid, q in QUESTIONS.items()}
    prompts = [
        {
            "prompt_token_ids": build_prompt(
                tokenizer,
                STATE,
                q,
                laya_config["max_len"],
                laya_config["head_max_len"],
            )
        }
        for q in questions.values()
    ]

    outputs = llm.encode(prompts, pooling_task="token_classify")
    answers = {
        qid: decode_answer(q, output.outputs.data.tolist())
        for (qid, q), output in zip(questions.items(), outputs)
    }
    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
