# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve StartLux's text decision protocol using native vLLM pooling.

Put the pinned StartLux-Decision source on PYTHONPATH for its prompt renderer
and answer protocol. No CUDA model or graph is constructed by this adapter.
"""

import argparse
import json
from pathlib import Path

import torch
from startlux_decision import StartLuxDecision, jevfmt
from transformers import AutoConfig, AutoTokenizer

from vllm import LLM


class VLLMStartLuxDecision(StartLuxDecision):
    def __init__(self, checkpoint: str, **llm_kwargs):
        config = json.loads((Path(checkpoint) / "decision_config.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.letters = jevfmt.check_tokenizer(self.tok)
        if self.letters != config["letter_token_ids"]:
            raise ValueError("Checkpoint and tokenizer letter IDs disagree")
        self.temperature = {
            kind: float(value) for kind, value in config["temperature_by_type"].items()
        }
        if any(not 0 < value < float("inf") for value in self.temperature.values()):
            raise ValueError("Temperatures must be finite and positive")
        wide = config["wide_choice"]
        self.group, self.keep, self.residual = (
            wide["group"],
            wide["keep"],
            wide["residual"],
        )
        self.max_length = llm_kwargs.get("max_model_len", 8192)
        self._kept = self._media = None
        hf_config = AutoConfig.from_pretrained(checkpoint)
        moe = getattr(hf_config.get_text_config(), "num_experts", 0)
        architecture = (
            "StartLuxDecisionMoeForSequenceClassification"
            if moe
            else "StartLuxDecisionForSequenceClassification"
        )
        self.llm = LLM(
            model=checkpoint,
            runner="pooling",
            hf_overrides={"architectures": [architecture], "num_labels": 26},
            pooler_config={"task": "classify", "use_activation": False},
            dtype="bfloat16",
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            **llm_kwargs,
        )

    def _logits(self, rows):
        encoded = [
            jevfmt.render_ids(row, self.tok, max_length=self.max_length)[0]
            for row in rows
        ]
        outputs = self.llm.encode(
            [{"prompt_token_ids": ids} for ids in encoded],
            pooling_task="classify",
            use_tqdm=False,
        )
        logits = []
        for row, output in zip(rows, outputs):
            values = output.outputs.data.float().cpu()
            if values.shape != (len(self.letters),) or not torch.isfinite(values).all():
                raise RuntimeError("Invalid StartLux letter logits")
            logits.append(values[: len(row["options"])])
        return logits, sum(map(len, encoded))

    def decide(self, state, questions, images=None):
        if images:
            raise ValueError("This text serving example does not accept images")
        return super().decide(state, questions)


def main():
    import threading

    import uvicorn
    from fastapi import FastAPI, HTTPException

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--port", type=int, default=18190)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    engine = VLLMStartLuxDecision(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=16,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=0.75,
        enforce_eager=args.enforce_eager,
        compilation_config=(
            None
            if args.enforce_eager
            else {
                "cudagraph_mode": "PIECEWISE",
                "cudagraph_capture_sizes": [256, 384, 512, 768, 1024, 2048, 4096],
            }
        ),
    )
    app = FastAPI()
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/systemone")
    def decide(request: dict):
        try:
            with lock:
                answers, usage = engine.decide(
                    request.get("state"), request["questions"], request.get("images")
                )
            return {"answers": answers, "usage": usage}
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
