# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fresh, provenance-checked HF reference in an isolated Python process."""

import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any

HF_VERSIONS = {"transformers": "5.16.1", "huggingface-hub": "1.33.0"}
SHARED_PACKAGES = ("torch", "torchvision", "numpy", "tokenizers", "safetensors")
PROTOCOL = {
    "dtype": "bfloat16",
    "attn_implementation": "eager",
    "use_cache": True,
    "do_sample": False,
    "seed": 0,
}
REPO = Path(__file__).resolve().parents[3]
GENERATION_TEST = REPO / "tests/models/language/generation/test_ncp_olmo.py"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_hashes(model: Path) -> dict[str, str]:
    return {
        path.name: file_hash(path)
        for path in sorted(model.iterdir())
        if path.suffix in {".py", ".json", ".model", ".tiktoken"}
        and "manifest" not in path.name
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def validate_reference(request: dict, result: dict, digest: str) -> list:
    assert result["request_sha256"] == digest
    assert result["nonce"] == request["nonce"]
    assert result["source_sha256"] == request["source_sha256"]
    assert result["test_sha256"] == request["test_sha256"]
    assert result["commit"] == request["commit"]
    assert result["runtime_hashes"] == request["runtime_hashes"]
    assert result["protocol"] == request["protocol"] == PROTOCOL
    assert result["shared_versions"] == request["shared_versions"]
    assert result["hf_versions"] == HF_VERSIONS
    assert len(result["input_ids"]) == len(request["prompts"])
    assert len(result["outputs"]) == len(request["prompts"])
    outputs = []
    for input_ids, record in zip(result["input_ids"], result["outputs"], strict=True):
        assert input_ids and all(type(token) is int for token in input_ids)
        ids, text, steps = record
        assert 0 < len(ids) == len(steps) <= request["max_tokens"]
        assert all(type(token) is int for token in ids)
        logprobs = []
        for token, pairs in zip(ids, steps, strict=True):
            assert len(pairs) == request["num_logprobs"]
            assert all(
                type(key) is int and math.isfinite(value) for key, value in pairs
            )
            step = dict(pairs)
            assert len(step) == len(pairs) and token in step
            logprobs.append(step)
        outputs.append((ids, text, logprobs))
    return outputs


def generate_reference(
    model: str, prompts: list[str], max_tokens: int, num_logprobs: int, output_dir: Path
) -> tuple[list, list[list[int]]]:
    """Run the unchanged public checkpoint with the pinned HF generation API."""
    python = os.environ.get("NCP_OLMO_HF_PYTHON")
    if not python:
        raise RuntimeError(
            "Set NCP_OLMO_HF_PYTHON to the isolated HF interpreter; "
            "see docs/models/ncp_olmo.md or .buildkite/scripts/run-ncp-olmo.sh."
        )
    model_path = Path(model).resolve()
    assert model_path.is_dir()
    source = Path(__file__).resolve()
    request = {
        "nonce": uuid.uuid4().hex,
        "model": str(model_path),
        "source_sha256": file_hash(source),
        "test_sha256": file_hash(GENERATION_TEST),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, timeout=10
        ).strip(),
        "runtime_hashes": runtime_hashes(model_path),
        "protocol": PROTOCOL,
        "shared_versions": {name: version(name) for name in SHARED_PACKAGES},
        "prompts": prompts,
        "max_tokens": max_tokens,
        "num_logprobs": num_logprobs,
    }
    request_path, result_path = (
        output_dir / "request.json",
        output_dir / "reference.json",
    )
    write_json(request_path, request)
    subprocess.run(
        [python, "-I", str(source), str(request_path), str(result_path)],
        check=True,
        timeout=900,
    )
    result = json.loads(result_path.read_text())
    assert file_hash(source) == request["source_sha256"]
    assert file_hash(GENERATION_TEST) == request["test_sha256"]
    assert runtime_hashes(model_path) == request["runtime_hashes"]
    return validate_reference(request, result, file_hash(request_path)), result[
        "input_ids"
    ]


def _generate_hf_greedy_logprobs(
    hf_model: Any,
    prompts: list[str],
    max_tokens: int,
    num_logprobs: int,
) -> list[tuple[list[int], str, list[dict[int, float]]]]:
    """Collect cached HF generation scores without requesting hidden states."""
    import torch

    outputs = []
    for inputs in hf_model.get_inputs(prompts):
        generated = hf_model.model.generate(
            **hf_model.wrap_device(inputs),
            tokenizer=hf_model.tokenizer,
            use_cache=True,
            do_sample=False,
            max_new_tokens=max_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
        scores = generated.scores
        output_ids = generated.sequences[0, -len(scores) :].tolist() if scores else []
        output_logprobs = []
        for score in scores:
            logprobs = torch.log_softmax(score[0].float(), dim=-1)
            topk = logprobs.topk(num_logprobs)
            output_logprobs.append(
                dict(zip(topk.indices.tolist(), topk.values.tolist(), strict=True))
            )
        outputs.append(
            (output_ids, hf_model.tokenizer.decode(output_ids), output_logprobs)
        )
    return outputs


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    assert request["protocol"] == PROTOCOL
    assert {name: version(name) for name in HF_VERSIONS} == HF_VERSIONS
    assert {name: version(name) for name in SHARED_PACKAGES} == request[
        "shared_versions"
    ]
    assert file_hash(Path(__file__)) == request["source_sha256"]
    assert file_hash(GENERATION_TEST) == request["test_sha256"]
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, timeout=10
        ).strip()
        == request["commit"]
    )
    assert runtime_hashes(Path(request["model"])) == request["runtime_hashes"]
    assert torch.cuda.is_available()
    torch.manual_seed(PROTOCOL["seed"])

    class HFTextReference:
        def __init__(self):
            model = AutoModelForCausalLM.from_pretrained(
                request["model"],
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            # Match HfRunner: preserve FP32 buffers when parameters already
            # have the requested dtype.
            if getattr(model, "quantization_method", None) is None and any(
                parameter.dtype != torch.bfloat16 for parameter in model.parameters()
            ):
                model = model.to(dtype=torch.bfloat16)
            self.model = model.to(device="cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(
                request["model"], trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                request["model"], trust_remote_code=True
            )
            self.input_ids = []

        def get_inputs(self, prompts):
            inputs = [self.processor(text=p, return_tensors="pt") for p in prompts]
            self.input_ids = [item["input_ids"][0].tolist() for item in inputs]
            return inputs

        def wrap_device(self, inputs):
            return inputs.to("cuda")

    model = HFTextReference()
    outputs = _generate_hf_greedy_logprobs(
        model, request["prompts"], request["max_tokens"], request["num_logprobs"]
    )
    result = {
        "nonce": request["nonce"],
        "request_sha256": file_hash(request_path),
        "source_sha256": file_hash(Path(__file__)),
        "test_sha256": file_hash(GENERATION_TEST),
        "commit": request["commit"],
        "runtime_hashes": runtime_hashes(Path(request["model"])),
        "protocol": PROTOCOL,
        "shared_versions": {name: version(name) for name in SHARED_PACKAGES},
        "hf_versions": {name: version(name) for name in HF_VERSIONS},
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "cuda": torch.version.cuda,
        },
        "input_ids": model.input_ids,
        "outputs": [
            (ids, text, [list(step.items()) for step in steps])
            for ids, text, steps in outputs
        ],
    }
    validate_reference(request, result, file_hash(request_path))
    assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)
    write_json(output_path, result)
    print("NCP_HF_REFERENCE_OK", flush=True)


if __name__ == "__main__":
    main()
