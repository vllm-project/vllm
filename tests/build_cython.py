# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import Cython.Compiler.Options
from Cython.Build import cythonize
from setuptools import setup

Cython.Compiler.Options.annotate = True

infiles = []

infiles += [
    "vllm/engine/llm_engine.py",
    "vllm/transformers_utils/detokenizer.py",
    "vllm/engine/output_processor/single_step.py",
    "vllm/outputs.py",
    "vllm/engine/output_processor/stop_checker.py",
]

infiles += [
    "vllm/core/scheduler.py",
    "vllm/sequence.py",
    "vllm/core/block_manager.py",
]

infiles += [
    "vllm/model_executor/layers/sampler.py",
    "vllm/sampling_params.py",
    "vllm/utils.py",
]

setup(ext_modules=cythonize(infiles,
                            annotate=False,
                            force=True,
                            compiler_directives={
                                'language_level': "3",
                                'infer_types': True
                            }))

# example usage: python3 build_cython.py build_ext --inplace
