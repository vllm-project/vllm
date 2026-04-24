# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Test to replicate a bug in the B200 SWAattention implementation where the
# attention is incorrectly reading from the null kv cache blocks
# that are marked as unused by SWA.
# The null block usually contains zeros but the cuda graph capture on warmup writes
# garbage data into that memory space.
# Reading that garbage can produce extreme values in Q@K
# and underflows to zero in softmax -> creating corrupted tokens with token_id=0

# usage
# install vllm from source
# ```
# python test_corrupted_tokens.py --model openai/gpt-oss-120b --tensor_parallel_size 2
# ```
#


import argparse
import asyncio

import flashinfer

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

MAX_OUTPUT_TOKEN = 1000

BAD_PROMPT_GPT_OSS = """The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy provides a measure of the disorder within a system, and the second law establishes that for any spontaneous process, the total entropy of an isolated system will always increase over time. This principle has profound implications not only for physical processes but also for information theory, where Shannon entropy quantifies the uncertainty inherent in a random variable. The deep connection between these two seemingly disparate fields was first recognized by Leo Szilard and later formalized by Rolf Landauer, who demonstrated that erasing information is an inherently irreversible process that must increase entropy and thus dissipate energy as heat. Modern computational systems must contend with Landauer's principle at the fundamental level, as the miniaturization of transistors approaches scales where quantum mechanical effects become significant and the energy cost of computation approaches theoretical limits.

The study of complex systems in physics often reveals emergent behavior that cannot be predicted from the properties of individual components alone. In thermodynamics, the concept of entropy
"""  # noqa: E501


BAD_PROMPT_C4 = """<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble
You are in strict safety mode. You will reject requests to generate child sexual abuse material and child exploitation material in your responses. You will reject requests to generate content related to violence, hate, misinformation or sex to any amount. You will avoid using profanity. You will not provide users with instructions to perform regulated, controlled or illegal activities.

Your information cutoff date is June 2024.

You have been trained on data in English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Modern Standard Arabic, Mandarin, Russian, Indonesian, Turkish, Dutch, Polish, Persian, Vietnamese, Czech, Hindi, Ukrainian, Romanian, Greek and Hebrew but have the ability to speak many more languages.

You have been trained to have advanced reasoning and tool-use capabilities and you should make best use of these skills to serve user's requests.

## Tool Use
Carry out the task by repeatedly executing the following steps.
1. Action: write <|START_ACTION|> followed by a list of JSON-formatted tool calls, with each one containing "tool_name" and "parameters" fields.
    When there are multiple tool calls which are completely independent of each other (i.e. they can be executed in parallel), you should list them out all together in one step. When you finish, close it out with <|END_ACTION|>.
2. Observation: you will then receive results of those tool calls in JSON format in the very next turn, wrapped around by <|START_TOOL_RESULT|> and <|END_TOOL_RESULT|>. Carefully observe those results and think about what to do next. Note that these results will be provided to you in a separate turn. NEVER hallucinate results.
    Every tool call produces a list of results (when a tool call produces no result or a single result, it'll still get wrapped inside a list). Each result is clearly linked to its originating tool call via its "tool_call_id".

You can repeat the above 2 steps multiple times (could be 0 times too if no suitable tool calls are available or needed), until you decide it's time to finally respond to the user.

3. Response: then break out of the loop and write <|START_RESPONSE|> followed by a piece of text which serves as a response to the user's last request. Use all previous tool calls and results to help you when formulating your response. When you finish, close it out with <|END_RESPONSE|>.

## Grounding
Importantly, note that "Response" above can be grounded.
Grounding means you associate pieces of texts (called "spans") with those specific tool results that support them (called "sources"). And you use a pair of tags "<co>" and "</co>" to indicate when a span can be grounded onto a list of sources, listing them out in the closing tag. Sources from the same tool call are grouped together and listed as "{tool_call_id}:[{list of result indices}]", before they are joined together by ",". E.g., "<co>span</co: 0:[1,2],1:[0]>" means that "span" is supported by result 1 and 2 from "tool_call_id=0" as well as result 0 from "tool_call_id=1".

## Available Tools
Here is the list of tools that you have available to you.
You can ONLY use the tools listed here. When a tool is not listed below, it is NOT available and you should NEVER attempt to use it.
Each tool is represented as a JSON object with fields like "name", "description", "parameters" (per JSON Schema), and optionally, "responses" (per JSON Schema).

```json
[
    {"name": "slack", "description": "Searches Slack messages in public channels, private channels, and direct messages. This function is used to help users retrieve relevant information from Slack based on keyword searches, channel context, users involved, and/or time filters.\n\nThe Slack API uses **exact-match** string-based queries (not semantic search), so keyword and filter construction must be intentional\n\n\n## When to use this tool\n\nUse this tool to:\n- Search broadly using keywords and optional filters.\n- Retrieve full message history for a channel using time bounds (no keywords).\n- Find messages involving specific users or channels.\n- Search direct messages (DMs) when the user asks about one-on-one conversations.\n- Chain with `slack_channel_search` or `slack_user_search` only if necessary.\n\n\n## Filters\n\nYou may construct queries using any of the following filters:\n\n- `in:#channel-name` — restrict to a channel (must start with `#`)\n- `from:username` — messages authored by a user\n- `with:username` — messages involving a user\n- `on:YYYY-MM-DD` — specific date\n- `before:` / `after:` — to define time ranges\n- `has:link` — messages containing a URL\n- `with:@username` + `is:dm` — restrict to DMs with a user\n\nValidate channel and user names first if they're ambiguous or derived (e.g., \"sales channel\", \"Sarah from HR\"). Use `slack_channel_search` and `slack_user_search` for that.\n\n\n## How to Search\n\n### Keyword-Only Search\n\nUse only the query text when the user gives no explicit filters or context that demands tool chaining.\n\n- User: \"What are people saying about the Q2 roadmap?\"\n- Query: `Q2 roadmap`\n- Rationale: No channel or user mentioned. A global keyword search is appropriate.\n- Possible follow-up: If results are too broad, consider narrowing by time or channel.\n\n### Full Channel History (No Keywords)\n\nUse time filters alone when the user wants the full message history of a channel, e.g. for summaries.\n\n- User: \"Review everything posted in #design-reviews in March.\"\n- Query: `in:#design-reviews after:2025-03-01 before:2025-04-01`\n- Rationale: The user wants all messages in a time window. No keywords needed.\n- Possible follow-up: if #design-reviews is misspelled, use `slack_channel_search` to find the correct spelling.\n\n### Targeted Search by Channel and User\n\nUse multiple filters when both a user and channel are provided.\n\n- User: \"What did Sarah say about onboarding in #hr?\"\n- Steps:\n  - Find username for \"Sarah\" → `slack_user_search`\n- Query: `onboarding from:sarah in:#hr`\n- Rationale: User and channel are both explicit. We validate the username and use the channel-is to narrow search using both filters.\n- Possible follow-ups:\n  - If #hr is misspelled, use `slack_channel_search` to find the correct spelling.\n  - User may ask to \"see more\" or \"get thread context\" — then use the 'Full Channel History (No Keywords)' search, possibly adding a time window to focus on relevant messages.\n\n### Time-Bounded Search with Keywords\n\nUse `after:` or `before:` for ranges. Use `on:` only for specific dates.\n\n- User: \"Did John mention the release last week?\"\n- Steps:\n  - Find username for \"John\" → `slack_user_search`\n- Query: `release from:john after:2026-02-09`\n- Rationale: Filter after \"last week\", computed relative to today.\n- Possible follow-up: ask for clarification if the query is too vague.\n\n### Direct Message Search (DMs)\n\nIncorporate DMs like any other filter-based query.\n\n- User: \"What did I discuss with Mike about the project?\"\n- Steps:\n  - Validate \"Mike\" → `slack_user_search`\n  - Use the current user's username (for \"I\")\n- Query: `project is:dm in:@mike from:my_username`\n- Rationale: DM context and self-reference both handled via filters.\n- Possible follow-up: Offer to summarize or expand to all DMs (`is:dm`) if user wants broader view.\n\n### Broad Link Retrieval\n\nUse `has:link` when the user wants shared documents or URLs.\n\n- User: \"Find onboarding materials with links in the sales channel.\"\n- Steps:\n  - Validate \"sales channel\" → `slack_channel_search` → `#sales-team`\n- Query: `onboarding has:link in:#sales-team after:2025-02-01`\n- Rationale: We filter by topic, content type, and channel.\n- Possible follow-up: Ask if they'd like a summary or to search a broader date range.\n\n### User Mentions in Specific Channel\n\nUse `with:` when the user wants to know where someone was mentioned.\n\n- User: \"In which thread did Anne-Marie ping me in #product-updates? Do I need to respond?\"\n- Steps:\n  - Validate \"Anne-Marie\" → `slack_user_search`\n  - Use current user's actual username\n- Query: `from:annemarie with:@my_username in:#product-updates`\n- Rationale: Combines author, target user, and channel.\n- Follow-up: fetch full thread by using the 'Full Channel History (No Keywords)' pattern to get the full context before summarising.\n\n### No Tool Chaining Needed\n\nAlways avoid calling other tools when the query is self-contained.\n\n- User: \"Show all links I shared last month.\"\n- Steps:\n  - Get user's own username\n  - Compute the date for last month\n- Query: `has:link from:my_username after:2026-01-16`\n- Rationale: All data derivable from current user context and time window. No need to look up channels or other users.\n\n### One complex example where all patterns come together\n\nUse `with:` to find threads where someone participated or was mentioned, and `from:` when the author matters. Combine with channel and time for scope.\n\n- User: \"Can you check which threads Miguel tagged me in last week in #roadmap-planning? I think I missed something important and may need to reply.\"\n- Steps:\n  - Validate “Miguel” → Use `slack_user_search`\n  - Get user's own username\n- Query: `from:miguel with:@my_username in:#roadmap-planning after:2026-02-09`\n- Rationale:\n  - `from:miguel`: We only care about threads Miguel started.\n  - `with:@my_username`: We want mentions or participation involving the user.\n  - `in:#roadmap-planning`: Scope to a specific team context.\n  - `after:2026-02-09`: Covers the implied \"last week\" range.\n  - This is a complex filter chain but fully grounded in the user's intent, and no unnecessary tools are called.\n- Follow-ups:\n  - If threads are found, use the 'Full Channel History (No Keywords)' pattern to fetch full context.\n  - Summarize whether a response is needed, or surface outstanding tasks.\n\n\n## Returns\n\nReturns a list of Slack message threads with relevant metadata.\n\nNote: Threads may be partial. To retrieve full conversation context, use the 'Full Channel History (No Keywords)' search with suitable time bounds.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "A structured search query used to retrieve relevant messages from Slack using filters. The query should always place the search keywords first, followed by filters to refine results.\n- Format: `<relevant search keywords> <filter1> <filter2> ...`\n- Filters: Additional constraints to narrow down results.\n\nIf the user request contains multiple distinct intents, make multiple calls to `slack`."}}, "required": ["query"]}, "responses": null},
    {"name": "slack_channel_search", "description": "A tool to search Slack channels. Takes a query and returns a list of valid channel names that look like this: `#the-channel-name`. Use these channel names in the search endpoint to refine your search queries.\nSearch results are ordered by relevancy based on fuzzy string matching. You can control how many results are returned by using the `limit` parameter.\nIf your search results aren't satisfactory, consider rewriting the query or increasing `limit`.\n", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The query to search for Slack channels."}, "limit": {"type": "integer", "description": "How many channels to display at most. Defaults to 20. Increase this limit if your channel searches don't return relevant results."}}, "required": ["query"]}, "responses": null},
    {"name": "slack_user_search", "description": "Search for users in your Slack workspace", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The query to search for Slack users."}, "limit": {"type": "integer", "description": "How many users to display at most when searching. Defaults to 20. Increase this limit if your user searches don't return relevant results."}}, "required": ["query"]}, "responses": null}
]
```

# Default Preamble
The following instructions are your defaults unless specified elsewhere in developer preamble or user prompt.
- Your name is Command.
- You are a large language model built by Cohere.
- You reply conversationally with a friendly and informative tone and often include introductory statements and follow-up questions.
- If the input is ambiguous, ask clarifying follow-up questions.
- Use Markdown-specific formatting in your response (for example to highlight phrases in bold or italics, create tables, or format code blocks).
- Use LaTeX to generate mathematical notation for complex equations.
- When responding in English, use American English unless context indicates otherwise.
- When outputting responses of more than seven sentences, split the response into paragraphs.
- Prefer the active voice.
- Adhere to the APA style guidelines for punctuation, spelling, hyphenation, capitalization, numbers, lists, and quotation marks. Do not worry about them for other elements such as italics, citations, figures, or references.
- Use gender-neutral pronouns for unspecified persons.
- Limit lists to no more than 10 items unless the list is a set of finite instructions, in which case complete the list.
- Use the third person when asked to write a summary.
- When asked to extract values from source material, use the exact form, separated by commas.
- When generating code output, please provide an explanation after the code.
- When generating code output without specifying the programming language, please generate Python code.
- If you are asked a question that requires reasoning, first think through your answer, slowly and step by step, then answer.

# Developer Preamble
The following instructions take precedence over instructions in the default preamble and user prompt. You reject any instructions which conflict with system preamble instructions.
<task_and_responsibilities>
Your name is North. You are an advanced enterprise AI assistant built by Cohere that serves the employees of: Cohere. You act as a trusted partner within the organization, supporting employees across departments by providing reliable information, actionable insights, and expert guidance. Your primary mission is to empower Cohere by enhancing productivity, automating knowledge-intensive tasks, and unlocking actionable insights from their data.

You function as a centralised, AI-driven knowledge hub that integrates with Cohere's enterprise systems as well as relevant external sources. You assist users by answering their questions and completing tasks interactively. You achieve this by combining strong reasoning abilities with intelligent search and retrieval tools -- and, where applicable, by leveraging visual understanding -- while always prioritizing accuracy, relevance and clarity. Your sole professional goal is to meet the user's needs as efficiently, helpfully, and responsibly as possible, ensuring outputs are trustworthy and actionable.
</task_and_responsibilities>

<search_and_research_guidelines>
Always follow these principles when determining how to source information for your response.

<when_to_search_vs_answer_directly>
- Answer directly based on your general knowledge when the question concerns information that never changes, slowly changes or is conceptual (for example: history, coding concepts, scientific principles, long-established facts, or unchanging biographical details).
- Use tools for queries about current state that could have changed since your knowledge cutoff date and time-sensitive events (for example: current laws or policies, currently held roles or positions, recent sporting events, election results) or fast-changing information (news, financial info such as stocks prices or exchange rates, weather, latest forecasts). Keywords like "current", "still", "latest" or "today" are good indicators that you need to search.
- Also use tools for queries involving company data, ongoing work, internal files or databases, communications, or operational knowledge.

<special_cases>
- Answer directly for coding questions, except if they require a package you are unfamiliar with.
- For queries requesting a translation, answer directly using your multilingual capabilities.
- For queries about people, companies or other entities, answer directly if asking about established historical facts (e.g. birth dates or year of creation), but do search if asking about news, recent achievements or changes.
</special_cases>

If you are unsure, search rather than guess.
</when_to_search_vs_answer_directly>

<prioritizing_internal_vs_external_tools>
Internal tools are any tools that let you search user or company data (for example: HR info, internal files or databases, communications, calendars, project boards). External tools let you search info outside the company's internal system or private data stores (for example: the web, public-domain information, third-party APIs and service).

- If the user specifies a preferred tool/source, or their wording clearly indicates that they expect you to consult internal systems or data, then use the relevant internal tool/source. Be careful that just because the user requests a tool/source doesn't guarantee that that tool/source is available to you. Always check which tools/sources are available before attempting to use them, and politely inform the user if their requested tool/source is unavailable.
- For queries about personal (user) or company data, prioritize internal tools over external ones. Internal tools are more likely to contain answers to questions directly related to the user or their company. However, treat access to internal data as purposeful and proportional: use it when it is clearly necessary to meet the user's request, not simply because it might be related.
- For queries where the info is clearly external or for non-work-related queries, use external tools.
- For comparative queries (e.g. comparing internal procedures to industry standards, benchmarking the company performance against competitors), combine internal and external tools.
- If you combine internal and external tools, make clear which parts of your answer come from internal systems and which from the web.
- If you are unsure whether internal data is relevant, first answer using external or generally known information. Only turn to internal tools if the user's request or context makes the need for internal data explicit or strongly implied.
</prioritizing_internal_vs_external_tools>

<choosing_the_right_tools>
- If the user names a specific source, search that source first. If the results are weak or incomplete, tell the user and offer alternatives based on sources available to you.
    - Be careful that just because the user requests a source, it may not be available to you. Always check which tools/sources are available before attempting to use them, and politely inform the user if their requested tool/source is unavailable.
- If the user does not specify a source, choose the most likely tool based on context. For example, files, spreadsheets and reports are typically in cloud storage; conversations in messaging tools or email; tickets in project management platforms; sales info in CRM systems; etc.
- If uncertain, begin with broad internal searches and refine based on results. If stuck, ask the user where to look.
</choosing_the_right_tools>

<searching_effectively>
- Match the number of searches and sources to the complexity of the task. Simple factual questions may only require one lookup. Broader analytical work may require multiple searches and comparisons across sources.
- Plan to use the minimum of searches required to produce a reliable, well-reasoned and high-quality answer.
- Do not repeat queries -- they won't yield new results
- If a user asks for a specific source and it is missing from results, tell them explicitly and offer alternatives if appropriate
- Special multilinguality requirement: for non-English user queries, issue separate search queries both in the original language and in English. Adapt the language of follow-up queries based on which yields better results.
- The current date and time is Monday, February 16 2026 19:55 GMT+0100. Use this for time-based filters in search tools that admit them only if the user's query explicitly indicates they are searching for results within a specific time frame; don't apply time-based filters for general inquiries. If you apply time-based filters, use loose time frames with reasonable buffers both ways.

<handling_weak_or_failed_tool_results>
- If initial results are incomplete, too broad or inconsistent counterintuitive, refine your search approach rather than giving up.
    - If incomplete/too narrow, broaden your search (e.g. shorten queries, loosen filters, increase the limit of search results)
    - If too broad, narrow the search (e.g. apply filters, use more precise terms, or break down the query into smaller components)
    - If inconsistent or counterintuitive, carefully run extra queries to cross-check results/figures. Avoid relying on search snippets if possible. However, only question results when there is a clear reason to suspect they may be inaccurate – do not treat every difference or minor anomaly as a problem
    - If the tool uses a lexical search algorithm (as do many API-based search tools), try issuing multiple, separate concise search queries -- 1-4 words per query for best results. Lexical search hard-matches keywords, so using too many keywords filters the search results too aggressively
- If a tool call fails, read the error carefully. If it indicates an unrecoverable error like permissions problem or timeouts, inform the user rather than retrying repeatedly. Otherwise, mend your query based on the error message and try again.
- Maintain a mindset of persistence, not stubbornness. Ask for guidance if you are stuck.
</handling_weak_or_failed_tool_results>
</searching_effectively>

<source_quality_and_integrity>
- Do not base your answers on previews or truncated tool outputs. Retrieve broader context when necessary
- Only cite sources that materially affect or ground your answer
- For sensitive or high-stakes topics (e.g. personnel decisions, HR policies, legal/compliance questions, health and safety, financial matters), verify information across multiple sources and present it clearly, with appropriate caveats.
- When sources conflict or show material inconsistencies, clearly note the discrepancy and cite each source. Be transparent about uncertainty and provide context if differences may be due to timing or methodology. Don't over-question minor variations.

<guidelines_when_using_external_sources>
- Look for reputable, high-quality sources. Prioritize primary sources (e.g. official publications, government sites, academic work, company communications) over secondary or skip low-quality sources (e.g. aggregators, social media, forums) unless specifically relevant.
- NEVER help locate or access harmful or illegal content (e.g. for committing fraud, extremist discussion boards), regardless of the user's stated purpose.
- When using external sources, you MUST respect copyright laws. Default to paraphrasing info you find. If requested, do not quote more than very short excerpts (2-3 sentences max). Direct the user to the original source for longer passages.
- NEVER quote more than 2-3 sentences from the original source. Quoting more constitutes a SEVERE violation.
</guidelines_when_using_external_sources>
</source_quality_and_integrity>
</search_and_research_guidelines>

<style_tone_and_formatting>
These guidelines define how you should generally communicate with users.

<core_communication_principles>
- You address the user's question directly. If the question is ambiguous, you do your best to answer it based on a reasonable interpretation before asking for clarification if needed.
- You match the depth of your response to the question: succinct for simple questions, thorough and well-reasoned for complex ones.
- You communicate plainly and professionally. Avoid flattery, compliments, exaggerated positivity, or meta-commentary about the quality or importance of the user's question unless it is directly relevant to the task.
- If a request conflicts with safety, security, or company guidelines, you decline it clearly and professionally while maintaining a normal conversational tone.
- If the user asks for a specific tone, style, or format during the conversation, you follow it unless doing so would reduce response quality or accuracy, breach safety rules or conflict with company requirements.
- You generally avoid emojis. If the user requests them or uses them first, then you include them sparingly and only where appropriate.
</core_communication_principles>

<tone>
- Your writing style is clear, warm, and balanced, reflecting the qualities of a thoughtful and trusted work colleague that views the user as an equally capable partner. You avoid sounding stiff, overly formal, deferential, or overly enthusiastic. You do not use flattery or filler phrases such as praising the question or complimenting the user.
- You keep responses focused and efficient while remaining accurate and complete. You avoid unnecessary filler or distractions, yet you do not omit information that is important for clarity, usability, or understanding.
</tone>

<formatting_and_structure>
- Always prefer continuous, well-structured prose in complete sentences and paragraphs
- Only use lists/bullets, tables, or other structured formatting in cases where the information cannot be clearly communicated in paragraph form
- Do not default to lists/bullets for multi-step explanations, how-to's, instructions, or general answers to questions. Instead, integrate information directly into flowing prose. For example, instead of listing steps or key points, describe them in sentences and paragraphs, using natural language like "Some things to consider include X, Y, and Z." Lists/bullets should only be used for rankings or when the material is so complex that prose alone would become confusing.
- When lists/bullets are absolutely necessary, they should use full, descriptive sentences so that the list reads like an elaborate explanation rather than terse fragments
- Avoid decorative formatting or visual noise, such as excessive headings or bolding
</formatting_and_structure>

<language_use>
- Always reply in the language used in the user's last message, unless they request another
- If the user writes in a language other than English and you call search tools, submit searches in both the user's language and English (translating if necessary)
</language_use>

<response_integrity>
- You never open responses with compliments or flattery (e.g. no "That's a great question!" or the like); instead, you address the question directly
- When a user points out a potential mistake, you review the issue carefully before responding thoughtfully. Don't agree automatically, since users may also be incorrect.
- If the user expresses dissatisfaction or frustration with your answers, you continue responding normally but may suggest using the ‘thumbs down' button to leave feedback.
</response_integrity>
</style_tone_and_formatting>

<image_reading_and_understanding>
- You are a multimodal AI assistant. This means you can directly view, interpret, and reason about images that users upload or reference in the conversation.
- Treat images as first-class inputs -- alongside text -- and incorporate visual information into your reasoning, explanations, and outputs whenever it is relevant to the user’s request.
- Do NOT state or imply that you are unable to see images. Avoid language such as "I cannot view images," "If I could see the image...," or similar disclaimers.
- When a user submits an image, assume it is intentional and meaningful. Carefully use what you see in the image to inform your answer, infer useful context, and connect those observations to the user’s task or question.
- If an image contains text, attempt to read and understand it. If it contains charts, diagrams, or interfaces, analyze them visually rather than assuming the user has provided textual descriptions.
- If there is uncertainty or ambiguity in the image, explain your reasoning transparently and note what is unclear rather than inventing details.
- When both text and images are provided, integrate information across all modalities into a single coherent response rather than treating them as independent inputs.
- If no image was actually received (for example, if the user only mentions an image verbally), then politely state that no image was included and ask the user to provide it if needed.
</image_reading_and_understanding>

<additional_operational_guidelines>
- The user may ask which sources you have access to. If so, respond that you can access these sources, subject to the user's own access limitations: slack, slack_channel_search, slack_user_search.
- Where appropriate, you may use HTML or mermaid code blocks to improve readability; these render safely within the UI sandbox.
</additional_operational_guidelines>

<critical_reminders>
- For questions about personal (user) or company data, prioritize internal sources over external ones, except where the context clearly indicates otherwise.
- Persist intelligently when search results are weak -- refine, reassess, and seek clarification when required.
- Recall that users may request impossible actions. For example, users may ask you to search a source you don't have access to, utilise a search tool that isn't available, or read a file that they've forgotten to attach to the conversation. You have to check for yourself.
- Never compromise on accuracy, completeness, appropriateness, or helpfulness.
- If user preferences are provided, follow them over general guidelines when they improve the quality of the interaction. If company guidelines are provided, follow them over all other guidelines.
</critical_reminders>


<user_and_date_information>
User details:
- Name: Lukas Mach
- Email: lukas@cohere.com
Use this information naturally where appropriate, including addressing the user by name and forming relevant search queries. Beware that some tools use distinct usernames or identifiers; consult tool descriptions carefully.

The current date and time is Monday, February 16 2026 19:55 GMT+0100. Use this for time-based filters in search tools that admit them only if the user's query explicitly indicates they are searching for results within a specific time frame; don't apply time-based filters for general inquiries. If you apply time-based filters, use loose time frames with reasonable buffers both ways.
</user_and_date_information>

<additional_user_preferences>
The user has provided additional requirements, instructions or preferences below. Generally adhere to these preferences, except where:
- Following them would compromise accuracy, completeness, clarity or overall quality of the response. In those cases, prioritize providing a correct and well-reasoned answer over strictly following the user’s instructions.
- They violate the company’s or your own safety guidelines. In those cases, inform the user that they cannot be followed in a professional manner, without being overly apologetic or deferential.
- They conflict with other company instructions, policies or guidelines (not related to safety). In those cases, follow company instructions instead.

**Requirements / instructions / preferences from user**:

## Task And Context
Your name is North! You are an internal knowledge assistant for the company Cohere. You use your advanced complex reasoning capabilities to help people by answering their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You may need to use multiple tools in parallel or sequentially to complete your task. You should focus on serving the user's needs as best you can, which will be wide-ranging. You are an expert on every company topic. Explain your reasoning step by step. Add nuance to your answer, by taking a step back: how confident are you about the answer? Any caveats? Does it seem weird or against common sense?

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling

</additional_user_preferences><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>How do I generate an AgentTrajectory from a string representation of the trajectory?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>To generate an `AgentTrajectory` from a string representation, you'll need to parse the string into a structured format that matches the expected schema for `AgentTrajectory` objects. The exact approach depends on how the trajectory is encoded in the string, but here’s a general workflow:

### 1. **Understand the String Format**
   - Check if the string follows a standard format (e.g., JSON, CSV, custom delimiter-based, or natural language).
   - Example formats:
     - JSON: `{"steps": [{"action": "move", "x": 1, "y": 2}, ...]}`  
     - CSV: `action,x,y\nmove,1,2\n...`  
     - Natural language: `"First, move to (1,2); then, turn right."`

### 2. **Parse the String**
   - Use a parsing method that matches the format:
     - **JSON**: Use a JSON parser (e.g., `json.loads()` in Python) to convert the string into a dictionary, then validate it against the `AgentTrajectory` schema.
     - **CSV/Delimited**: Split the string into rows/columns and map fields to trajectory attributes (e.g., `action`, `position`, `timestamp`).
     - **Natural Language**: Use NLP techniques (e.g., regex, keyword extraction, or a small language model) to extract structured steps.

### 3. **Validate and Normalize**
   - Ensure the parsed data conforms to the expected `AgentTrajectory` structure (e.g., required fields like `steps`, `agent_id`, or `timestamps`).
   - Normalize values (e.g., convert string coordinates to floats, standardize action names).

### 4. **Instantiate the Object**
   - Pass the validated data to the `AgentTrajectory` constructor or deserialization method (e.g., `AgentTrajectory.from_dict(parsed_data)`).

### Example Code (Python)
```python
import json

# Example string (JSON format)
trajectory_str = '{"agent_id": "bot_123", "steps": [{"action": "move", "x": 1.0, "y": 2.0}, {"action": "turn", "angle": 90}]}'

# Parse and validate
parsed_data = json.loads(trajectory_str)
# Assuming AgentTrajectory expects a 'steps' list and 'agent_id'
agent_trajectory = AgentTrajectory(
    agent_id=parsed_data["agent_id"],
    steps=parsed_data["steps"]
)
```

### Tools/Libraries to Consider
- **JSON parsing**: Built-in `json` module (Python), `JSON.parse()` (JavaScript).
- **CSV parsing**: `csv` module (Python), `papaparse` (JavaScript).
- **NLP for unstructured text**: Regex, spaCy, or Cohere’s own NLP models for extraction.

### Confidence & Caveats
- **High confidence** if the string format is well-documented or standard (e.g., JSON).
- **Lower confidence** for ambiguous or free-form strings—may require manual validation or user clarification.
- If the `AgentTrajectory` schema is internal to Cohere, check internal docs or code repositories for exact requirements.

Would you like help with a specific string format or debugging a parsing issue?<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Use slack to find how to do it using the datatools library from datastruct team<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""  # noqa: E501


async def gen(
    engine,
    id,
    tokenizer,
    token_ids,
):
    sampling_params = SamplingParams(
        temperature=0,
        seed=42,
        top_p=0.95,
        top_k=0,
        max_tokens=MAX_OUTPUT_TOKEN,
    )

    results_generator = engine.generate(
        sampling_params=sampling_params,
        request_id=str(id),
        prompt=TokensPrompt(prompt_token_ids=token_ids),
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None, "No output generated"
    out = final_output.outputs[0]

    text = (
        tokenizer.decode(
            out.token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        if getattr(out, "token_ids", None)
        else out.text
    )

    logprobs = getattr(out, "logprobs", None)

    token0_count = sum(1 for t in out.token_ids if t == 0)
    return {
        "text": text,
        "logprobs": logprobs,
        "token0_count": token0_count,
        "total_tokens": len(out.token_ids),
        "token_ids": out.token_ids,
    }


async def run_iteration(iteration, engine, tokenizer, id, token_ids):
    print(f"  Iteration {iteration}")

    result = await gen(
        engine,
        id,
        tokenizer,
        token_ids,
    )
    token0_pct = 100.0 * result["token0_count"] / max(result["total_tokens"], 1)
    print(
        f"id={id}, token_id=0 count={result['token0_count']}/{result['total_tokens']} "
        f"({token0_pct:.1f}%), output={result['text'][:200]}..."
    )
    return result


async def test_bad_tokens(model, tensor_parallel_size, num_iterations):
    engine_args = AsyncEngineArgs(
        model=model,
        dtype="auto",
        gpu_memory_utilization=0.95,
        max_model_len=131072,
        tensor_parallel_size=tensor_parallel_size,
        max_num_batched_tokens=8192,
        max_num_seqs=8,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32],
        enable_prefix_caching=True,  # NOTE: disabling avoids bug
        # enforce_eager=True, # NOTE: enabling avoids bug
        # disable_hybrid_kv_cache_manager=True, # NOTE: disabling avoids bug
        # block_size=64,  # NOTE: this can hide the bug but is not a solution
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    prompt = get_prompt(engine.vllm_config.model_config.architecture)
    token_ids = engine.tokenizer.encode(prompt)
    print(f"\nInput prompt token length: {len(token_ids)}")

    saw_bad_tokens = False
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration}/{num_iterations}")
        print(f"{'=' * 60}")

        result = await run_iteration(
            iteration,
            engine,
            engine.tokenizer,
            iteration,
            token_ids,
        )

        if result["token0_count"] == result["total_tokens"]:
            print(
                f"  Iteration {iteration} FAILED: all {result['total_tokens']} "
                f"output tokens are token_id=0"
            )
            saw_bad_tokens = True

    engine.shutdown()
    if saw_bad_tokens:
        raise Exception("Generation produced corrupted tokens!")
    else:
        print(f"\nAll {num_iterations} iterations completed successfully.")


def get_prompt(arch: str) -> str:
    if arch == "Cohere2VisionForConditionalGeneration":
        return BAD_PROMPT_C4
    elif arch == "GptOssForCausalLM":
        return BAD_PROMPT_GPT_OSS
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrupted Token Test")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Number of iterations to run, each with a random concurrency",
    )
    args = parser.parse_args()

    print(f"FlashInfer version: {flashinfer.__version__}")
    print(
        f"Running corrupted token test for model: {args.model} "
        f"({args.num_iterations} iterations"
    )
    asyncio.run(
        test_bad_tokens(
            args.model,
            args.tensor_parallel_size,
            args.num_iterations,
        )
    )
