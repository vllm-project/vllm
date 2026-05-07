# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight bee-eval sample checker.

Loads samples from local data files (the same datasets used by the full
bee eval suite in apiary) and validates vLLM responses against ground truth.
Does **not** own the HTTP call -- the caller sends the request and passes the
raw generation text back for scoring.

Supported tasks:
  mmlupro      -- MMLUPro Artificial-Analysis style (text, letter extraction)
  ocrbench     -- OCRBench (vision, accuracy / soft_accuracy / anls)
  infovqa      -- InfoVQA (vision, anls)
  mathvista    -- MathVista (vision, heuristic answer extraction)
  aime         -- AIME 2025 (text, simplified numeric match)
  mgsm         -- MGSM multilingual math (text, numeric extraction)
  mbpp_plus    -- MBPP+ coding (text, simplified syntax + output check)
  niah         -- Needle In A Haystack (text, substring retrieval)
"""

from __future__ import annotations

import abc
import ast
import csv
import json
import string
from dataclasses import dataclass, field
from typing import Any

import regex as re

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """A single eval sample ready to send to a vLLM server."""

    messages: list[dict[str, Any]]
    ground_truth: str | list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of checking a vLLM response against ground truth."""

    passed: bool
    score: float
    metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric helpers  (ported from apiary bee/bee/task_utils/vision/metrics.py)
# ---------------------------------------------------------------------------


def calculate_anls(label: str, generation: str) -> float:
    """Normalised edit-distance score, thresholded at 0.5."""
    label = label.strip().lower()
    generation = generation.strip().lower()
    if not label and not generation:
        return 1.0
    distance = _edit_distance(label, generation)
    max_len = max(len(label), len(generation))
    if not max_len:
        return 0.0
    nls = distance / max_len
    return 0.0 if 1 - nls < 0.5 else 1 - nls


def _edit_distance(s: str, t: str) -> int:
    """Levenshtein distance (no external dependency)."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def _normalize_text(text: str) -> str:
    """Lowercase, strip, remove trailing punctuation (except %)."""
    text = text.strip().lower()
    while text and text[-1] in string.punctuation and text[-1] != "%":
        text = text[:-1]
    return text


def compute_soft_accuracy(prediction: str, target: str) -> float:
    """OCRBench-style soft accuracy: normalise and compare."""
    pred = _normalize_text(prediction)
    tgt = _normalize_text(target)
    if pred == tgt:
        return 1.0
    pred_no_space = pred.replace(" ", "")
    tgt_no_space = tgt.replace(" ", "")
    if pred_no_space == tgt_no_space:
        return 1.0
    if tgt_no_space in pred_no_space or pred_no_space in tgt_no_space:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------


class BeeEvalTask(abc.ABC):
    """Abstract base for lightweight bee-eval tasks."""

    task_name: str = ""

    @abc.abstractmethod
    def load_samples(self, path: str, n: int | None = None) -> list[EvalSample]:
        """Load up to *n* samples from a local data file."""

    @abc.abstractmethod
    def check_response(self, sample: EvalSample, generation: str) -> EvalResult:
        """Score a vLLM generation against the sample's ground truth."""


# ---------------------------------------------------------------------------
# MMLUPro (Artificial Analysis)
# ---------------------------------------------------------------------------


class MMLUProAATask(BeeEvalTask):
    task_name = "mmlupro"

    MC_OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    EXTRACTION_PATTERNS = [
        r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])",
        r"\\boxed\{[^}]*([A-Z])[^}]*\}",
        r"answer is ([a-zA-Z])",
        r"answer is \(([a-zA-Z])",
        r"([A-Z])\)\s*[^A-Z]*",
        r"([A-Z])\s+is\s+the\s+correct\s+answer",
        r"([A-Z])\s*$",
        r"([A-Z])\s*\.",
        r"([A-Z])\s*[^\w]",
    ]

    def _format_prompt(
        self, question: str, options: list[str], _category: str = ""
    ) -> str:
        prompt = (
            "Answer the following multiple choice question. The last line of your"
            " response should be in the following format: 'Answer: A/B/C/D' (e.g."
            f" 'Answer: A').\n\n{question}\n"
        )
        for i, opt in enumerate(options):
            prompt += f"\n{self.MC_OPTIONS[i]}) {opt}"
        return prompt

    def _extract_answer(self, text: str) -> str | None:
        for pattern in self.EXTRACTION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1).upper()
        return None

    def load_samples(self, path: str, n: int | None = None) -> list[EvalSample]:
        samples: list[EvalSample] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                rec = json.loads(line)
                prompt = self._format_prompt(
                    rec["question"], rec["options"], rec.get("category", "")
                )
                messages = [{"role": "user", "content": prompt}]
                samples.append(
                    EvalSample(
                        messages=messages,
                        ground_truth=rec["answer"],
                        metadata={
                            "question_id": rec.get("question_id"),
                            "category": rec.get("category"),
                        },
                    )
                )
        return samples

    def check_response(self, sample: EvalSample, generation: str) -> EvalResult:
        extracted = self._extract_answer(generation)
        accuracy = (
            float(extracted == sample.ground_truth) if extracted is not None else 0.0
        )
        return EvalResult(passed=accuracy == 1.0, score=accuracy)


# ---------------------------------------------------------------------------
# OCRBench  (vision)
# ---------------------------------------------------------------------------


class OCRBenchTask(BeeEvalTask):
    task_name = "ocrbench"

    def load_samples(self, path: str, n: int | None = None) -> list[EvalSample]:
        samples: list[EvalSample] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                rec = json.loads(line)
                content: list[dict[str, Any]] = []
                for img_uri in rec.get("images", []):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_uri},
                        }
                    )
                content.append({"type": "text", "text": rec["prompt"]})
                messages = [{"role": "user", "content": content}]
                samples.append(
                    EvalSample(
                        messages=messages,
                        ground_truth=rec["completion"],
                        metadata={
                            "question_id": rec.get("question_id"),
                            "type": rec.get("type"),
                            "dataset_name": rec.get("dataset_name"),
                        },
                    )
                )
        return samples

    def check_response(self, sample: EvalSample, generation: str) -> EvalResult:
        gt_str = (
            sample.ground_truth
            if isinstance(sample.ground_truth, str)
            else sample.ground_truth[0]
        )

        accuracy = float(generation.strip().lower() == gt_str.strip().lower())
        soft = compute_soft_accuracy(generation, gt_str)
        anls = calculate_anls(gt_str, generation)

        return EvalResult(
            passed=(accuracy == 1.0 or soft == 1.0),
            score=max(accuracy, soft, anls),
        )


# ---------------------------------------------------------------------------
# InfoVQA  (vision)
# ---------------------------------------------------------------------------


class InfoVQATask(BeeEvalTask):
    task_name = "infovqa"

    POST_PROMPT = "\nAnswer the question using a single word or phrase."

    def load_samples(self, path: str, n: int | None = None) -> list[EvalSample]:
        samples: list[EvalSample] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                rec = json.loads(line)
                text = rec["prompt"] + self.POST_PROMPT
                content: list[dict[str, Any]] = []
                for img_uri in rec.get("images", []):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_uri},
                        }
                    )
                content.append({"type": "text", "text": text})
                messages = [{"role": "user", "content": content}]
                gt = rec["completion"]
                samples.append(
                    EvalSample(
                        messages=messages,
                        ground_truth=gt,
                        metadata={"question_id": rec.get("question_id")},
                    )
                )
        return samples

    def check_response(self, sample: EvalSample, generation: str) -> EvalResult:
        gt = sample.ground_truth
        if isinstance(gt, list):
            anls = max((calculate_anls(c, generation) for c in gt), default=0.0)
        else:
            anls = calculate_anls(gt, generation)
        return EvalResult(passed=anls >= 0.5, score=anls)


# ---------------------------------------------------------------------------
# MathVista  (vision)
# ---------------------------------------------------------------------------


class MathVistaTask(BeeEvalTask):
    task_name = "mathvista"

    _ANSWER_PATTERNS = [
        r"(?i)(?:the\s+)?answer\s*:\s*(.+?)(?:\.(?!\d)|$)",
        r"(?i)(?:the\s+)?answer\s+is[:\s]*(.+?)(?:\.(?!\d)|$)",
        r"\\boxed\{(.+?)\}",
        r"(?i)final\s+answer[:\s]*(.+?)(?:\.(?!\d)|$)",
        r"=\s*(.+?)$",
    ]

    def _extract_answer(self, text: str) -> str:
        cleaned = text.replace("*", "")
        cleaned = re.sub(r"\\[()]", "", cleaned)
        for pattern in self._ANSWER_PATTERNS:
            match = re.search(pattern, cleaned, re.MULTILINE)
            if match:
                return match.group(1).strip()
        last_line = cleaned.strip().split("\n")[-1].strip()
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", last_line)
        if nums:
            return nums[-1].replace(",", "")
        return last_line

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.strip().lower()
        text = text.rstrip(".")
        text = text.replace(",", "")
        text = text.replace("$", "")
        text = text.replace("%", "")
        text = text.replace("*", "")
        text = text.strip()
        return text

    @staticmethod
    def _safe_equal(pred: str, gt: str) -> bool:
        if pred == gt:
            return True
        try:
            return abs(float(pred) - float(gt)) < 1e-6
        except ValueError:
            return False

    def load_samples(self, path: str, n: int | None = None) -> list[EvalSample]:
        samples: list[EvalSample] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                rec = json.loads(line)
                content: list[dict[str, Any]] = []
                b64 = rec.get("image_base64", "")
                if b64:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                            },
                        }
                    )
                content.append({"type": "text", "text": rec["query"]})
                messages = [{"role": "user", "content": content}]
                samples.append(
                    EvalSample(
                        messages=messages,
                        ground_truth=str(rec["answer"]),
                        metadata={
                            "pid": rec.get("pid"),
                            "question_type": rec.get("question_type"),
                            "answer_type": rec.get("answer_type"),
                            "choices": rec.get("choices"),
                        },
                    )
                )
        return samples

    def check_response(self, sample: EvalSample, generation: str) -> EvalResult:
        extracted = self._extract_answer(generation)
        pred = self._normalize(extracted)
        gt_raw = (
            sample.ground_truth
            if isinstance(sample.ground_truth, str)
            else sample.ground_truth[0]
        )
        correct = self._safe_equal(pred, self._normalize(gt_raw))
        return EvalResult(passed=correct, score=float(correct))


# ---------------------------------------------------------------------------
# AIME 2025  (text, simplified numeric match)
# ---------------------------------------------------------------------------


class AIMETask(BeeEvalTask):
    task_name = "aime"

    _NUMERIC_PATTERNS = [
        r"\\boxed\{(\d+)\}",
        r"(?i)(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(\d+)",
        r"(?i)answer[:\s]*(\d+)",
        r"(\d+)\s*$",
    ]

    def _extract_numeric(self, text: str) -> str | None:
        for pattern in self._NUMERIC_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        nums = re.findall(r"\d+", text)
        return nums[-1] if nums else None

    def load_samples(self, path: str, n: int | None = None) -> list[EvalSample]:
        samples: list[EvalSample] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for i, rec in enumerate(reader):
                if n is not None and i >= n:
                    break
                messages = [{"role": "user", "content": rec["prompt"]}]
                samples.append(
                    EvalSample(
                        messages=messages,
                        ground_truth=rec["completion"].strip(),
                        metadata={},
                    )
                )
        return samples

    def check_response(self, sample: EvalSample, generation: str) -> EvalResult:
        extracted = self._extract_numeric(generation)
        gt = (
            sample.ground_truth
            if isinstance(sample.ground_truth, str)
            else sample.ground_truth[0]
        )
        matched = extracted is not None and extracted == gt.strip()
        return EvalResult(passed=matched, score=float(matched))


# ---------------------------------------------------------------------------
# MGSM  (multilingual math, text)
# ---------------------------------------------------------------------------


class MGSMTask(BeeEvalTask):
    """Multilingual Grade-School Math (0-shot).

    Data is GSM8k-style CSV with ``prompt`` and ``completion`` columns.
    The prompt already asks the model to output ``#### <number>`` at the end.
    """

    task_name = "mgsm"

    _DELIMITER_PATTERN = re.compile(r"####\s*(.+)")
    _NUMERIC_CLEANUP = re.compile(r"[^\d.\-]")

    def _extract_numeric(self, text: str) -> str | None:
        m = self._DELIMITER_PATTERN.search(text)
        if m:
            return self._NUMERIC_CLEANUP.sub("", m.group(1)).strip()
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else None

    def load_samples(self, path: str, n: int | None = None) -> list[EvalSample]:
        samples: list[EvalSample] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for i, rec in enumerate(reader):
                if n is not None and i >= n:
                    break
                messages = [{"role": "user", "content": rec["prompt"]}]
                gt = rec["completion"].strip()
                samples.append(
                    EvalSample(
                        messages=messages,
                        ground_truth=gt,
                        metadata={},
                    )
                )
        return samples

    def check_response(self, sample: EvalSample, generation: str) -> EvalResult:
        extracted = self._extract_numeric(generation)
        gt = (
            sample.ground_truth
            if isinstance(sample.ground_truth, str)
            else sample.ground_truth[0]
        )
        matched = extracted is not None and extracted == gt.strip()
        return EvalResult(passed=matched, score=float(matched))


# ---------------------------------------------------------------------------
# MBPP+  (coding, simplified scoring)
# ---------------------------------------------------------------------------


class MBPPPlusTask(BeeEvalTask):
    """MBPP+ code generation with simplified scoring.

    Full ``pass_at_1`` requires sandboxed execution.  This lightweight checker
    instead verifies:
      - ``parseable``: the generated code is valid Python (``ast.parse``)
      - ``contains_function``: it defines at least one function
      - ``name_match``: the expected function name appears in the code

    These are necessary-but-not-sufficient conditions for correctness; they
    catch common failure modes (refusals, garbage output, wrong language)
    without needing a sandbox.
    """

    task_name = "mbpp_plus"

    _FUNC_NAME_RE = re.compile(r"def\s+(\w+)\s*\(")

    def _expected_func_name(self, reference_code: str) -> str | None:
        m = self._FUNC_NAME_RE.search(reference_code)
        return m.group(1) if m else None

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Pull the first fenced code block, or return the full text."""
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        return m.group(1) if m else text

    def load_samples(self, path: str, n: int | None = None) -> list[EvalSample]:
        samples: list[EvalSample] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                rec = json.loads(line)
                prompt = (
                    f"{rec['prompt']}\n\n"
                    "Write a Python function that solves the problem. "
                    "Return only the function definition."
                )
                messages = [{"role": "user", "content": prompt}]
                samples.append(
                    EvalSample(
                        messages=messages,
                        ground_truth=rec["code"],
                        metadata={
                            "task_id": rec.get("task_id"),
                            "test_list": rec.get("test_list", []),
                        },
                    )
                )
        return samples

    def check_response(self, sample: EvalSample, generation: str) -> EvalResult:
        code = self._extract_code_block(generation)

        try:
            ast.parse(code)
            parseable = True
        except SyntaxError:
            parseable = False

        has_function = bool(self._FUNC_NAME_RE.search(code))

        ref = (
            sample.ground_truth
            if isinstance(sample.ground_truth, str)
            else sample.ground_truth[0]
        )
        expected_name = self._expected_func_name(ref)
        name_match = bool(expected_name and expected_name in code)

        checks = [parseable, has_function, name_match]
        return EvalResult(
            passed=all(checks),
            score=sum(checks) / len(checks),
        )


# ---------------------------------------------------------------------------
# Needle In A Haystack  (classic single-needle retrieval)
# ---------------------------------------------------------------------------


class NIAHTask(BeeEvalTask):
    """Needle In A Haystack — classic single-needle retrieval.

    Based on https://github.com/gkamradt/LLMTest_NeedleInAHaystack and the
    apiary ``NeedleInAHaystack`` task.  Each sample specifies a target context
    length and a depth at which to insert the needle.  The haystack is built
    dynamically from repeating filler paragraphs so the data file stays small.
    """

    task_name = "niah"

    _SYSTEM_PROMPT = (
        "You are a helpful AI bot that answers questions for a user. "
        "Keep your response short and direct"
    )

    _PROMPT_TEMPLATE = (
        "<context>\n{context}\n</context>\n\n"
        "{retrieval_question} Don't give information outside the document "
        "or repeat your findings"
    )

    _FILLER_PARAGRAPHS = [
        "The concept of time has fascinated philosophers and scientists for centuries. From Newton's absolute time to Einstein's relativistic framework, our understanding of temporal dynamics has undergone dramatic shifts. Modern physics suggests that time is not a fixed backdrop against which events unfold, but rather a flexible dimension intimately connected to space, gravity, and velocity. The implications of this understanding extend far beyond theoretical physics, touching on questions of consciousness, causality, and the very nature of reality itself.",  # noqa: E501
        "Advances in agricultural technology have transformed food production over the past century. The Green Revolution of the 1960s introduced high-yield crop varieties, synthetic fertilizers, and modern irrigation techniques that dramatically increased global food output. Today, precision agriculture uses satellite imagery, soil sensors, and automated machinery to optimize every aspect of farming. These technologies have enabled farmers to produce more food on less land while reducing environmental impact, though challenges related to sustainability and equitable access remain.",  # noqa: E501
        "The development of written language represents one of humanity's most significant achievements. Early writing systems, such as Sumerian cuneiform and Egyptian hieroglyphics, emerged around 3400 BCE as tools for record-keeping and administration. Over millennia, these systems evolved into alphabetic scripts that could represent the full range of human speech. The invention of the printing press in the 15th century democratized access to written knowledge, while digital technology has further transformed how we create, distribute, and consume text.",  # noqa: E501
        "Ocean currents play a crucial role in regulating Earth's climate. The global thermohaline circulation, sometimes called the ocean conveyor belt, transports warm water from the tropics toward the poles and cold water back toward the equator. This massive system of currents influences weather patterns, marine ecosystems, and atmospheric carbon dioxide levels. Scientists are closely monitoring changes in ocean circulation patterns, as disruptions could have far-reaching consequences for global climate stability.",  # noqa: E501
        "The history of mathematics is a story of abstraction and generalization. Ancient civilizations developed arithmetic and geometry to solve practical problems in commerce, construction, and astronomy. Greek mathematicians introduced the concept of proof and laid the foundations for formal logic. The development of algebra, calculus, and modern analysis expanded the scope of mathematical inquiry, while the rise of computing has opened new frontiers in numerical methods, cryptography, and data science.",  # noqa: E501
        "Biodiversity is essential for maintaining healthy ecosystems and supporting human well-being. Tropical rainforests, coral reefs, and wetlands are among the most biodiverse habitats on Earth, housing millions of species of plants, animals, and microorganisms. These ecosystems provide critical services including air and water purification, pollination of crops, regulation of disease, and carbon sequestration. The ongoing loss of biodiversity due to habitat destruction, pollution, and climate change poses serious risks to ecological stability and human societies.",  # noqa: E501
        "The evolution of transportation technology has shaped the course of human civilization. From the domestication of horses to the development of sailing vessels, steam engines, automobiles, and aircraft, each advance in transportation has expanded the reach of trade, communication, and cultural exchange. Modern transportation networks connect virtually every corner of the globe, enabling the rapid movement of goods and people that underpins the global economy.",  # noqa: E501
        "Understanding the human brain remains one of the greatest challenges in science. The brain contains approximately 86 billion neurons, each forming thousands of connections with other neurons, creating an extraordinarily complex network. Neuroscientists study this network at multiple scales, from the molecular mechanisms of individual synapses to the large-scale patterns of activity that underlie cognition, emotion, and behavior. Advances in brain imaging technology, computational modeling, and genetic analysis are providing new insights into how the brain works and what goes wrong in neurological and psychiatric disorders.",  # noqa: E501
        "The periodic table of elements represents a masterpiece of scientific organization. Dmitri Mendeleev first published his version in 1869, arranging elements by atomic weight and noting periodic patterns in their properties. This framework not only organized known elements but successfully predicted the existence and properties of elements yet to be discovered. Today, the periodic table contains 118 confirmed elements, with the heaviest ones created artificially in particle accelerators.",  # noqa: E501
        "Architecture reflects the values, technologies, and aspirations of the societies that create it. From the monumental pyramids of ancient Egypt to the soaring Gothic cathedrals of medieval Europe, from the sleek skyscrapers of modern cities to the sustainable buildings of the 21st century, architectural design has continually evolved. Contemporary architects increasingly focus on energy efficiency, environmental sustainability, and the integration of buildings with their natural surroundings, seeking to create structures that are not only functional and beautiful but also ecologically responsible.",  # noqa: E501
    ]

    def _build_haystack(
        self, target_chars: int, needle: str, depth_percent: int
    ) -> str:
        """Build a haystack of ~*target_chars* with *needle*
        at *depth_percent*."""
        filler_block = "\n\n".join(self._FILLER_PARAGRAPHS) + "\n\n"
        reps = (target_chars // len(filler_block)) + 2
        full_text = (filler_block * reps)[:target_chars]

        if depth_percent >= 100:
            return full_text.rstrip() + " " + needle

        insertion_point = int(len(full_text) * (depth_percent / 100))
        while insertion_point > 0 and full_text[insertion_point - 1] != ".":
            insertion_point -= 1

        return (
            full_text[:insertion_point]
            + " "
            + needle
            + " "
            + full_text[insertion_point:]
        )

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Lower, strip punctuation/articles, collapse whitespace (QuAC-style)."""
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = "".join(ch for ch in text if ch not in string.punctuation)
        return " ".join(text.split())

    def load_samples(self, path: str, n: int | None = None) -> list[EvalSample]:
        samples: list[EvalSample] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if n is not None and i >= n:
                    break
                rec = json.loads(line)
                needle = rec["needle"]
                question = rec["retrieval_question"]
                references = rec.get("references", [needle])
                context_length_chars = rec["context_length_chars"]
                depth_percent = rec["depth_percent"]

                context = self._build_haystack(
                    context_length_chars, needle, depth_percent
                )
                prompt = self._PROMPT_TEMPLATE.format(
                    context=context, retrieval_question=question
                )
                messages = [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                samples.append(
                    EvalSample(
                        messages=messages,
                        ground_truth=references,
                        metadata={
                            "context_length_chars": context_length_chars,
                            "depth_percent": depth_percent,
                        },
                    )
                )
        return samples

    def check_response(self, sample: EvalSample, generation: str) -> EvalResult:
        gen_norm = self._normalize_answer(generation)

        refs = (
            sample.ground_truth
            if isinstance(sample.ground_truth, list)
            else [sample.ground_truth]
        )
        for ref in refs:
            if self._normalize_answer(ref) in gen_norm:
                return EvalResult(passed=True, score=1.0)

        return EvalResult(passed=False, score=0.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, type[BeeEvalTask]] = {
    "mmlupro": MMLUProAATask,
    "ocrbench": OCRBenchTask,
    "infovqa": InfoVQATask,
    "mathvista": MathVistaTask,
    "aime": AIMETask,
    "mgsm": MGSMTask,
    "mbpp_plus": MBPPPlusTask,
    "niah": NIAHTask,
}


def get_task(name: str) -> BeeEvalTask:
    """Instantiate a task by its registry name."""
    cls = TASK_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown task {name!r}. Available: {sorted(TASK_REGISTRY)}")
    return cls()
