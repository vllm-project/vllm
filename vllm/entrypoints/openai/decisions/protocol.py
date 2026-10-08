# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI Decisions API request and response models."""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StrictBool, StrictStr, model_validator

from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel

ShortText = Annotated[StrictStr, Field(max_length=1048576)]
InputText = Annotated[StrictStr, Field(max_length=10485760)]
ChoiceValue = StrictStr | StrictBool


class DecisionModel(OpenAIBaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionInputText(DecisionModel):
    type: Literal["input_text"]
    text: InputText


class DecisionInputImage(DecisionModel):
    type: Literal["input_image"]
    image_url: Annotated[
        StrictStr,
        Field(max_length=20971520, pattern=r"^(https?://|data:image/).+"),
    ]
    detail: Literal["auto", "low", "high"] = "auto"


DecisionInputPart = Annotated[
    DecisionInputText | DecisionInputImage, Field(discriminator="type")
]


class DecisionInputMessage(DecisionModel):
    role: Literal["user"]
    content: InputText | Annotated[list[DecisionInputPart], Field(max_length=16384)]
    type: Literal["message"] = "message"


class QuestionBase(DecisionModel):
    instructions: ShortText
    name: ShortText | None = None

    @model_validator(mode="after")
    def validate_name(self) -> "QuestionBase":
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name must be a string when provided")
        return self


class PredicateQuestion(QuestionBase):
    type: Literal["predicate"]


class ChoiceOption(DecisionModel):
    value: ChoiceValue
    description: ShortText = ""


class ChoiceQuestion(QuestionBase):
    type: Literal["choice"]
    choices: list[ChoiceOption] = Field(min_length=2, max_length=255)

    @model_validator(mode="after")
    def validate_choices(self) -> "ChoiceQuestion":
        values = [(type(c.value), c.value) for c in self.choices]
        if len(set(values)) != len(values):
            raise ValueError("choice values must be distinct")
        return self


class ScoreLevel(DecisionModel):
    label: ShortText
    description: ShortText = ""


class ScoreQuestion(QuestionBase):
    type: Literal["score"]
    levels: list[ScoreLevel] = Field(min_length=2, max_length=10)


DecisionQuestion = Annotated[
    PredicateQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class DecisionRequest(DecisionModel):
    model: ShortText
    input: InputText | Annotated[list[DecisionInputMessage], Field(max_length=131072)]
    questions: list[DecisionQuestion] = Field(min_length=1, max_length=200)
    safety_identifier: Annotated[StrictStr, Field(max_length=128)] | None = None


class PredicateAnswer(DecisionModel):
    type: Literal["predicate"] = "predicate"
    name: str | None
    probability: float


class ChoiceProbability(DecisionModel):
    value: ChoiceValue
    probability: float


class ChoiceAnswer(DecisionModel):
    type: Literal["choice"] = "choice"
    name: str | None
    choice: ChoiceValue
    probabilities: list[ChoiceProbability]
    confidence: float


class ScoreProbability(DecisionModel):
    value: int
    label: str
    probability: float


class ScoreAnswer(DecisionModel):
    type: Literal["score"] = "score"
    name: str | None
    score: float
    probabilities: list[ScoreProbability]
    confidence: float


DecisionAnswer = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class InputTokensDetails(DecisionModel):
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class OutputTokensDetails(DecisionModel):
    reasoning_tokens: int = 0


class OpenAIDecisionUsage(DecisionModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


class DecisionResponse(DecisionModel):
    model: str
    answers: list[DecisionAnswer]
    usage: OpenAIDecisionUsage
