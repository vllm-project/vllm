import dataclasses

@dataclasses.dataclass
class BeamDebugInfo:
    final_score: float = dataclasses.field(default_factory=float)
    cummulative_penalty: float = dataclasses.field(default_factory=float)
    cummulative_ranking_score: float = dataclasses.field(default_factory=float)
    penalty_classifiers_that_are_over_threshold: list[str] = dataclasses.field(default_factory=list)
    content: str = dataclasses.field(default_factory=str)
    filtered_classifiers: list[str] = dataclasses.field(default_factory=list)


