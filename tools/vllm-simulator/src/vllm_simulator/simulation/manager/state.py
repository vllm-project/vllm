class StateManager:
    _global_clock: float = 0

    @classmethod
    def reset(cls):
        cls._global_clock = 0

    @classmethod
    def get_global_clock(cls) -> float:
        return cls._global_clock

    @classmethod
    def step_global_clock(cls, dur: float) -> None:
        cls._global_clock += dur

    @classmethod
    def set_global_clock(cls, clock: float) -> None:
        cls._global_clock = clock
