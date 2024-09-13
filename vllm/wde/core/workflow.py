

class Workflow:
    EngineArgs: str
    Scheduler: str
    AttnBackend: str
    Tokenizer: str = "vllm.wde.core.inputs.tokenizer:Tokenizer"
    InputProcessor: str
    RequestProcessor: str
    OutputProcessor: str
    ModelInputBuilder: str
    Executor: str
    Worker: str

    @classmethod
    def from_engine(cls, engine: "LLMEngine"):
        return cls()
