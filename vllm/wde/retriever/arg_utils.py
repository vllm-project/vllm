from vllm.wde.decode_only.arg_utils import DecodeOnlyEngineArgs


class RetrieverDecodeOnlyEngineArgs(DecodeOnlyEngineArgs):

    def __post_init__(self):
        super().__post_init__()
        self.output_last_hidden_states = True
