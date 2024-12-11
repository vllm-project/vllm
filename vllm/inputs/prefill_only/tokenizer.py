from vllm.transformers_utils.tokenizer import get_tokenizer


class Tokenizer:

    def __init__(self, tokenizer_name: str, **kwargs):
        self.tokenizer_name = tokenizer_name
        self.tokenizer_kwargs = kwargs

        self.tokenizer = get_tokenizer(tokenizer_name=self.tokenizer_name,
                                       **self.tokenizer_kwargs)

    @classmethod
    def from_engine(cls, engine):
        init_kwargs = dict(
            tokenizer_name=engine.engine_config.model_config.tokenizer,
            tokenizer_mode=engine.engine_config.model_config.tokenizer_mode,
            trust_remote_code=engine.engine_config.model_config.
            trust_remote_code,
            revision=engine.engine_config.model_config.tokenizer_revision)

        return cls(**init_kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id