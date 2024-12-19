from my_attention import DummyAttentionBackend


class DummyModelRunner:

    def __init__(self):
        self.attn_backend = DummyAttentionBackend()
