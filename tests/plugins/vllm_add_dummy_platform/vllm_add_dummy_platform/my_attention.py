class DummyAttentionImpl:

    def forward(self):
        pass


class DummyAttentionBackend:

    def __init__(self):
        pass

    def get_impl_cls(self):
        return DummyAttentionImpl
