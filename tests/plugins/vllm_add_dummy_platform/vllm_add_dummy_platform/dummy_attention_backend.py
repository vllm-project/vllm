from vllm.attention.backends.flash_attn import FlashAttentionBackend


class DummyAttentionBackend(FlashAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "Dummy_Backend"
