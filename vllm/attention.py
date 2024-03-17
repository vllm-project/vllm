import enum


class AttentionBackend(enum.Enum):

    FLASHINFER = enum.auto()
    XFORMERS = enum.auto()


class KVCacheLayout(enum.Enum):

    # (batch, num_heads, seq_len, head_dim)
    # (batch, seq_len, num_heads, head_dim)
    # (batch, num_heads, head_dim, seq_len)
    # (batch, head_dim, num_heads, seq_len)
    # (batch, head_dim, seq_len