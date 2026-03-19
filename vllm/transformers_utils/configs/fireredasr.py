from transformers import PretrainedConfig

# Configuration class
class FireRedASRConfig(PretrainedConfig):
    """Configuration class for FireRedASR model"""
    model_type = "fireredasr_aed"

    def __init__(
        self,
        num_mel_bins: int = 80,
        vocab_size: int = 7832,
        sos_id: int = 3,
        eos_id: int = 4,
        pad_id: int = 2,
        encoder_layers: int = 16,
        decoder_layers: int = 16,
        encoder_attention_heads: int = 20,
        decoder_attention_heads: int = 20,
        d_model: int = 1280,
        residual_dropout: float = 0.1,
        dropout_rate: float = 0.1,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
        activation_function: str = "gelu",
        max_target_positions: int = 448,
        max_source_positions: int = 1500,
        is_encoder_decoder: bool = True,
        scale_embedding: bool = False,
        **kwargs
    ):
        self.num_mel_bins = num_mel_bins
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.d_model = d_model
        self.residual_dropout = residual_dropout
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.pe_maxlen = pe_maxlen
        self.activation_function = activation_function
        self.max_target_positions = max_target_positions
        self.max_source_positions = max_source_positions
        self.is_encoder_decoder = is_encoder_decoder
        self.scale_embedding = scale_embedding
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(**kwargs)
