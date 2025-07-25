from transformers import PretrainedConfig


class TiltConfig(PretrainedConfig):
    model_type = "tilt"

    def __init__(
        self,
        vocab_size=32128,
        d_model=1024,
        d_ff=4096,
        d_kv=64,
        num_attention_heads=16,
        num_hidden_layers=24,
        bias_num_buckets=32,
        bias_max_distance=128,
        bias_planar_max_distance=100,
        bias_horz_scale=100,
        bias_vert_scale=100,
        layer_norm_epsilon=1e-6,
        pad_token_id=0,
        eos_token_id=1,
        roi_expansion_width=0,
        unet_in_channels=1,
        unet_init_features=32,
        unet_out_features=128,
        max_seq_length=125000,
        max_output_length=512,
        max_question_length=256,
        chunk_length=1024,
        max_images_in_chunk=4,
        image_limit=250,
        image_width=768,
        max_image_height=2048,
        crop_bboxes=True,
        prefix_separator=" : ",
        answer_separator=" | ",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_kv = d_kv
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        # Relative positional bias parameters
        self.bias_num_buckets = bias_num_buckets
        self.bias_max_distance = bias_max_distance
        self.bias_planar_max_distance = bias_planar_max_distance
        self.bias_horz_scale = bias_horz_scale
        self.bias_vert_scale = bias_vert_scale

        self.layer_norm_epsilon = layer_norm_epsilon

        # Image processing parameters
        self.roi_expansion_width = roi_expansion_width
        self.unet_in_channels = unet_in_channels
        self.unet_init_features = unet_init_features
        self.unet_out_features = unet_out_features

        # Chunking and sequence length limit parameters
        self.max_seq_length = max_seq_length
        self.max_output_length = max_output_length
        self.max_question_length = max_question_length
        self.chunk_length = chunk_length
        self.max_images_in_chunk = max_images_in_chunk
        self.image_limit = image_limit
        self.image_width = image_width
        self.max_image_height = max_image_height
        self.crop_bboxes = crop_bboxes

        # Separators
        self.prefix_separator = prefix_separator
        self.answer_separator = answer_separator

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=pad_token_id,
            decoder_start_token_id=pad_token_id,
            is_encoder_decoder=True,
            **kwargs,
        )
        self.architectures = ["TiltModel"]

    @property
    def hidden_size(self) -> int:
        return self.d_model
