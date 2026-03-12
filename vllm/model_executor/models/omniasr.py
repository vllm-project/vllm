# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import nn
import math
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear

class OmniASRModel(nn.Module):
    """Full OmniASR: encoder + projection + LLaMA decoder.
    
    TODO: Integrate with vLLM's LlamaForCausalLM for decoder.
    """
    def __init__(self):
        super().__init__()
        self.encoder_frontend = Wav2Vec2Frontend()
        self.encoder = Wav2Vec2TransformerEncoder()
        self.encoder_proj = nn.Linear(1024, 4096, bias=True)
        # TODO: Replace with vLLM's LlamaForCausalLM
        # self.language_model = LlamaForCausalLM(vllm_config)
        self.text_frontend = nn.Embedding(9813, 4096)
        self.lang_embeddings = nn.Embedding(1694, 4096)
        self.final_proj = nn.Linear(4096, 9812, bias=False)

    def forward(self, audio):
        x = self.encoder_frontend(audio)
        x = self.encoder(x)
        x = self.encoder_proj(x)
        return x  # [batch, seq, 4096] ready for LLaMA decoder
        
class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = [1, 512, 512, 512, 512, 512, 512]
        kernel_sizes = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]
        for i in range(7):
            conv = nn.Conv1d(in_channels[i], 512, kernel_sizes[i], stride=strides[i], bias=True)
            layer_norm = nn.LayerNorm(512)
            self.layers.append(nn.ModuleDict({"conv": conv, "layer_norm": layer_norm}))

    def forward(self, x):
        for layer in self.layers:
            x = layer["conv"](x)
            x = x.transpose(1, 2)
            x = layer["layer_norm"](x)
            x = x.transpose(1, 2)
            x = nn.functional.gelu(x)
        return x


class Wav2Vec2Attention(nn.Module):
    """Self-attention with separate q/k/v/output projections (matching checkpoint)"""
    def __init__(self, embed_dim=1024, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
        )
        self.output_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
        )
        self.attn = MMEncoderAttention(
            num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=num_heads,
        )
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_heads * self.head_dim

    def forward(self, x):
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.output_proj(attn_output)
        return output


class Wav2Vec2FFN(nn.Module):
    def __init__(self, embed_dim=1024, ffn_dim=4096):
        super().__init__()
        self.inner_proj = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.output_proj = nn.Linear(ffn_dim, embed_dim, bias=True)

    def forward(self, x):
        x = self.inner_proj(x)
        x = nn.functional.gelu(x)
        x = self.output_proj(x)
        return x


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, embed_dim=1024, ffn_dim=4096, num_heads=16):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = Wav2Vec2Attention(embed_dim, num_heads)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = Wav2Vec2FFN(embed_dim, ffn_dim)

    def forward(self, x):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + x
        return x


class Wav2Vec2Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor()
        self.post_extract_layer_norm = nn.LayerNorm(512)
        self.model_dim_proj = nn.Linear(512, 1024, bias=True)
        # pos_encoder: store as plain conv, handle weight_norm in weight loading
        self.pos_encoder = nn.ModuleDict({
            "conv": nn.utils.weight_norm(
                nn.Conv1d(1024, 1024, kernel_size=128, padding=64, groups=16, bias=True),
                name="weight",
                dim=2
            )
        })

    def forward(self, audio):
        x = self.feature_extractor(audio)
        x = x.transpose(1, 2)
        x = self.post_extract_layer_norm(x)
        x = self.model_dim_proj(x)
        pos = self.pos_encoder["conv"](x.transpose(1, 2))
        pos = pos[:, :, :x.shape[1]]
        x = x + pos.transpose(1, 2)
        return x

class Wav2Vec2TransformerEncoder(nn.Module):
    """encoder.* keys"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            Wav2Vec2EncoderLayer(embed_dim=1024, ffn_dim=4096, num_heads=16)
            for _ in range(24)
        ])
        self.layer_norm = nn.LayerNorm(1024)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x
