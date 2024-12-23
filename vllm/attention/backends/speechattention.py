import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer with relative position encoding."""
    def __init__(self, n_head, n_feat, dropout_rate, query_bias=True, key_bias=True, value_bias=True, use_sdpa=False, n_kv_head=None, head_dim=None):
        super(RelPositionMultiHeadedAttention, self).__init__()
        self.n_head = n_head
        self.d_k = n_feat // n_head
        self.h = n_head
        self.dropout_rate = dropout_rate
        self.linear_q = nn.Linear(n_feat, n_feat, bias=query_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=value_bias)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward(self, query, key, value, mask=torch.ones((0, 0, 0), dtype=torch.bool), pos_emb=torch.empty(0), cache=(torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0)))):
        q, k, v = self.linear_q(query), self.linear_k(key), self.linear_v(value)
        q = q.view(q.size(0), q.size(1), self.h, self.d_k).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.h, self.d_k).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.h, self.d_k).transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if not self.use_sdpa:
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
            if mask.size(0) > 0:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout_rate, training=self.training)
            output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(query.size(0), -1, self.h * self.d_k)
            return self.linear_out(output), cache
        else:
            assert mask.dtype != torch.bool
            mask = mask.unsqueeze(1)
            mask = (matrix_bd + mask) / math.sqrt(self.d_k)
            output = F.scaled_dot_product_attention(q_with_bias_u, k, v, attn_mask=mask, dropout_p=self.dropout_rate, scale=1 / math.sqrt(self.d_k))
            output = output.transpose(1, 2).contiguous().view(query.size(0), -1, self.h * self.d_k)
            return self.linear_out(output), cache

class FastSpeech2ConformerAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super(FastSpeech2ConformerAttention, self).__init__()
        self.attention = RelPositionMultiHeadedAttention(n_head, n_feat, dropout_rate)

    def forward(self, x, pos_emb):
        output, _ = self.attention(query=x, key=x, value=x, pos_emb=pos_emb)
        return output

import torch

# Example usage of the FastSpeech2ConformerAttention class
n_head = 8  # Number of attention heads
n_feat = 512  # Feature dimension
dropout_rate = 0.1  # Dropout rate

# Initialize the FastSpeech2ConformerAttention layer
attention_layer = FastSpeech2ConformerAttention(n_head, n_feat, dropout_rate)

# Create some dummy input data
batch_size = 4  # Number of sequences in a batch
sequence_length = 20  # Length of each sequence
input_tensor = torch.rand(batch_size, sequence_length, n_feat)  # Input tensor
pos_emb = torch.rand(sequence_length, n_feat)  # Positional embeddings

# Forward pass through the attention layer
output = attention_layer(input_tensor, pos_emb)

print(output.shape)  # Should print torch.Size([4, 20, 512])

# Example usage
