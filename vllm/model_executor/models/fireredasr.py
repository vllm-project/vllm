import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, d_model: int, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = ReplicatedLinear(
            input_size=out_channels * subsample_idim,
            output_size=d_model,
            bias=True,
        )

        self.subsampling = 4
        left_context = right_context = 3  # both exclude current frame
        self.context = left_context + 1 + right_context  # 7

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.conv(x)
        N, C, T, D = x.size()
        x, _ = self.out(x.transpose(1, 2).contiguous().view(N, T, C * D))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)).item() / d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self.pe = torch.cat([pe_positive, pe_negative], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tmax = 2 * max_len - 1
        Tmax, T = self.pe.size(1), x.size(1)
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T].clone().detach()
        return pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.linear_expand = ReplicatedLinear(
            input_size=d_model,
            output_size=d_model * 4,
            bias=True,
        )
        self.nonlinear = Swish()
        self.linear_project = ReplicatedLinear(
            input_size=d_model * 4,
            output_size=d_model,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_layer_norm(x)
        x, _ = self.linear_expand(x)
        x = self.nonlinear(x)
        x, _ = self.linear_project(x)
        output = x + residual
        return output


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = ReplicatedLinear(
            input_size=d_model, output_size=n_head * self.d_k, bias=False
        )
        self.w_ks = ReplicatedLinear(
            input_size=d_model, output_size=n_head * self.d_k, bias=False
        )
        self.w_vs = ReplicatedLinear(
            input_size=d_model, output_size=n_head * self.d_v, bias=False
        )

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.fc = ReplicatedLinear(
            input_size=n_head * self.d_v, output_size=d_model, bias=False
        )

    def forward_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q)[0].view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k)[0].view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v)[0].view(sz_b, len_v, n_head, d_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_output(
        self, output: torch.Tensor, residual: torch.Tensor, sz_b: int, len_q: int
    ) -> torch.Tensor:
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        fc_out, _ = self.fc(output)
        output = fc_out
        output = output + residual
        return output

    def forward_attention(
        self, attn: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -float("inf"))
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)

        d_attn = attn
        output = torch.matmul(d_attn, v)

        return output, attn


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head: int, d_model: int):
        super().__init__(n_head, d_model)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k**0.5)
        self.linear_pos = ReplicatedLinear(
            input_size=d_model, output_size=n_head * d_k, bias=False
        )
        self.pos_bias_u = nn.Parameter(torch.empty([n_head, d_k]))
        self.pos_bias_v = nn.Parameter(torch.empty([n_head, d_k]))

    def _rel_shift(self, x):
        N, H, T1, T2 = x.size()
        zero_pad = torch.zeros((N, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(N, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)
        x = x[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sz_b, len_q = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        q = q.transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb)[0].view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)

        attn_scores = matrix_ac + matrix_bd
        attn_scores.mul_(self.scale)

        output, attn = self.forward_attention(attn_scores, v, mask=mask)

        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 33):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 4, kernel_size=1, bias=False
        )
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model * 2,
            d_model * 2,
            kernel_size,
            stride=1,
            padding=self.padding,
            groups=d_model * 2,
            bias=False,
        )
        self.batch_norm = nn.LayerNorm(d_model * 2)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            d_model * 2, d_model, kernel_size=1, bias=False
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        out = self.pre_layer_norm(x)
        out = out.transpose(1, 2)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)

        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)

        out = self.pointwise_conv2(out)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = out.transpose(1, 2)
        return out + residual


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model, n_head, kernel_size=33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model)
        self.conv = ConformerConvolution(d_model, kernel_size)
        self.ffn2 = ConformerFeedForward(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        slf_attn_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        idim: int,
        n_layers_enc: int,
        n_head: int,
        d_model: int,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
    ):
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)

        self.layer_stack = nn.ModuleList()
        for _ in range(n_layers_enc):
            block = RelPosEmbConformerBlock(d_model, n_head, kernel_size)
            self.layer_stack.append(block)

    def forward(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor, pad: bool = True
    ):
        if pad:
            padded_input = F.pad(
                padded_input,
                (0, 0, 0, self.input_preprocessor.context - 1),
                "constant",
                0.0,
            )
        src_mask = self.padding_position_is_0(padded_input, input_lengths)

        embed_output, input_lengths, src_mask = self.input_preprocessor(
            padded_input, src_mask
        )
        enc_output = embed_output

        pos_emb = self.positional_encoding(embed_output)

        enc_outputs = []
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
            )
            enc_outputs.append(enc_output)

        return enc_output, input_lengths, src_mask

    def padding_position_is_0(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        N, T = padded_input.size()[:2]
        mask = torch.ones((N, T)).to(padded_input.device)
        for i in range(N):
            mask[i, input_lengths[i] :] = 0
        mask = mask.unsqueeze(dim=1)
        return mask.to(torch.uint8)
