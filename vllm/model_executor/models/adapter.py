import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

class AutoMTPSizeHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            # nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits_output = self.mlp(hidden_states)
        return logits_output



class AutoMTPSizeHeadV2(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False)
        )
        self.norm =  Qwen3RMSNorm(hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, 5, bias=False)

    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states) + hidden_states
        hidden_states = self.norm(hidden_states)
        logits_output = self.classifier(hidden_states)
        return logits_output

# TODO: try different stop head frameworks

# ============== 消融实验：不同层数 & 结构的 MLP ==============
# 
# 层数消融：
#   AutoMTPStopHeadSimple:  0层 MLP (只有 classifier)
#   AutoMTPStopHead:        2层 MLP (平行投影 dim→dim→dim)
#   AutoMTPStopHeadDeep:    4层 MLP (平行投影)
#
# 结构消融 (Bottleneck vs 平行)：
#   AutoMTPStopHeadBottleneck:      2层 MLP (bottleneck: dim→dim/4→dim)
#   AutoMTPStopHeadDeepBottleneck:  4层 MLP (bottleneck)

class AutoMTPStopHeadSimple(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = Qwen3RMSNorm(hidden_size=hidden_size + 1)
        self.classifier = nn.Linear(hidden_size + 1, 1, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        logits_output = self.classifier(hidden_states)
        return logits_output


# TODO: 看下是否编译，改异步计算
class AutoMTPStopHeadMid(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size + 1, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size + 1, hidden_size + 1, bias=False)
        )
        self.norm = Qwen3RMSNorm(hidden_size=hidden_size + 1)
        self.classifier = nn.Linear(hidden_size + 1, 1, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states) + hidden_states
        hidden_states = self.norm(hidden_states)
        logits_output = self.classifier(hidden_states)
        return logits_output


class AutoMTPStopHeadDeep(nn.Module):
    """更复杂的 Bottleneck 版本：4层 MLP (bottleneck) + 多残差
    
    对比 AutoMTPStopHeadDeep，验证深层网络中 bottleneck 的效果
    """
    def __init__(self, hidden_size: int, bottleneck_ratio: int = 4):
        super().__init__()
        dim = hidden_size + 1
        bottleneck_dim = dim // bottleneck_ratio
        
        # 第一个 block: bottleneck MLP + 残差
        self.proj1 = nn.Sequential(
            nn.Linear(dim, bottleneck_dim, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, dim, bias=False)
        )
        self.norm1 = Qwen3RMSNorm(hidden_size=dim)
        
        # 第二个 block: bottleneck MLP + 残差
        self.proj2 = nn.Sequential(
            nn.Linear(dim, bottleneck_dim, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, dim, bias=False)
        )
        self.norm2 = Qwen3RMSNorm(hidden_size=dim)
        
        self.classifier = nn.Linear(dim, 1, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Block 1
        hidden_states = self.proj1(hidden_states) + hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Block 2
        hidden_states = self.proj2(hidden_states) + hidden_states
        hidden_states = self.norm2(hidden_states)
        
        logits_output = self.classifier(hidden_states)
        return logits_output

class AutoMTPStopHeadV3(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        bottleneck_size = (hidden_size + 1) // 4
        self.proj = nn.Sequential(
            nn.Linear(hidden_size + 1, bottleneck_size, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck_size, hidden_size + 1, bias=False)
        )
        self.norm = Qwen3RMSNorm(hidden_size=hidden_size + 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 1, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, 1, bias=False)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        x = self.proj(hidden_states)
        x = self.norm(x + residual)
        
        logits = self.classifier(x)
        return logits


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x: [B, N, D]
        """
        weights = self.attn(x)          # [B, N, 1]
        weights = torch.softmax(weights, dim=1)
        return (weights * x).sum(dim=1) # [B, D]

class HiddenTrajectoryEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

        self.conv1 = nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self.pool = AttentionPooling(out_dim)

    def forward(self, h_traj):
        """
        h_traj: [B, N, D]
        """
        h = self.norm(h_traj)
        h = h.transpose(1, 2)           # [B, D, N]

        h = self.act(self.conv1(h))
        h = self.dropout(h)
        h = self.act(self.conv2(h))

        h = h.transpose(1, 2)           # [B, N, out_dim]
        z = self.pool(h)                # [B, out_dim]
        return z

class LogitsEncoder(nn.Module):
    def __init__(self, topk=32, out_dim=128):
        super().__init__()
        self.topk = topk

        self.norm = nn.LayerNorm(topk)
        self.mlp = nn.Sequential(
            nn.Linear(topk, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, logits):
        """
        logits: [B, V]
        """
        # NOTE: logits is [B, Seq_len, Vocab_size]
        logits = logits.squeeze(1)
        topk_logits, _ = torch.topk(logits, self.topk, dim=-1)

        topk_logits = torch.sort(topk_logits, descending=True).values

        x = self.norm(topk_logits)
        return self.mlp(x)

class AttentionEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        window_size=256,
        out_dim=64
    ):
        super().__init__()
        self.window_size = window_size
        self.out_dim = out_dim

        # per-(layer, head) encoder
        self.per_head_mlp = nn.Sequential(
            nn.Linear(window_size, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim)
        )

        # attention over layer-head pairs
        self.lh_attn = nn.Linear(out_dim, 1)

    def forward(self, attn_full):
        """
        attn_full: [B, N_layers, H, K, K] or [B, H, K, K]
        """
        # Handle both 4D and 5D inputs
        if attn_full.dim() == 4:
            # Input is [B, H, K, K], add layer dimension: [B, 1, H, K, K]
            attn_full = attn_full.unsqueeze(1)
        
        # attn_full is now [B, N_layers, H, K, K]
        # 2. 只取 query = last token
        attn = attn_full[:, :, :, -1, :]             # [B, L', H, K]

        # 3. 只保留最近 window_size 个 key
        attn = attn[:, :, :, -self.window_size:]  # [B, L', H, M] [4, 1, 32, 256]

        B, Lp, H, M = attn.shape

        # 4. flatten layer-head
        x = attn.reshape(B * Lp * H, M)

        # 5. per-head encoding
        x = self.per_head_mlp(x)               # [B*L'*H, out_dim]
        x = x.view(B, Lp * H, self.out_dim)    # [B, L'*H, out_dim] [4, 32, 64]

        # 6. attention pooling over (layer, head)
        weights = self.lh_attn(x)               # [B, L'*H, 1] [4, 32, 1]
        weights = torch.softmax(weights, dim=1)

        z = (weights * x).sum(dim=1)            # [B, out_dim]
        return z


class AutoMTPStopHeadV4(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 1, traj_len: int = 8, topk: int = 32, attn_window: int = 16):
        super().__init__()
        self.traj_len = traj_len

        self.hidden_encoder = HiddenTrajectoryEncoder(
            hidden_dim=hidden_size,
            out_dim=256
        )

        self.logits_encoder = LogitsEncoder(
            topk=topk,
            out_dim=128
        )

        # self.attn_encoder = AttentionEncoder(
        #     num_layers=1,
        #     num_heads=num_heads,
        #     window_size=attn_window,
        #     out_dim=64
        # )

        fusion_dim = 256 + 128

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states,
        logits,
    ):
        """
        hidden_states: [mtp_size, valid_len, H] 
        logits:        [mtp_size, valid_len, V]    
        """

        h_traj = hidden_states[:, -self.traj_len:, :]

        z_h = self.hidden_encoder(h_traj) # [mtp_size, 256]
        z_l = self.logits_encoder(logits) # [mtp_size, 128]

        # z = torch.cat([z_h, z_l, z_a], dim=-1)
        z = torch.cat([z_h, z_l], dim=-1)
        return self.classifier(z)


class SVIPStopHead():
    def __init__(self, hidden_size: int):
        pass

    def __call__(self, logits: torch.Tensor, threshold: float = 0.3) -> bool:
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, dtype=torch.float32)

        condition = torch.sqrt(entropy) > threshold
        
        return condition.item() if condition.dim() == 0 else condition
