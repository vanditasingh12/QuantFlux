import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        groups = min(8, in_channels) if in_channels >= 8 else 1
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1)
        )

        out_groups = min(8, out_channels) if out_channels >= 8 else 1
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(out_groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[..., None]
        h = self.block2(h)
        return h + self.residual_conv(x)