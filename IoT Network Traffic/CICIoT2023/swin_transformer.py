import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """Window-based self-attention mechanism."""
    def __init__(self, dim, num_heads, dropout=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch_size, num_windows * window_size, dim)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, N, num_heads, dim_head)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # Scaled dot-product attention
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Combine heads
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    """A single Swin Transformer block."""
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, dropout=0.0):
        super(SwinBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, C = x.shape
        # Cyclic shift
        x = x.roll(shifts=-self.shift_size, dims=1) if self.shift_size > 0 else x

        # Self-attention within windows
        x = self.norm1(x)
        x = self.attn(x) + x

        # MLP
        x = self.norm2(x)
        x = self.mlp(x) + x
        return x


class SwinTimeSeriesTransformer(nn.Module):
    """Swin Transformer for time series."""
    def __init__(self, input_dim, seq_len, patch_size, num_classes, dim, depth, num_heads, window_size, mlp_ratio=4.0, dropout=0.1):
        super(SwinTimeSeriesTransformer, self).__init__()

        assert seq_len % patch_size == 0, "Sequence length must be divisible by patch size"
        self.num_patches = seq_len // patch_size

        # Patch embedding
        self.patch_embedding = nn.Conv1d(input_dim, dim, kernel_size=patch_size, stride=patch_size)

        # Stacked Swin Transformer blocks
        self.swin_blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for i in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # Patch embedding
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = self.patch_embedding(x)  # (batch_size, dim, num_patches)
        x = x.permute(0, 2, 1)  # (batch_size, num_patches, dim)

        # Apply Swin Transformer blocks
        for block in self.swin_blocks:
            x = block(x)  # (batch_size, num_patches, dim)

        # Global average pooling and classification
        x = x.mean(dim=1)  # (batch_size, dim)
        x = self.norm(x)
        x = self.fc(x)  # (batch_size, num_classes)

        return x