from functools import partial
import torch
from torch import nn
from torch.nn import Module, LayerNorm

# UPGRADE: Use standard Linear for speed and fusion compatibility
class PreNormResidual(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

# UPGRADE: Fused FeedForward using Linear
# Conv1d(1) is mathematically same as Linear, but Linear is often 
# more optimized for the last dimension in PyTorch.
class FeedForward(Module):
    def __init__(self, dim, dim_hidden, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MLPMixer1D(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        expansion_factor = 4,
        expansion_factor_token = 0.5,
        dropout = 0.
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Token Mixing: (B, N, D) -> Mix N
                PreNormResidual(dim, nn.Sequential(
                    # Transpose to (B, D, N) to mix N via Linear
                    TransposeLast(), 
                    nn.Linear(seq_len, int(expansion_factor * seq_len)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(expansion_factor * seq_len), seq_len),
                    nn.Dropout(dropout),
                    TransposeLast() # Back to (B, N, D)
                )),
                # Channel Mixing: (B, N, D) -> Mix D
                PreNormResidual(dim, FeedForward(dim, int(expansion_factor_token * dim), dropout))
            ]))
            
        self.norm = LayerNorm(dim)

    def forward(self, x):
        for token_mix, channel_mix in self.layers:
            x = token_mix(x)
            x = channel_mix(x)
        return self.norm(x)

# Helper for clean transposes
class TransposeLast(Module):
    def forward(self, x):
        return x.transpose(-1, -2)

# Quick test
if __name__ == '__main__':
    tokens = torch.randn(1, 1024, 512)
    mixer = MLPMixer1D(dim = 512, depth = 4, seq_len = 1024)
    out = mixer(tokens)
    assert out.shape == tokens.shape
    print("MLPMixer1D optimized forward pass successful.")