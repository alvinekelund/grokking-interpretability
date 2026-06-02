import torch
import torch.nn as nn


class Grokker(nn.Module):
    def __init__(self, p, d_model=128, n_heads=4, d_mlp=512):
        super().__init__()
        self.p = p
        # p+1 tokens: numbers 0..p-1, plus '=' at index p
        self.embed = nn.Embedding(p + 1, d_model)
        self.pos_embed = nn.Embedding(3, d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, x):
        # x: (batch, 3) — tokens [a, b, =]
        pos = torch.arange(3, device=x.device)
        h = self.embed(x) + self.pos_embed(pos)
        h2, _ = self.attn(h, h, h)
        h = self.ln1(h + h2)
        h = self.ln2(h + self.mlp(h))
        return self.unembed(h[:, 2])  # predict from the '=' position
