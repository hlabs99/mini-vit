
from config import D_MODEL, MLP_DIM, DROPOUT
from src.attention import MultiHeadAttention

import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attention = MultiHeadAttention()
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, MLP_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_DIM, D_MODEL),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
