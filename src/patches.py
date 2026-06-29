import torch
import torch.nn as nn
from config import PATCH_SIZE, NUM_PATCHES, D_MODEL, DROPOUT

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels = 3,
            out_channels = D_MODEL,
            kernel_size = PATCH_SIZE,
            stride = PATCH_SIZE
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL)) # nn.Parameter because the cls token is learnable
        self.position_embedding = nn.Parameter(torch.randn(1, NUM_PATCHES + 1, D_MODEL)) # +1 for the cls token

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x) # (batch_size, D_MODEL, num_patches_h, num_patches_w)
        x = x.flatten(2) # (batch_size, D_MODEL, num_patches)
        x = x.transpose(1, 2) # (batch_size, num_patches, D_MODEL)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) #(batch_size, 1, D_MODEL)
        x = torch.cat((cls_tokens, x), dim=1) # (batch_size, num_patches + 1, D_MODEL)
        x = x + self.position_embedding # (batch_size, num_patches + 1, D_MODEL)
        x = self.dropout(x)
        return x

