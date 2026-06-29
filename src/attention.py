import torch
import torch.nn as nn
from config import D_MODEL, N_HEADS, DROPOUT

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.dropout = nn.Dropout(DROPOUT)


        # Instead of N_HEADS separate query matrices (one per head) as proposed in the original 
        # "Attention is All You Need" paper, we use a single large linear layer to project the input 
        # into a single query matrix that we can then split afterwards before computing attention.
        # This is more efficient and allows for better parameter sharing across heads.
        self.W_Q = nn.Linear(D_MODEL, D_MODEL) 
        self.W_K = nn.Linear(D_MODEL, D_MODEL)
        self.W_V = nn.Linear(D_MODEL, D_MODEL)
        # Output projection layer to combine the outputs of all heads back into a single meaningful representation
        self.W_O = nn.Linear(D_MODEL, D_MODEL) 


    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Compute the query, key and value matrices
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape Q, K, V to (batch_size, n_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # shape after: (batch_size, n_heads, seq_len, head_dim)

        # Compute the attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 2)  # shape: (batch_size, n_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # Apply the attention weights to the value matrix
        out = torch.matmul(attention_weights, V)  # shape: (batch_size, n_heads, seq_len, head_dim)

        # Reshape the output back to (batch_size, seq_len, D_MODEL)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.n_heads * self.head_dim)

        # Project the output back to D_MODEL
        out = self.W_O(out)

        return out