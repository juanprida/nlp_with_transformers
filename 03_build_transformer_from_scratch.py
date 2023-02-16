# Code implementation of the paper "Attention is all you need" for the encoder/decoder architecture.
# Author: Juan Prida

import torch
import torch.nn as nn

from dataclasses import dataclass


@dataclass
class config:
    # Model hyperparameters.
    vocab_size: int = 1000
    embed_dim: int = 768
    n_head: int = 3
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 3072
    dropout: float = 0.1
    apply_mask: bool = True

    # Data hyperparameters.
    max_input_length: int = 13
    max_target_length: int = 1


class AttentionHead(nn.Module):
    """Single attention head."""

    def __init__(self, config):
        super().__init__()
        self.query, self.key, self.value = nn.ModuleList(
            [nn.Linear(config.embed_dim, config.embed_dim // config.n_head) for _ in range(3)]
        )
        self.d_k = config.embed_dim

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        qk = q @ k.transpose(1, 2) / self.d_k**0.5
        # Masking the attention weights to prevent the model from attending to future tokens.
        if config.apply_mask:
            mask = torch.tril(torch.ones(qk.shape[-2], qk.shape[-1])).bool()
            qk = qk.masked_fill(mask, float("-inf"))
        qk_scaled = torch.softmax(qk, dim=-1)
        weights = qk_scaled @ v
        return weights


class MultiHeadAttention(nn.Module):
    """Combine multiple attention heads into one attention layer."""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_head)])
        self.linear = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x):
        attn = [head(x) for head in self.heads]
        concat = torch.cat(attn, dim=-1)
        output = self.linear(concat)
        return output


class FeedForward(nn.Module):
    """
    Two layer fully connected network using GELU activation and dropout.

    Note that when we apply a linear layer, the transformations will treat
    sequences and samples in an independently manner.
    """

    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.embed_dim, config.dim_feedforward)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(config.dim_feedforward, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Combine attention with fully connected layers adding layer normalization and skipconnections."""

    def __init__(self, config):
        super().__init__()
        self.mhattn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.layer_norm_1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        x = x + self.mhattn(self.layer_norm_1(x))
        x = x + self.ff(self.layer_norm_2(x))
        return x


class Transformer(nn.Module):
    """Combine multiple blocks to form a transformer encoder."""

    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_embeddings = nn.Embedding(config.max_input_length, config.embed_dim)
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_encoder_layers)])

    def forward(self, input_ids):
        # Compute embeddings.
        token_embedding = self.token_embeddings(input_ids)
        # Compute positional embeddings.
        positional_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        positional_embedding = self.positional_embeddings(positional_ids)
        # Add embeddings and apply layer norm + dropout.
        embeddings = token_embedding + positional_embedding
        x = self.dropout(self.layer_norm(embeddings))
        # Pass embeddings through the encoder.
        for block in self.blocks:
            x = block(x)
        return x

