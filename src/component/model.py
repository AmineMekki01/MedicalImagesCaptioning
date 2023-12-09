"""
Script : 
    model.py

Description : 
    The 'model.py' module defines a custom implementation of the GPT2 model combined with a Vision Transformer (ViT), for the task of medical image captioning. This module comprises several classes that together construct a model capable of understanding and generating textual descriptions based on visual input.

Classes:
    GPT2Attention: Implements the attention mechanism used in GPT2.
        - __init__(self, config): Initializes the GPT2Attention module with the specified configuration.
        - forward(self, x): Forward pass for the attention mechanism.

    GPT2CrossAttention: Implements a cross-attention mechanism for integrating visual features from the Vision Transformer with textual features from GPT2.
        - __init__(self, config): Initializes the GPT2CrossAttention module.
        - forward(self, query, key, value): Forward pass for the cross-attention mechanism.

    GPT2MLP: Implements the multi-layer perceptron used in GPT2.
        - __init__(self, config): Initializes the GPT2MLP module.
        - forward(self, x): Forward pass for the MLP.

    GPT2Block: Represents a single block in the GPT2 model, combining attention, cross-attention, and MLP mechanisms.
        - __init__(self, config): Initializes a GPT2Block with the given configuration.
        - forward(self, x, enc_out): Forward pass for the block, processing both textual and visual inputs.

    VisionGPT2Model: The main model class that combines GPT2 with a Vision Transformer for image captioning.
        - __init__(self, config): Initializes the model with the specified configuration.
        - _pos_embed(self, x): Applies positional embeddings to the input.
        - pretrained_layers_trainable(self, trainable): Sets the trainability of the pretrained layers.
        - unfreeze_gpt_layers(self): Unfreezes the GPT2 layers for fine-tuning.
        - from_pretrained(self, config): Initializes the model with pretrained weights from a standard GPT2 model.
        - forward(self, image, input_ids, labels): Forward pass for the model, processing both image and text inputs.
        - generate(self, image, sequence, max_tokens, temperature, deterministic): Generates captions for a given image.

Dependencies:
    - torch: For constructing neural network layers and tensor operations.
    - transformers: Provides access to GPT2LMHeadModel and GPT2TokenizerFast.
    - timm: For creating the Vision Transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


class GPT2Attention(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'Embedding dimension must be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.c_attn = nn.Linear(
            self.embed_dim, self.head_size * self.n_heads * 3, bias=True)
        self.scale = self.head_size ** -0.5

        self.register_buffer('mask', torch.tril(
            torch.ones(1, 1, self.seq_len, self.seq_len)))

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, feature_size = x.shape
        query, key, value = self.c_attn(x).chunk(3, dim=-1)
        query = query.view(batch_size, sequence_len, self.n_heads, self.head_size).permute(
            0, 2, 1, 3) 
        key = key.view(batch_size, sequence_len, self.n_heads,
                       self.head_size).permute(0, 2, 1, 3)
        value = value.view(batch_size, sequence_len,
                           self.n_heads, self.head_size).permute(0, 2, 1, 3)

        qk_t = (query@key.transpose(-2, -1)) * self.scale
        qk_t = qk_t.masked_fill(
            self.mask[:, :, :sequence_len, :sequence_len] == 0, float('-inf'))
        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ value  
        attention = attention.permute(0, 2, 1, 3).contiguous().view(
            batch_size, sequence_len, feature_size) 

        out = self.c_proj(attention)
        out = self.resid_dropout(out)

        return out

class GPT2CrossAttention(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'Embedding dimension must be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_size ** -0.5

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, feature_size = query.shape

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(batch_size, query.size(1), self.n_heads, self.head_size).permute(
            0, 2, 1, 3) 
        key = key.view(batch_size, key.size(1), self.n_heads,
                       self.head_size).permute(0, 2, 1, 3)
        value = value.view(batch_size, value.size(
            1), self.n_heads, self.head_size).permute(0, 2, 1, 3)

        qk_t = (query@key.transpose(-2, -1)) * self.scale
        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ value  
        attention = attention.permute(0, 2, 1, 3).contiguous().view(
            batch_size, sequence_len, feature_size) 

        out = self.c_proj(attention)
        out = self.resid_dropout(out)

        return out

class GPT2MLP(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout

        self.c_fc = nn.Linear(self.embed_dim, self.embed_dim*self.mlp_ratio)
        self.c_proj = nn.Linear(self.embed_dim*self.mlp_ratio, self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPT2Block(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        x = x+self.attn(self.ln_1(x))
        x = x+self.cross_attn(self.ln_2(x), enc_out, enc_out)
        x = x+self.mlp(self.ln_3(x))
        return x
