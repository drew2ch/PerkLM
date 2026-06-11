""" Decoder-only, GPT-style Transformer Implementation.

    Author: Andrew Chung
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

def precompute_rope_freqs(d_head: int, seq_len: int, base: float = 10000.0, device = None):
    """ Compute complex frequency tensor for RoPE
        Returns:
            freqs_cis: (seq_len, d_head // 2);
            freqs_cis[m, i] = exp(i*m*theta_i)
    """
    # theta_i = base^{-2i/d}, shape (head_dim // 2,)
    i = torch.arange(0, d_head, 2, dtype = torch.float32, device = device)
    theta = 1.0 / (base ** (i / d_head))

    # outer product: m * theta_i -> angle m&theta_i
    positions = torch.arange(seq_len, dtype = torch.float32, device = device)
    angles = torch.outer(positions, theta)

    # exp(i * angle) = cos(angle) + i*sin(angle)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """ Apply RoPE to Q/K tensors.
        Args:
            x: (batch, seq_len, n_heads, d_head)
            freqs_cis: (seq_len, d_head // 2)
        Returns:
            x_rot: (batch, seq_len, n_heads, d_head)
    """
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)

    # Broadcast freqs_cis over batch and heads
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Elementwise rotation (complex mult)
    x_rot = x_complex * freqs

    # Back to real
    x_out = torch.view_as_real(x_rot) # (B, T, H, d/2, 2)
    x_out = x_out.reshape(*x.shape)   # (B, T, H, d)
    
    return x_out.type_as(x)
    

class DialogueEmbedding(nn.Module):
    """ Embedding Layer for Friends Dialogue Corpus
        Note: I implemented RoPE in lieu of sinusoidal position embeddings
        in the Attention Head.
    """
    def __init__(self, tokenizer, d_model, maxt = 128):
        
        super().__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.maxt = maxt

        self.embedding = nn.Embedding(num_embeddings = len(self.tokenizer),
                                      embedding_dim = self.d_model)

    def forward(self, batch):
        """ Params
                batch: DataLoader batch
        """

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # batch-wise mask trimming
        max_batch_length = attention_mask.sum(dim = 1).max().item()
        if max_batch_length > self.maxt:
            print(f'Warning: max_batch_length {max_batch_length} > maxt {self.maxt}')
            input_ids = input_ids[:, :self.maxt]
            attention_mask = attention_mask[:, :self.maxt]
        else:
            input_ids = input_ids[:, :max_batch_length]
            attention_mask = attention_mask[:, :max_batch_length]
        
        embeddings = self.embedding(input_ids)

        return embeddings, attention_mask
    
class DialogueMultiHeadAttention(nn.Module):
    """ Transformer Decoder Block w/ RoPE
    """
    def __init__(self, d_model: int, n_heads: int, base: float = 10000.0, dropout: float = 0.2, maxt: int = 128):
        
        super().__init__()
        assert d_model % n_heads == 0, \
            f'Error: d_model {d_model} must be divisible by n_heads {n_heads}'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.dropout = dropout
        self.maxt = maxt

        # vectorized Q, K, V
        self.w_q = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias = False)
        self.scale = 1 / math.sqrt(self.d_head)

        freqs_cis = precompute_rope_freqs(self.d_head, self.maxt, base = base)
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(self, embedding, attention_mask):
        
        assert embedding.shape[-1] == self.d_model, \
            f'Error: embedding dimension {embedding.shape[-1]} and d_model {self.d_model} must match'
        
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        MASK = (1 - attention_mask.float()) * -1e9
        B, T = embedding.shape[:2]
        
        Q = self.w_q(embedding).view(B, T, self.n_heads, self.d_head)
        K = self.w_k(embedding).view(B, T, self.n_heads, self.d_head)
        V = self.w_v(embedding).view(B, T, self.n_heads, self.d_head)
        
        freqs = self.freqs_cis[:T]
        Q_rope = apply_rope(Q, freqs).transpose(1, 2)
        K_rope = apply_rope(K, freqs).transpose(1, 2)
        V = V.transpose(1, 2)
        A = (Q_rope @ K_rope.transpose(-2, -1)) * self.scale

        # Masked Multi-Head Attention
        X = A.size(-1)
        mask = torch.triu(torch.ones(X, X,device = embedding.device), diagonal = 1).bool()
        A = A.masked_fill(mask, float('-inf')) + MASK

        A = F.softmax(A, dim = -1)
        A = F.dropout(A, p = self.dropout, training = self.training)
        attention = A @ V

        attn_out = attention.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(attn_out)
    
class DialogueDecoderLayer(nn.Module):
    """ Transformer Decoder Block for Dialogue Embeddings
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.2, maxt: int = 128):

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.maxt = maxt

        self.attn = DialogueMultiHeadAttention(self.d_model,
                                               self.n_heads,
                                               dropout = self.dropout,
                                               maxt = self.maxt)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.ff1 = nn.Linear(self.d_model, self.d_ff)
        self.ff2 = nn.Linear(self.d_ff, self.d_model)
        
    def forward(self, embedding, attention_mask):
        """ Pre-LN (GPT) Add/Norm and FFN
        """

        attn_out = self.attn(self.norm1(embedding), attention_mask)
        x = embedding + F.dropout(attn_out, p = self.dropout, training = self.training)
        ff_out = self.ff2(F.dropout(F.gelu(
            self.ff1(self.norm2(x))), p = self.dropout, training = self.training))
        x = x + F.dropout(ff_out, p = self.dropout, training = self.training)

        return x
    
class FriendsTransformer(nn.Module):
    """ Master Class encompassing the Embedding Layer plus
        a stack of DialogueDecoderLayers.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float = 0.2, maxt = 128, tokenizer = None):

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.maxt = maxt
        self.tokenizer = tokenizer

        self.embedder = DialogueEmbedding(self.tokenizer, self.d_model, self.maxt)
        self.decoder = nn.ModuleList([DialogueDecoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout, self.maxt)\
                                      for _ in range(self.n_layers)])
        self.final_norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, len(self.tokenizer), bias = False)
        self.lm_head.weight = self.embedder.embedding.weight

    def forward(self, batch):
        """ Params
            batch: DataLoader batch
        """

        embedding, attention_mask = self.embedder(batch)
        x = embedding
        for layer in self.decoder:
            x = layer(x, attention_mask)
        logits = self.lm_head(self.final_norm(x))

        return logits
