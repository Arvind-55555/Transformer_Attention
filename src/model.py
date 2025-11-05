"""
model.py
---------
Modular Transformer implementation (small, educational) with clear mapping
between math (from "Attention Is All You Need") and code.

This implementation returns attention weights for visualization.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    PE(pos,2i) = sin(pos / 10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
    Adds positional information to token embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V
    Code computes attention scores and returns output and attention weights.
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, heads, seq_len, d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, heads, seq_q, seq_k)
        if mask is not None:
            # mask expected to be broadcastable to scores
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)  # (batch, heads, seq_len, d_k)
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head1...headh) W_o
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final linear
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def split_heads(self, x):
        # x shape: (batch, seq_len, d_model) -> (batch, heads, seq_len, d_k)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        # x shape: (batch, heads, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous()
        batch_size, seq_len, _, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads * self.d_k)

    def forward(self, query, key, value, mask=None):
        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        # Scaled dot-product attention
        out, attn = self.attention(Q, K, V, mask=mask)
        # Combine heads
        out = self.combine_heads(out)
        # Final linear projection
        out = self.w_o(out)
        return out, attn  # attn shape: (batch, heads, seq_q, seq_k)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network:
    FFN(x) = max(0, xW1 + b1) W2 + b2
    Implemented with two linear layers and ReLU activation.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    One encoder layer: Multi-Head Attention -> Add & Norm -> FeedForward -> Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention (query=key=value=x)
        attn_out, attn = self.mha(x, x, x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x, attn


class DecoderLayer(nn.Module):
    """
    One decoder layer: Masked MHA -> Add & Norm -> MHA(encoder-decoder) -> Add & Norm -> FF -> Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # Masked self-attention
        masked_out, masked_attn = self.masked_mha(x, x, x, mask=tgt_mask)
        x = x + self.dropout(masked_out)
        x = self.norm1(x)

        # Encoder-Decoder attention
        encdec_out, encdec_attn = self.mha(x, enc_out, enc_out, mask=src_mask)
        x = x + self.dropout(encdec_out)
        x = self.norm2(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x, masked_attn, encdec_attn


class SmallTransformer(nn.Module):
    """
    Small Transformer wrapper with N encoder and decoder layers.
    Exposes attention weights for visualization as a list of tensors.
    """
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_encoder_layers=2, num_decoder_layers=2, d_ff=512, max_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])
        self.out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        # mask for decoder to prevent attending to future tokens
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool().to(next(self.parameters()).device)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: (batch, src_seq), tgt: (batch, tgt_seq)
        src_emb = self.embed(src) * math.sqrt(self.d_model)
        tgt_emb = self.embed(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_enc(src_emb)
        tgt_emb = self.pos_enc(tgt_emb)

        enc_attns = []  # collect encoder self-attention per layer
        x = src_emb
        for layer in self.encoder_layers:
            x, attn = layer(x, mask=src_mask)
            enc_attns.append(attn.detach().cpu())

        enc_out = x

        dec_attns_masked = []  # decoder masked self-attn per layer
        dec_attns_encdec = []  # encoder-decoder attn per layer
        y = tgt_emb
        for layer in self.decoder_layers:
            y, masked_attn, encdec_attn = layer(y, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
            dec_attns_masked.append(masked_attn.detach().cpu())
            dec_attns_encdec.append(encdec_attn.detach().cpu())

        logits = self.out(y)
        return logits, {
            "encoder_self": enc_attns,
            "decoder_masked_self": dec_attns_masked,
            "decoder_encdec": dec_attns_encdec
        }
