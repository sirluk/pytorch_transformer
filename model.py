import yaml
import argparse
import torch
import math
from torch import nn
from torch.utils.data import DataLoader
from data_loader import TokenizedDataset
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, context_length, model_dim):
        super().__init__()

        pos = torch.arange(context_length).unsqueeze(1)
        dim = torch.arange(0, model_dim, step=2).unsqueeze(0)
        val = pos * torch.exp(dim * -math.log(10000) / model_dim)

        pos_enc = torch.empty((context_length, model_dim))
        pos_enc[:,0::2] = torch.sin(val)
        pos_enc[:,1::2] = torch.cos(val)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        return x + self.pos_enc



class Embedding(nn.Module):

    def __init__(self, vocab_size, context_length, model_dim, dropout = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.model_dim = model_dim
        self.sqrt_model_dim = math.sqrt(model_dim)

        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_enc = PositionalEncoding(context_length, model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.token_emb(x)
        x = x * self.sqrt_model_dim
        x = self.pos_enc(x)
        return self.dropout(x)


class CausalSelfAttention(nn.Module):

    def __init__(self, context_length, model_dim, n_heads, dropout = 0.1, bias = False):
        super().__init__()

        assert model_dim % n_heads == 0, 'model_dim needs to be a multiple of n_heads'

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.context_length = context_length
        self.hidden_dim = model_dim // n_heads

        self.qkv = nn.Linear(self.model_dim, 3 * self.model_dim, bias=bias)
        self.final_proj = nn.Linear(self.model_dim, self.model_dim, bias=bias)

        causal_mask = torch.tril(torch.ones((context_length, context_length), dtype=bool)) \
            .view(1, 1, context_length, context_length)
        self.register_buffer('causal_mask', causal_mask)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, D = x.shape

        q, k, v = self.qkv(x).split(self.model_dim, dim=-1)

        q = q.reshape(B, C, self.n_heads, self.hidden_dim).transpose(1, 2)
        k = k.reshape(B, C, self.n_heads, self.hidden_dim).transpose(1, 2)
        v = v.reshape(B, C, self.n_heads, self.hidden_dim).transpose(1, 2)

        attn = q @ k.transpose(-1, -2)
        attn_scaled = attn * (1.0 / math.sqrt(self.model_dim))
        attn_masked = attn_scaled.masked_fill(~self.causal_mask[:, :, :C,:C], -math.inf)
        attn_probs = F.softmax(attn_masked, dim=-1)
        attn_out = attn_probs @ v
        attn_cat = attn_out.transpose(1,2).contiguous().view(B, C, D)
        return self.dropout(self.final_proj(attn_cat))
    

class FFN(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout = 0.1, bias = False):
        super().__init__()

        self.model_dim = model_dim

        self.linear1 = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, model_dim, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return self.dropout(x)
    

class TransformerBlock(nn.Module):

    def __init__(self, context_length, model_dim, n_attn_heads, ffn_hidden_dim, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(context_length, model_dim, n_attn_heads, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, ffn_hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    

class Transformer(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.embedding = Embedding(cfg.vocab_size, cfg.context_length, cfg.model_dim, cfg.dropout)
        
        self.blocks = nn.Sequential(*[TransformerBlock(
            cfg.context_length,
            cfg.model_dim,
            cfg.n_attn_heads,
            cfg.ffn_hidden_dim,
            cfg.dropout
        ) for _ in range(cfg.n_blocks)])

        self.output_proj = nn.Linear(cfg.model_dim, cfg.vocab_size)

        self.output_proj.weight = self.embedding.token_emb.weight

    def forward(self, x):
        emb = self.embedding(x)
        out = self.blocks(emb)
        return self.output_proj(out)
        


if __name__ == '__main__':

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg['model_cfg'])

    setattr(cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)

    ds = TokenizedDataset(filenames='train.bin', context_length=cfg.context_length)

    dl = DataLoader(ds, batch_size=4)

    x, y = next(iter(dl))

    """     emb = Embedding(
        vocab_size = TokenizedDataset.TOKENIZER.n_words,
        context_length=cfg.context_length,
        model_dim=cfg.model_dim
    )
    x = emb(x) """

    model = Transformer(cfg)

    logits = model(x)

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    import IPython; IPython.embed(); exit(1)

