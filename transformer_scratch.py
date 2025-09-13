# transformer_scratch.py
import math, json, pickle, argparse, os, random
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# Data / Dataset
# -------------------------
class JsonlEventDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab: Dict[str,int], seq_len: int = 256):
        self.seq_len = seq_len
        self.samples: List[List[int]] = []
        self.vocab = vocab
        with open(jsonl_path) as f:
            for line in f:
                events = json.loads(line)["events"]
                ids = [vocab[t] for t in events if t in vocab]
                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        # take a random window up to seq_len
        if len(ids) > self.seq_len:
            start = random.randint(0, len(ids) - self.seq_len)
            ids = ids[start:start+self.seq_len]
        x = torch.tensor(ids[:-1], dtype=torch.long)  # input
        y = torch.tensor(ids[1:], dtype=torch.long)   # next token
        return x, y

# -------------------------
# Positional encoding (sinusoidal)
# -------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

# -------------------------
# Multi-Head Self-Attention (causal)
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.size()
        q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, T, dh]
        k = self.Wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, h, T, T]

        # causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        att = att.masked_fill(mask.bool().unsqueeze(0).unsqueeze(0), float("-inf"))

        w = torch.softmax(att, dim=-1)
        w = self.drop(w)
        out = w @ v                                 # [B, h, T, dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # concat heads
        return self.Wo(out)

# -------------------------
# Transformer Block
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# -------------------------
# Language Model head
# -------------------------
class TinyMusicTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4, d_ff: int = 256, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        # idx: [B, T]
        x = self.tok_emb(idx)           # [B, T, D]
        x = self.pos_enc(x)             # add positions
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)             # [B, T, V]

# -------------------------
# Train / Sample
# -------------------------
def train(jsonl, vocab_pkl, out_dir="ckpts", seq_len=256, batch=8, epochs=5, lr=3e-4, d_model=128, n_layers=4, n_heads=4, d_ff=256, dropout=0.1):
    os.makedirs(out_dir, exist_ok=True)
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)["token_to_id"]

    ds = JsonlEventDataset(jsonl, vocab, seq_len=seq_len)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyMusicTransformer(vocab_size=len(vocab), d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for xb, yb in tqdm(dl, desc=f"Epoch {ep}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)                   # [B, T, V]
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep}: loss={np.mean(losses):.4f}")
        torch.save(model.state_dict(), os.path.join(out_dir, f"tiny_ep{ep}.pt"))

def sample(jsonl, vocab_pkl, ckpt, max_new_tokens=128, temperature=1.0, top_k=0):
    with open(vocab_pkl, "rb") as f:
        obj = pickle.load(f)
        token_to_id = obj["token_to_id"]; id_to_token = obj["id_to_token"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyMusicTransformer(vocab_size=len(token_to_id)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # seed from first example
    first_events = json.loads(open(jsonl).readline())["events"]
    ctx = [token_to_id[t] for t in first_events[:64] if t in token_to_id]  # small prefix
    x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)    # [1, T]

    @torch.no_grad()
    def decode_step(x):
        logits = model(x)[:, -1, :] / max(1e-8, temperature)
        if top_k > 0:
            v, ix = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, ix, v)
            logits = mask
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        return idx

    for _ in range(max_new_tokens):
        idx = decode_step(x)
        x = torch.cat([x, idx], dim=1)

    out_ids = x[0].tolist()
    out_tokens = [id_to_token[i] for i in out_ids]
    print("\n--- SAMPLE TOKENS ---")
    print(" ".join(out_tokens[-200:]))  # print tail

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/tiny.jsonl")
    ap.add_argument("--vocab", default="data/tiny.vocab.pkl")
    ap.add_argument("--mode", choices=["train","sample"], required=True)
    ap.add_argument("--out_dir", default="ckpts")
    # train hparams
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    # sampling
    ap.add_argument("--ckpt", type=str, default="ckpts/tiny_ep5.pt")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "train":
        train(args.jsonl, args.vocab, args.out_dir, args.seq_len, args.batch, args.epochs,
              args.lr, args.d_model, args.n_layers, args.n_heads, args.d_ff, args.dropout)
    else:
        sample(args.jsonl, args.vocab, args.ckpt, args.max_new_tokens, args.temperature, args.top_k)
