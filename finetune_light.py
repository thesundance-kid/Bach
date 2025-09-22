#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, random, math, os, sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import torch.nn.functional as F

# at the top of finetune_light.py
from music_transformer_parts.vocabulary import pad_token as PAD_ID, vocab_size as VOCAB_SIZE


# ---- import your local model/tokenizer parts ----
# Adjust these to your actual package/module names:
from music_transformer_parts.model import MusicTransformer  # or whatever your class is called
# If you have a separate load fn for checkpoints, import that too.

# ----------------------------
# Dataset: robust .pt loader
# ----------------------------
class EventDataset(Dataset):
    """
    Expects a .pt that contains either:
      - a 2D LongTensor of shape [num_sequences, seq_len], or
      - a Python list of 1D lists/LongTensors, or
      - a dict with a 'sequences' key to one of the above.
    """
    def __init__(self, pt_path):
        obj = torch.load(pt_path, map_location='cpu')
        if isinstance(obj, dict) and 'sequences' in obj:
            self.data = obj['sequences']
        else:
            self.data = obj
        if isinstance(self.data, torch.Tensor):
            # Expect [N, L]
            assert self.data.dim() == 2, f"Tensor must be 2D [N, L], got shape {tuple(self.data.shape)}"
            self.len = self.data.size(0)
            self.tensor_mode = True
        else:
            # List mode
            assert isinstance(self.data, (list, tuple)), "Unsupported .pt format; expected list/tuple or Tensor"
            self.len = len(self.data)
            self.tensor_mode = False

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.tensor_mode:
            seq = self.data[idx]
            # Ensure 1D long
            if seq.dim() == 1:
                return seq.long()
            elif seq.dim() == 2:
                # handle [L, 1] edge case
                return seq.view(-1).long()
            else:
                raise ValueError(f"Unexpected sequence ndim={seq.dim()}")
        else:
            seq = self.data[idx]
            return torch.as_tensor(seq, dtype=torch.long)

# ----------------------------
# Collate: crop + pad + masks
# ----------------------------
def make_collate_fn(max_len=256, pad_id=0, crop_mode="random"):
    """
    Crop each sequence to at most max_len, then right-pad to batch max.
    Returns (inp, targ, attn_mask)
    """
    assert crop_mode in ("random", "head")
    rng = random.Random(0)

    def collate(batch):
        cropped = []
        for seq in batch:
            seq = torch.as_tensor(seq, dtype=torch.long)
            if seq.numel() <= max_len:
                cropped.append(seq)
            else:
                if crop_mode == "random":
                    # keep at least 1 token for target
                    start = rng.randint(0, seq.numel() - max_len - 1)
                else:
                    start = 0
                cropped.append(seq[start:start+max_len])

        # Right-pad to batch max len
        L = max(s.numel() for s in cropped)
        x = torch.full((len(cropped), L), pad_id, dtype=torch.long)
        for i, s in enumerate(cropped):
            x[i, :s.numel()] = s

        # Next-token prediction
        inp  = x[:, :-1]
        targ = x[:, 1:]
        attn_mask = (inp != pad_id).long()
        return inp, targ, attn_mask
    return collate

# ----------------------------
# Freezing helpers
# ----------------------------
def freeze_lower_layers(model, unfreeze_top_k=2):
    """
    Freezes input_embedding and the lower decoder layers.
    Assumes model.decoder.layers is an indexable iterable (list/ModuleList).
    Adjust if your attribute names differ.
    """
    # Freeze input embedding
    if hasattr(model, "input_embedding"):
        for p in model.input_embedding.parameters():
            p.requires_grad = False

    # Find decoder layers
    layers = getattr(getattr(model, "decoder", None), "layers", None)
    if layers is None:
        raise AttributeError("Could not find decoder.layers; adjust freeze helper.")

    # Freeze all but top-K decoder layers
    n = len(layers)
    freeze_upto = max(0, n - max(1, unfreeze_top_k))
    for li, layer in enumerate(layers):
        train_this = (li >= freeze_upto)
        for p in layer.parameters():
            p.requires_grad = train_this

    # Keep final projection trainable
    if hasattr(model, "final"):
        for p in model.final.parameters():
            p.requires_grad = True

# ----------------------------
# Training step
# ----------------------------
def train_light(args):
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.force_cpu
                          else ("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"))
    print(f"[info] device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Data
    ds = EventDataset(args.train_pt)
    collate = make_collate_fn(max_len=args.max_len, pad_id=args.pad_id, crop_mode=args.crop_mode)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate
    )

    # Model
    # Load checkpoint dict (it has 'state_dict' and 'hparams')
    ck = torch.load(args.pretrained_ckpt, map_location='cpu')
    # after: ck = torch.load(args.pretrained_ckpt, map_location='cpu')
    hp = ck.get('hparams', {})
    model = MusicTransformer(
        d_model=hp.get('d_model', 512),
        num_layers=hp.get('num_layers', 6),
        num_heads=hp.get('num_heads', 8),
        d_ff=hp.get('d_ff', 2048),
        max_rel_dist=hp.get('max_rel_dist', 0),
        max_abs_position=hp.get('max_abs_position', 2048),
        vocab_size=hp.get('vocab_size', 4096),
        bias=hp.get('bias', True),
        dropout=hp.get('dropout', 0.1),
        layernorm_eps=hp.get('layernorm_eps', 1e-6),
    )

    raw_sd = ck.get('state_dict', None)
    if raw_sd is None:
        raw_sd = ck.get('model', ck)

    def remap_keys(sd):
        mapped = {}
        for k, v in sd.items():
            nk = k
            # 1) Multihead attention submodule rename
            nk = nk.replace(".mha.", ".self_attn.")
            # 2) Some repos name final layer "linear" or "final_layer"
            nk = nk.replace(".linear.", ".final.")
            nk = nk.replace(".final_layer.", ".final.")
            # 3) If the embedding name differs anywhere else, add here
            mapped[nk] = v
        return mapped

    sd = remap_keys(raw_sd)

    # Optional: report how many params will actually load (by name AND shape)
    msd = model.state_dict()
    match = sum(1 for k in sd if k in msd and msd[k].shape == sd[k].shape)
    total = len(msd)
    print(f"[info] loading remapped state_dict: matches {match}/{total}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:   print(f"[warn] missing keys: {missing[:12]}{' ...' if len(missing)>12 else ''}")
    if unexpected:print(f"[warn] unexpected keys: {unexpected[:12]}{' ...' if len(unexpected)>12 else ''}")




    model.to(device)

    # Freeze plan
    freeze_lower_layers(model, unfreeze_top_k=args.unfreeze_top_k)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # Train (light)
    model.train()
    step = 0
    losses = []
    t0 = time.time()

    for (inp, targ, attn_mask) in loader:
        batch_t0 = time.time()
        inp = inp.to(device, non_blocking=True)
        targ = targ.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        # Forward → logits
        # Adjust the forward signature if your model differs:
        pad_mask = (inp == args.pad_id).long()
        logits = model(inp, mask=pad_mask)

        # Loss (ignore padding)
        loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targ.reshape(-1),
        ignore_index=args.pad_id
        )
        loss.backward()
        optim.step()

        step += 1
        losses.append(loss.item())
        batch_dt = time.time() - batch_t0

        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
            print(f"step {step}/{args.step_limit} | loss {avg:.4f} | {batch_dt:.2f}s")

        if step >= args.step_limit:
            break

    total_dt = time.time() - t0
    print(f"[done] steps={step} | mean_loss={sum(losses)/len(losses):.4f} | total={total_dt:.1f}s (~{total_dt/max(1,step):.2f}s/step)")

    # Save small checkpoint
    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "config": vars(args),
        "steps": step,
        "mean_loss": sum(losses)/len(losses) if losses else math.inf
    }, args.save_path)
    print(f"[save] {args.save_path}")

# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Light-config local smoke test for Music Transformer fine-tune")
    # I/O
    p.add_argument("--train_pt", type=str, required=True, help="Path to tokenized .pt (e.g., training_data/processed/maestro_events_subset.pt)")
    p.add_argument("--pretrained_ckpt", type=str, required=True, help="Path to pretrained model checkpoint (e.g., model6v2.pt)")
    p.add_argument("--save_path", type=str, default="ckpts/smoke.pt")

    # Light config defaults
    p.add_argument("--config", type=str, default="light", choices=["light", "real"])
    p.add_argument("--step_limit", type=int, default=10)
    p.add_argument("--max_len", type=int, default=256)  # crop length
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pad_id", type=int, default=0)
    p.add_argument("--crop_mode", type=str, default="random", choices=["random", "head"])
    p.add_argument("--unfreeze_top_k", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_cpu", action="store_true")
    return p

def apply_presets(args):
    if args.config == "light":
        # Aggressively fast + small
        args.pad_id = getattr(args, "pad_id", PAD_ID)
        args.step_limit    = 10
        args.max_len       = 256
        args.batch_size    = 1
        args.num_workers   = 0
        args.unfreeze_top_k= 2    # train only top 2 layers + head
        args.lr            = 2e-5
        args.weight_decay  = 0.1
        args.log_every     = 1
    elif args.config == "real":
        # Sensible starting point for GPU runs (override on CLI as needed)
        args.step_limit    = 1000000  # effectively no cap
        args.max_len       = 1024
        args.batch_size    = 2
        args.num_workers   = 2
        args.unfreeze_top_k= 3
        args.lr            = 2e-5
        args.weight_decay  = 0.1
        args.log_every     = 50
    return args

def main():
    parser = build_parser()
    args = parser.parse_args()
    args = apply_presets(args)
    print("[cfg]", args)
    train_light(args)

if __name__ == "__main__":
    main()
