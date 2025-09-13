# build_dataset.py
import argparse, json, os, pickle
from typing import List
from simple_midi_tokenizer import midi_to_events  # uses your existing file

def build_dataset(midi_paths: List[str], out_jsonl: str, min_len: int = 64):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    vocab = {}
    inv = []
    n_written = 0
    with open(out_jsonl, "w") as f:
        for mp in midi_paths:
            events = midi_to_events(mp)  # list[str]
            if len(events) < min_len: 
                continue
            # write one sequence
            f.write(json.dumps({"events": events}, ensure_ascii=False) + "\n")
            n_written += 1
            # update vocab
            for t in events:
                if t not in vocab:
                    vocab[t] = len(vocab)
                    inv.append(t)
    # persist vocab
    vpath = out_jsonl.replace(".jsonl", ".vocab.pkl")
    with open(vpath, "wb") as pf:
        pickle.dump({"token_to_id": vocab, "id_to_token": inv}, pf)
    print(f"Wrote {n_written} sequences to {out_jsonl}")
    print(f"Wrote vocab of size {len(vocab)} to {vpath}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tiny.jsonl")
    ap.add_argument("--min_len", type=int, default=64)
    ap.add_argument("midis", nargs="+", help="One or more MIDI paths")
    args = ap.parse_args()
    build_dataset(args.midis, args.out, args.min_len)
