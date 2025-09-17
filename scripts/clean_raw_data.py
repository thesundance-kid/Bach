import os, sys, csv, hashlib, warnings, argparse, shutil
from pathlib import Path
import numpy as np
import pretty_midi

# Quiet known pretty_midi warnings about meta events on non-zero tracks
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi.pretty_midi")

def midi_iter(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                yield Path(dirpath) / f

def safe_load(path):
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
        return pm
    except Exception:
        return None

def median_bpm(pm):
    tempos = pm.get_tempo_changes()[1]
    if tempos.size == 0:
        return None
    return float(np.median(tempos))

def first_timesig(pm):
    ts = pm.time_signature_changes
    if not ts:
        return None
    return f"{ts[0].numerator}/{ts[0].denominator}"

def content_hash(pm, round_time=0.01):
    """
    Lightweight de-dup hash:
    - round note start/end times to reduce inconsequential jitter
    - include pitch + program/channel info
    """
    items = []
    for inst in pm.instruments:
        prog = inst.program
        is_drum = inst.is_drum
        for n in inst.notes:
            items.append((
                prog, int(is_drum),
                n.pitch,
                round(n.start, 2 if round_time is None else 2),
                round(n.end, 2 if round_time is None else 2),
                int(n.velocity)
            ))
    items.sort()
    m = hashlib.sha1()
    for it in items[:5000]:  # cap to keep hashing fast on monsters
        m.update(repr(it).encode())
    return m.hexdigest()

def copy_to(src, dst_root):
    dst = dst_root / src.name
    # Avoid collisions by prefixing parent folder if necessary
    if dst.exists():
        dst = dst_root / f"{src.parent.name}__{src.name}"
    shutil.copy2(src, dst)
    return dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="training_data/raw", help="Path to raw MIDI root")
    ap.add_argument("--out_dir", default="training_data/cleaned", help="Where to copy cleaned MIDIs")
    ap.add_argument("--bpm_min", type=float, default=60.0)
    ap.add_argument("--bpm_max", type=float, default=180.0)
    ap.add_argument("--timesig", default="4/4", help="Keep only this time signature (e.g., 4/4)")
    ap.add_argument("--dedupe", action="store_true", help="Enable simple content-based de-duplication")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "cleaned_manifest.csv"

    seen_hashes = set()
    kept, skipped_bad, skipped_bpm, skipped_ts, skipped_dup = 0, 0, 0, 0, 0

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["rel_path", "bpm_median", "timesig", "hash"])

        for idx, path in enumerate(midi_iter(raw_dir), 1):
            pm = safe_load(path)
            if pm is None:
                skipped_bad += 1
                continue

            bpm = median_bpm(pm)
            if bpm is None or (bpm < args.bpm_min or bpm > args.bpm_max):
                skipped_bpm += 1
                continue

            ts = first_timesig(pm)
            if args.timesig and ts != args.timesig:
                skipped_ts += 1
                continue

            h = content_hash(pm) if args.dedupe else ""
            if args.dedupe and h in seen_hashes:
                skipped_dup += 1
                continue
            if args.dedupe:
                seen_hashes.add(h)

            dst = copy_to(path, out_dir)
            rel = dst.relative_to(out_dir)
            writer.writerow([str(rel), f"{bpm:.3f}", ts or "Unknown", h])
            kept += 1

    print(f"Done.\n Kept: {kept}"
          f"\n Skipped bad: {skipped_bad}"
          f"\n Skipped BPM out of [{args.bpm_min},{args.bpm_max}]: {skipped_bpm}"
          f"\n Skipped time sig != {args.timesig}: {skipped_ts}"
          f"\n Skipped duplicates: {skipped_dup}"
          f"\n Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
