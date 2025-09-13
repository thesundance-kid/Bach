# simple_midi_tokenizer.py
# Minimal, readable event-based tokenizer (no music21, no project structure).
from __future__ import annotations
import argparse
import json
from typing import List, Tuple
import numpy as np
import pretty_midi as pm

# --- simple, opinionated defaults ---
# We’ll quantize time as 16th notes based on the *first* tempo we see.
# We’ll bucket velocity into 32 bins (0..31).
# We’ll assume 4/4 only for bar printing (not required for tokens).

def qpm_from_midi(m: pm.PrettyMIDI) -> float:
    tempi = m.get_tempo_changes()[1]
    return float(tempi[0]) if len(tempi) else 120.0

def time_step_from_qpm(qpm: float) -> float:
    # 16th-note = beat/4 ; beat(sec) = 60/qpm
    return (60.0 / qpm) / 4.0

def vel_bucket(v: int) -> int:
    # MIDI 1..127 -> 0..31
    return int(np.clip(np.interp(v, [1, 127], [0, 31]), 0, 31))

def midi_to_events(midi_path: str, section_tag: str = "A") -> List[str]:
    m = pm.PrettyMIDI(midi_path)
    qpm = qpm_from_midi(m)
    dt_unit = time_step_from_qpm(qpm)

    # Collect note on/off + velocity events with absolute times
    evs: List[Tuple[float, str]] = []
    for inst in m.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            evs.append((n.start, f"NOTE_ON={n.pitch}"))
            evs.append((n.end,   f"NOTE_OFF={n.pitch}"))
            evs.append((n.start, f"VEL={vel_bucket(n.velocity)}"))

    evs.sort(key=lambda x: x[0])

    tokens: List[str] = []
    # minimal header; keep it tiny
    tokens += ["[BOS]", "[SECTION]", f"SEC={section_tag}"]

    cur_t = 0.0
    for t, ev in evs:
        # Insert a time shift if needed (number of 16th-note steps)
        dt = max(0.0, t - cur_t)
        if dt > 0:
            steps = int(round(dt / dt_unit))
            # If large gap, emit multiple 99-step chunks
            while steps > 99:
                tokens.append("TSHIFT=99")
                steps -= 99
                cur_t += 99 * dt_unit
            if steps > 0:
                tokens.append(f"TSHIFT={steps}")
                cur_t += steps * dt_unit
        # Then the event
        tokens.append(ev)

    tokens.append("[EOS]")
    return tokens

def make_test_midi(out_path: str):
    """Creates a tiny 8-bar arpeggio at ~120bpm to test tokenization."""
    m = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    t = 0.0
    q = 0.5  # quarter at 120bpm
    pattern = [60, 64, 67, 72]  # C major arpeggio
    for _ in range(8):          # 8 bars
        for p in pattern:
            inst.notes.append(pm.Note(velocity=96, pitch=p, start=t, end=t+q))
            t += q
    m.instruments.append(inst)
    m.write(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", type=str, help="Path to a MIDI file to tokenize")
    ap.add_argument("--make_test_midi", type=str, help="Write a synthetic test MIDI here and exit")
    ap.add_argument("--print_json", action="store_true", help="Print JSON instead of plain lines")
    args = ap.parse_args()

    if args.make_test_midi:
        make_test_midi(args.make_test_midi)
        print(f"Wrote test MIDI to {args.make_test_midi}")
        return

    if not args.midi:
        ap.error("Provide --midi path or use --make_test_midi to generate one.")

    tokens = midi_to_events(args.midi)
    if args.print_json:
        print(json.dumps({"midi_path": args.midi, "events": tokens}, ensure_ascii=False))
    else:
        for tok in tokens:
            print(tok)

if __name__ == "__main__":
    main()
