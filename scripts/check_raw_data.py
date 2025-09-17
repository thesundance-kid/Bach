import os
import pretty_midi
import numpy as np

RAW_DIR = "training_data/raw"

midi_files = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.lower().endswith(".mid") or f.lower().endswith(".midi"):
            midi_files.append(os.path.join(root, f))

print(f"Found {len(midi_files)} MIDI files")

# Quick checks
tempos = []
time_sigs = []
bad_files = []

for path in midi_files[:200]:  # just sample first 200 for speed
    try:
        pm = pretty_midi.PrettyMIDI(path)
        # Tempo
        tempo = np.median(pm.get_tempo_changes()[1]) if pm.get_tempo_changes()[1].size > 0 else None
        tempos.append(tempo)
        # Time signature
        tsigs = pm.time_signature_changes
        if tsigs:
            time_sigs.append(f"{tsigs[0].numerator}/{tsigs[0].denominator}")
        else:
            time_sigs.append("Unknown")
    except Exception as e:
        bad_files.append(path)

print(f"Checked {len(tempos)} files")
print(f"Median tempo: {np.nanmedian([t for t in tempos if t])}")
print("Tempo range:", np.nanmin(tempos), "-", np.nanmax(tempos))
print("Most common time signatures:", {ts: time_sigs.count(ts) for ts in set(time_sigs)})
print(f"Bad files: {len(bad_files)}")
