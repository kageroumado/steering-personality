"""Quick diagnostic: does contextual calibration break the anchor-prior degeneracy?
Scores a diverse subset of SPI items raw vs calibrated; prints argmax spread + EV range."""
import torch
from collections import Counter
from statistics import mean, pstdev
from common import load_model, DEFAULT_MODEL
from administer import load_spi, expected_value, compute_calibration, ANCHORS

spi, items = load_spi()
ids = list(items)[::9]          # ~15 spread-out items
model, tok, device = load_model(DEFAULT_MODEL, dtype=torch.float16)
calib = compute_calibration(model, tok, device=device)
print("calibration (content-free anchor logprobs):",
      {a: round(c, 2) for (a, _), c in zip(ANCHORS, calib)})

for label, cal in [("RAW", None), ("CALIBRATED", calib)]:
    rows = [(items[i]["text"][:42], expected_value(model, tok, items[i]["text"],
             device=device, calib=cal)) for i in ids]
    evs = [r[1]["ev"] for r in rows]
    args = [r[1]["argmax"] for r in rows]
    print(f"\n=== {label} ===  EV mean={mean(evs):.2f} sd={pstdev(evs):.2f} "
          f"range=[{min(evs):.2f},{max(evs):.2f}]  argmax={dict(sorted(Counter(args).items()))}")
    for txt, r in rows[:8]:
        print(f"  {r['ev']:.2f}  {txt}")
