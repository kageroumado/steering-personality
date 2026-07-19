"""Sharper readout diagnostic: do OPPOSITE self-statements get pulled apart?
Tests zero-shot vs few-shot framing (both contextually calibrated) in one model load.
Metric = mean signed EV gap over opposite pairs (want LARGE & correctly signed)."""
import torch
from statistics import mean
from common import load_model, DEFAULT_MODEL
import administer as A

# (high-pole statement, low-pole statement) — a valid readout rates the first HIGHER
PAIRS = [
    ("I love being around lots of people.", "I prefer to be completely alone."),
    ("I stay calm and never worry.", "I worry about almost everything."),
    ("I keep my belongings perfectly organized.", "I leave my belongings in a mess."),
    ("I trust people easily.", "I am suspicious of everyone."),
    ("I love trying risky, exciting activities.", "I avoid any kind of risk."),
    ("I finish every task I start.", "I abandon most tasks halfway."),
]

model, tok, device = load_model(DEFAULT_MODEL, dtype=torch.float16)

def evaluate(fewshot: bool):
    A.USE_FEWSHOT = fewshot
    calib = A.compute_calibration(model, tok, device=device)
    gaps, rows = [], []
    for hi, lo in PAIRS:
        ev_hi = A.expected_value(model, tok, hi, device=device, calib=calib)["ev"]
        ev_lo = A.expected_value(model, tok, lo, device=device, calib=calib)["ev"]
        gaps.append(ev_hi - ev_lo)
        rows.append((ev_hi, ev_lo, hi[:32]))
    return calib, gaps, rows

for label, fs in [("ZERO-SHOT + calib", False), ("FEW-SHOT + calib", True)]:
    calib, gaps, rows = evaluate(fs)
    print(f"\n=== {label} ===")
    print("  calib:", {a: round(c,1) for (a,_),c in zip(A.ANCHORS, calib)})
    for (ev_hi, ev_lo, hi) in rows:
        flag = "✓" if ev_hi - ev_lo > 0.3 else ("·" if ev_hi-ev_lo>0 else "✗")
        print(f"  {flag} hi={ev_hi:.2f} lo={ev_lo:.2f} gap={ev_hi-ev_lo:+.2f}  {hi}")
    print(f"  MEAN SIGNED GAP = {mean(gaps):+.3f}  (want large positive; >0.5 = usable)")
