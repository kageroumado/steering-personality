"""Find the usable steering-coefficient scale for repeng unit-norm vectors.
Sweeps a fine grid for one facet, administers its own keyed items (fast), and prints the
target facet score + a coherence proxy (whether scores stay sane vs collapse)."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
import os
from common import DEFAULT_MODEL, pick_device
from administer import administer_spi, score_scales, load_spi, keyed_items_for, compute_calibration
from extract_vectors import LAYER_IDS, VEC_DIR

facet = sys.argv[1] if len(sys.argv) > 1 else "SelfControl"
GRID = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]

spi, items = load_spi()
subset = keyed_items_for(spi["keys"], [facet])
device = pick_device()
tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
cmodel = ControlModel(base, LAYER_IDS)
vec = ControlVector.import_gguf(os.path.join(VEC_DIR, f"{facet}.gguf"))
calib = compute_calibration(cmodel, tok, device=device)

print(f"facet={facet}  items={len(subset)}  layers={LAYER_IDS}")
print(f"{'k':>6} | {facet+' score':>14} | per-item EVs")
prev = None
for k in GRID:
    cmodel.reset()
    if k != 0: cmodel.set_control(vec, float(k))
    ev = administer_spi(cmodel, tok, spi, items, device=device, log_every=0,
                        calib=calib, subset=subset)
    score = score_scales(ev, spi["keys"]).get(facet)
    evs = sorted(round(v["ev"], 2) for v in ev.values())
    print(f"{k:>6} | {score:>14.3f} | {evs}", flush=True)
cmodel.reset()
