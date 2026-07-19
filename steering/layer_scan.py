"""Layer-band diagnostic (open decision §14.1), ONE band per invocation (robust: single
model load, no reloads — the reload-per-band version deadlocked MPS).

    python layer_scan.py mid          # band from BANDS below
Compares against late-band numbers already on record in LAB-NOTES.
"""
import sys, torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from administer import administer_spi, score_scales, load_spi, keyed_items_for, compute_calibration
from extract_vectors import build_dataset, FACETS

BANDS = {
    "late":      list(range(-1, -16, -1)),    # layers 17-31
    "mid":       list(range(-12, -21, -1)),   # layers 12-20
    "early-mid": list(range(-16, -25, -1)),   # layers 8-16
}
FACETS_TO_SCAN = ["SelfControl", "Trust", "Conformity"]
GRID = [-2, -1, -0.5, 0, 0.5, 1, 2]
GATE = [-2.0, -1.0, 0.0, 1.0, 2.0]

band_name = sys.argv[1] if len(sys.argv) > 1 else "mid"
layer_ids = BANDS[band_name]
spi, items = load_spi()
device = pick_device()
print(f"BAND {band_name}  layers={layer_ids}", flush=True)
tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
cmodel = ControlModel(base, layer_ids)
calib = compute_calibration(cmodel, tok, device=device)

for facet in FACETS_TO_SCAN:
    ds = build_dataset(tok, FACETS[facet]["pos"], FACETS[facet]["neg"])
    vec = ControlVector.train(cmodel, tok, ds, method="pca_diff")
    subset = keyed_items_for(spi["keys"], [facet])
    curve = {}
    for k in GRID:
        cmodel.reset()
        if k != 0:
            cmodel.set_control(vec, float(k))
        ev = administer_spi(cmodel, tok, spi, items, device=device, log_every=0,
                            calib=calib, subset=subset)
        curve[k] = score_scales(ev, spi["keys"]).get(facet)
    cmodel.reset()
    gate_y = [curve[k] for k in GATE]
    rho, _ = spearmanr(GATE, gate_y)
    usable = list(curve.values())
    d = max(usable) - min(usable)
    flag = "✅" if abs(rho) >= 0.8 and d >= 0.3 else "  "
    print(f"{flag} [{facet:12}] ρ(gate)={rho:+.2f} Δ(±2)={d:+.2f}  "
          f"curve={[round(curve[k],2) for k in GRID]}", flush=True)
