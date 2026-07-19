"""Verify the fast (KV-cache) administration matches the slow path UNDER STEERING.
The gate numbers came from the slow path; the loop will use fast — confirm they agree."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from administer import administer_spi, score_scales, load_spi, keyed_items_for, compute_calibration
from extract_vectors import VEC_DIR
import os

BAND = list(range(-1, -16, -1))   # SelfControl @ late
spi, items = load_spi()
device = pick_device()
tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
cmodel = ControlModel(base, BAND)
vec = ControlVector.import_gguf(os.path.join(VEC_DIR, "SelfControl.gguf"))
subset = keyed_items_for(spi["keys"], ["SelfControl"])
calib = compute_calibration(base, tok, device=device)   # unsteered, slow

for k in (-2.0, 0.0, 2.0):
    cmodel.reset()
    if k: cmodel.set_control(vec, k)
    fast = score_scales(administer_spi(cmodel, tok, spi, items, device=device, log_every=0,
                                       calib=calib, subset=subset, fast=True), spi["keys"])["SelfControl"]
    slow = score_scales(administer_spi(cmodel, tok, spi, items, device=device, log_every=0,
                                       calib=calib, subset=subset, fast=False), spi["keys"])["SelfControl"]
    print(f"  k={k:+.0f}  fast={fast:.3f}  slow={slow:.3f}  diff={abs(fast-slow):.4f}")
cmodel.reset()
print("OK if all diffs < 0.02 (matches slow-path gate numbers: -2=1.84, 0=2.64, +2=3.20)")
