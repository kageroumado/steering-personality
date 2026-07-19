"""
build_steering.py — definitive steering build + Tier-0 gate, ONE model load.

Consolidates the piecemeal scans. Key facts established (LAB-NOTES):
  * repeng directions are ALL-LAYER → train once on the base model, apply at any band.
  * injection band is facet-specific (SelfControl→late, Trust→mid).
  * pca_diff sign is arbitrary → align behaviorally (negate if +k lowers the readout).

For each facet: train once, evaluate {late, mid} bands with auto sign-alignment, pick the
(band, sign) with the cleanest dose-response, save the sign-aligned vector + record the band
in provenance, and tally the Tier-0 gate (ρ≥0.8 ∧ Δ≥0.3 on the canonical {-2..2} sweep).
"""
from __future__ import annotations
import os, json, time
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from administer import (administer_spi, score_scales, load_spi,
                        keyed_items_for, compute_calibration)
from extract_vectors import build_dataset, FACETS, VEC_DIR

OUT_DIR = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
BANDS = {"late": list(range(-1, -16, -1)), "mid": list(range(-12, -21, -1))}
GATE = [-2.0, -1.0, 0.0, 1.0, 2.0]
ALL = ["SelfControl", "Trust", "Conformity", "Sociability", "Adaptability"]


def main():
    spi, items = load_spi()
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()

    # calibration on the UNSTEERED base (no wrap), reused everywhere
    calib = compute_calibration(base, tok, device=device)

    # train all vectors once on the base (band-independent, all-layer directions)
    vecs = {}
    for facet in ALL:
        ds = build_dataset(tok, FACETS[facet]["pos"], FACETS[facet]["neg"])
        t0 = time.time()
        vecs[facet] = ControlVector.train(base, tok, ds, method="pca_diff")
        print(f"[train {facet}] {time.time()-t0:.0f}s ({len(ds)} pairs)", flush=True)

    results = {}
    for facet in ALL:
        subset = keyed_items_for(spi["keys"], [facet])
        best = None
        for band_name, band in BANDS.items():
            cmodel = ControlModel(base, band)
            curve = []
            for k in GATE:
                cmodel.reset()
                if k != 0:
                    cmodel.set_control(vecs[facet], float(k))
                ev = administer_spi(cmodel, tok, spi, items, device=device, log_every=0,
                                    calib=calib, subset=subset)
                curve.append(score_scales(ev, spi["keys"]).get(facet))
            cmodel.unwrap()
            # sign alignment: pick orientation with higher (positive) Spearman
            rho_pos, _ = spearmanr(GATE, curve)
            rho = abs(rho_pos)
            sign = 1 if rho_pos >= 0 else -1
            delta = max(curve) - min(curve)
            cand = {"band": band_name, "sign": sign, "rho": float(rho),
                    "delta": float(delta), "curve": [round(c, 3) for c in curve]}
            print(f"  [{facet:12}] {band_name:4} ρ±={rho:+.2f} sign={sign:+d} "
                  f"Δ={delta:.2f} curve={cand['curve']}", flush=True)
            # prefer clean+strong: rho gated by delta>=0.3, tie-break on delta
            score = (rho if delta >= 0.3 else rho * 0.5) + 0.1 * min(delta, 1.5)
            cand["score"] = score
            if best is None or score > best["score"]:
                best = cand
        # save sign-aligned vector at chosen band
        final = vecs[facet] if best["sign"] == 1 else vecs[facet] * -1.0
        gguf = os.path.join(VEC_DIR, f"{facet}.gguf")
        final.export_gguf(gguf)
        best["gguf"] = gguf
        best["pass"] = bool(best["rho"] >= 0.8 and best["delta"] >= 0.3)
        results[facet] = best
        print(f"  >> {facet}: band={best['band']} sign={best['sign']:+d} "
              f"ρ={best['rho']:.2f} Δ={best['delta']:.2f} "
              f"{'✅PASS' if best['pass'] else '❌'}", flush=True)

    n_pass = sum(r["pass"] for r in results.values())
    out = {"model": DEFAULT_MODEL, "bands": BANDS, "gate_sweep": GATE,
           "n_pass": n_pass, "gate_pass": n_pass >= 3, "facets": results}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUT_DIR, "steering-config.json"), "w"), indent=2)
    print(f"\n=== GATE: {n_pass}/5 facets pass (ρ≥0.8 ∧ Δ≥0.3) — "
          f"{'PASS' if n_pass>=3 else 'FAIL'} ===")
    for f, r in results.items():
        print(f"  {'✅' if r['pass'] else '❌'} {f:12} band={r['band']:4} "
              f"sign={r['sign']:+d} ρ={r['rho']:.2f} Δ={r['delta']:.2f}")
    print(f"\nsaved → {os.path.join(OUT_DIR, 'steering-config.json')}")


if __name__ == "__main__":
    main()
