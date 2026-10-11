"""
orthogonalize.py — is the M4 cross-trait leakage removable, or intrinsic to the geometry?

iter-001 finding: pca_diff steering vectors are not clean single-facet axes — steering one facet
drags others by |r| up to 1.0. Two hypotheses:
  (H-removable) leakage is a shared-component artifact — strip each facet direction of its
                projection onto the others and steering still works but stops co-moving them.
  (H-intrinsic) the trait *is* its entanglement — orthogonalizing destroys the steering effect.

DECISIVE TEST (closed, within the 5-facet set so the orthogonalization basis == the leakage set):
  * orthogonalize each facet's all-layer direction against the span of the other 4 (per layer),
    rescaled to the original (unit) norm so magnitude is held fixed — isolates DIRECTION effect.
  * measure the 5×5 leakage matrix L[steer][read] = Pearson(θ_steer, score_read) over a dose
    sweep, for RAW and ORTHO vectors. Diagonal ≈ M1 (must survive); off-diagonal = leakage (want ↓).

Also reports the raw entanglement geometry (pairwise cosine, residual fraction surviving
orthogonalization) so we can see *how* coupled the directions are.
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from administer import (administer_spi, score_scales, load_spi, keyed_items_for,
                        _make_scorer, calibration_cached)
from extract_vectors import VEC_DIR

from paths import OUT
FACETS = ["SelfControl", "Trust", "Sociability", "Conformity", "Adaptability"]
BAND_OF = {"SelfControl": "late", "Trust": "mid", "Sociability": "mid",
           "Conformity": "late", "Adaptability": "late"}
BANDS = {"late": list(range(-1, -16, -1)), "mid": list(range(-12, -21, -1))}
# usable θ grids (respect one-sided ranges; 4–5 points is enough for a Pearson over doses)
DOSE = {
    "SelfControl":  [-2.0, -1.0, 0.0, 1.0, 2.0],
    "Trust":        [-2.0, -1.0, 0.0, 1.0, 2.0],
    "Sociability":  [-2.0, -1.0, 0.0, 1.0, 2.0],
    "Conformity":   [-2.0, -1.0, 0.0, 1.0],
    "Adaptability": [-2.0, -1.0, 0.0, 1.0],
}


def orthogonalize(vecs):
    """Per-layer: project each facet direction onto the orthogonal complement of the span of the
    other 4, then rescale to the original norm. Returns ({facet: ControlVector}, diagnostics)."""
    layers = sorted(set.intersection(*[set(v.directions) for v in vecs.values()]))
    new = {f: {} for f in FACETS}
    resid = defaultdict(list)        # surviving fraction per facet (per layer)
    cos = defaultdict(list)          # pairwise cosine per (f,g)
    for l in layers:
        D = {f: np.asarray(vecs[f].directions[l], dtype=np.float64) for f in FACETS}
        for i, f in enumerate(FACETS):
            for g in FACETS:
                if g != f:
                    a, b = D[f], D[g]
                    cos[(f, g)].append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            others = np.stack([D[g] for g in FACETS if g != f], axis=1)  # H×4
            di = D[f]
            coef, *_ = np.linalg.lstsq(others, di, rcond=None)
            orth = di - others @ coef
            n, n0 = np.linalg.norm(orth), np.linalg.norm(di)
            resid[f].append(float(n / n0))
            orth = orth * (n0 / n) if n > 1e-8 else di
            new[f][l] = orth.astype(np.float32)
    diag = {"residual_frac": {f: round(float(np.mean(resid[f])), 3) for f in FACETS},
            "cosine": {f: {g: round(float(np.mean(cos[(f, g)])), 3)
                           for g in FACETS if g != f} for f in FACETS}}
    ovecs = {f: ControlVector(model_type=vecs[f].model_type, directions=new[f]) for f in FACETS}
    return ovecs, diag


def leakage_matrix(base, tok, spi, items, device, calib, subset, vecs, tag):
    """5×5 L[steer][read] = Pearson(θ_steer, score_read) over the dose sweep. Band-grouped:
    wrap each band ONCE (memory rule). Returns nested dict."""
    by_band = defaultdict(list)
    for f in FACETS:
        by_band[BAND_OF[f]].append(f)
    series = {f: {"theta": [], "scores": defaultdict(list)} for f in FACETS}
    t0 = time.time()
    for band, facets in by_band.items():
        cmodel = ControlModel(base, BANDS[band])
        for f in facets:
            for k in DOSE[f]:
                cmodel.reset()
                if k != 0:
                    cmodel.set_control(vecs[f], float(k))
                ev = administer_spi(cmodel, tok, spi, items, device=device, log_every=0,
                                    calib=calib, subset=subset, fast=True)
                sc = score_scales(ev, spi["keys"])
                series[f]["theta"].append(k)
                for g in FACETS:
                    series[f]["scores"][g].append(sc.get(g))
        cmodel.reset(); cmodel.unwrap()
        del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    print(f"  [{tag}] swept in {time.time()-t0:.0f}s", flush=True)
    L = {}
    for f in FACETS:
        th = series[f]["theta"]
        L[f] = {g: round(float(pearsonr(th, series[f]["scores"][g])[0]), 3) for g in FACETS}
    return L


def summarize(L):
    diag = np.mean([L[f][f] for f in FACETS])
    off = np.mean([abs(L[f][g]) for f in FACETS for g in FACETS if g != f])
    return float(diag), float(off)


def main():
    spi, items = load_spi()
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    calib = calibration_cached(_make_scorer(base, tok, device))
    subset = keyed_items_for(spi["keys"], FACETS)

    raw = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in FACETS}
    ortho, diag = orthogonalize(raw)

    print("\n=== raw entanglement geometry ===")
    print("residual fraction surviving orthogonalization (low = facet lives in others' span):")
    for f in FACETS:
        print(f"  {f:12} {diag['residual_frac'][f]:.3f}")

    L_raw = leakage_matrix(base, tok, spi, items, device, calib, subset, raw, "RAW")
    L_ortho = leakage_matrix(base, tok, spi, items, device, calib, subset, ortho, "ORTHO")

    def show(L, name):
        print(f"\n=== {name} leakage matrix  L[steer][read] = Pearson(θ, score) ===")
        print("steer\\read   " + "".join(f"{g[:5]:>7}" for g in FACETS))
        for f in FACETS:
            print(f"  {f:11}" + "".join(f"{L[f][g]:>7.2f}" for g in FACETS))
    show(L_raw, "RAW")
    show(L_ortho, "ORTHO")

    draw, oraw = summarize(L_raw)
    dor, oor = summarize(L_ortho)
    print(f"\n=== verdict ===")
    print(f"  RAW   : mean diagonal (steering) = {draw:+.2f} | mean |off-diag| (leakage) = {oraw:.2f}")
    print(f"  ORTHO : mean diagonal (steering) = {dor:+.2f} | mean |off-diag| (leakage) = {oor:.2f}")
    removable = dor >= 0.8 * draw and oor < oraw - 0.1
    print(f"  -> leakage is {'REMOVABLE (steering survived, leakage dropped)' if removable else 'NOT cleanly removable (see numbers)'}")

    out = {"residual_frac": diag["residual_frac"], "cosine": diag["cosine"],
           "L_raw": L_raw, "L_ortho": L_ortho,
           "summary": {"raw": {"diag": draw, "offdiag": oraw},
                       "ortho": {"diag": dor, "offdiag": oor}},
           "dose": DOSE, "removable": bool(removable)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "orthogonalize.json"), "w"), indent=2)
    print(f"\nsaved → {os.path.join(OUT, 'orthogonalize.json')}")


if __name__ == "__main__":
    main()
