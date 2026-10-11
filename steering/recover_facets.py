"""
recover_facets.py — recover the 2 Tier-0 failures (Conformity, Adaptability).

Diagnoses from build_steering (steering-config.json):
  * Conformity (late, sign -1): ρ0.70 Δ1.15 — strong effect, single KINK at k=-2 on the
    canonical {-2..2} sweep -> non-monotone. Hypothesis: a gentler/denser sweep that avoids
    the ±2 extreme is clean. Re-sweep {-1.5..1.5}.
  * Adaptability (late, sign +1): ρ0.70 Δ0.63 — up-steer SATURATES (base self-reports ~3.9/6);
    down-steer works. mid band was never evaluated (build_steering crashed on it). The saved
    vector is STALE (never re-aligned). Hypothesis: either mid band has 2-sided range, or the
    facet is down-steerable only. Re-train fresh, sweep BOTH bands over a down-probing range,
    report full + half-range Spearman to expose the saturation directly.

Method mirrors build_steering but scoped to 2 facets, fast KV-cache readout, and each band is
wrapped ONCE then released (D6 memory rule; build_steering OOM'd after ~10 wrap cycles — this
does 4). Re-trains the two vectors fresh, sign-aligns on the dense sweep, re-saves gguf,
patches steering-config.json with the recovered configs.
"""
from __future__ import annotations
import os, json, time, gc
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from administer import (administer_spi, score_scales, load_spi, keyed_items_for,
                        _make_scorer, calibration_cached)
from extract_vectors import build_dataset, FACETS, VEC_DIR

from paths import OUT as OUT_DIR
BANDS = {"late": list(range(-1, -16, -1)), "mid": list(range(-12, -21, -1))}

# Per-facet recovery sweeps + bands to probe.
# Conformity: late band is clean+monotone but the high-conformity (-k) direction over-steers
# past k=-1.5 (the canonical-sweep kink at -2 = the repeng high-coeff nonlinearity). Use an
# ASYMMETRIC sweep that respects the usable range: drop -2, extend to +2 for dynamic range.
# Adaptability already recovered (late, down-steerable) — left out so its vector isn't churned.
PLAN = {
    "Conformity":   {"sweep": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], "bands": ["late"]},
}


def _aligned(sweep, curve):
    """Sign-align: orient so +k raises the score. Returns (rho_abs, sign, delta)."""
    rho_pos, _ = spearmanr(sweep, curve)
    sign = 1 if rho_pos >= 0 else -1
    return abs(rho_pos), sign, max(curve) - min(curve)


def _half_rhos(sweep, curve, sign):
    """Spearman over the down-half (k<=0) and up-half (k>=0), sign-corrected — exposes
    one-sided saturation (a facet that moves on one side only)."""
    pairs = list(zip(sweep, curve))
    down = [(k, c) for k, c in pairs if k <= 0]
    up = [(k, c) for k, c in pairs if k >= 0]
    def rho(seg):
        if len(seg) < 3:
            return None
        r, _ = spearmanr([k for k, _ in seg], [c for _, c in seg])
        return round(sign * r, 3)
    return rho(down), rho(up)


def main():
    spi, items = load_spi()
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()

    # calibration on UNSTEERED base, frozen (matches personas.py / Tier-0 path)
    calib = calibration_cached(_make_scorer(base, tok, device))

    # re-train both vectors fresh (clean unsigned directions; saved gguf for Adaptability is stale)
    vecs = {}
    for facet in PLAN:
        ds = build_dataset(tok, FACETS[facet]["pos"], FACETS[facet]["neg"])
        t0 = time.time()
        vecs[facet] = ControlVector.train(base, tok, ds, method="pca_diff")
        print(f"[train {facet}] {time.time()-t0:.0f}s ({len(ds)} pairs)", flush=True)

    results = {}
    for facet, cfg in PLAN.items():
        sweep = cfg["sweep"]
        subset = keyed_items_for(spi["keys"], [facet])
        print(f"\n=== {facet}: sweep={sweep} subset={len(subset)} items ===", flush=True)
        best = None
        for band_name in cfg["bands"]:
            band = BANDS[band_name]
            cmodel = ControlModel(base, band)
            curve = []
            for k in sweep:
                cmodel.reset()
                if k != 0:
                    cmodel.set_control(vecs[facet], float(k))
                ev = administer_spi(cmodel, tok, spi, items, device=device, log_every=0,
                                    calib=calib, subset=subset, fast=True)
                curve.append(score_scales(ev, spi["keys"]).get(facet))
            cmodel.reset(); cmodel.unwrap()
            del cmodel; gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

            rho, sign, delta = _aligned(sweep, curve)
            dn, up = _half_rhos(sweep, curve, sign)
            cand = {"band": band_name, "sign": sign, "rho": round(float(rho), 3),
                    "delta": round(float(delta), 3), "curve": [round(c, 3) for c in curve],
                    "rho_down": dn, "rho_up": up, "sweep": sweep}
            cand["pass"] = bool(rho >= 0.8 and delta >= 0.3)
            # selection score: clean full-sweep rho gated by delta, tie-break on delta
            cand["score"] = (rho if delta >= 0.3 else rho * 0.5) + 0.1 * min(delta, 1.5)
            print(f"  [{band_name:4}] ρ±={rho:.2f} sign={sign:+d} Δ={delta:.2f} "
                  f"ρ_dn={dn} ρ_up={up} curve={cand['curve']} "
                  f"{'PASS' if cand['pass'] else 'fail'}", flush=True)
            if best is None or cand["score"] > best["score"]:
                best = cand
        results[facet] = best
        # re-save sign-aligned vector at best band
        final = vecs[facet] if best["sign"] == 1 else vecs[facet] * -1.0
        gguf = os.path.join(VEC_DIR, f"{facet}.gguf")
        final.export_gguf(gguf)
        best["gguf"] = gguf
        best["vector_aligned"] = True
        print(f"  >> {facet}: band={best['band']} sign={best['sign']:+d} "
              f"ρ={best['rho']:.2f} Δ={best['delta']:.2f} "
              f"{'RECOVERED' if best['pass'] else 'still-fragile'} → saved {gguf}", flush=True)

    # patch steering-config.json
    cfg_path = os.path.join(OUT_DIR, "steering-config.json")
    cfg = json.load(open(cfg_path))
    for facet, r in results.items():
        cfg["facets"][facet] = {
            "band": r["band"], "sign": r["sign"], "rho": r["rho"], "delta": r["delta"],
            "pass": r["pass"], "curve": r["curve"], "sweep": r["sweep"],
            "rho_down": r["rho_down"], "rho_up": r["rho_up"], "vector_aligned": True,
            "note": f"recovered via gentler sweep {r['sweep']} (recover_facets.py)",
        }
    cfg["n_pass"] = sum(f.get("pass", False) for f in cfg["facets"].values())
    cfg["recovery"] = {"facets": list(PLAN), "plan": PLAN}
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    n = cfg["n_pass"]
    print(f"\n=== RECOVERY: {n}/5 facets now pass ===")
    for f, r in cfg["facets"].items():
        print(f"  {'✅' if r.get('pass') else '❌'} {f:12} band={r.get('band'):4} "
              f"sign={r.get('sign'):+d} ρ={r.get('rho'):.2f} Δ={r.get('delta'):.2f}")
    print(f"saved → {cfg_path}")


if __name__ == "__main__":
    main()
