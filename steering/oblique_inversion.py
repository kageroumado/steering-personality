"""
oblique_inversion.py — iter-10 (product viability): does the OBLIQUE battery invert to a 27D reading?

The decisive product test. Per the modeled-manifold philosophy (iter-6), individual questions need
not be pure — what matters is whether the loading matrix inverts. The oblique calibration (iter-9)
gave a battery with effective rank ~14 / cond 27 (not a rank-1 valence axis), so in principle it
should recover an imposed personality. This checks it end to end:

  1. S_oblique, b — fit each oblique item's response from the iter-9 single-facet sweep
     (oblique_calibration.json delta), over the K usable facets.
  2. Impose multi-facet personas from a known bifactor covariance (same design family as iter-7).
  3. Administer ONLY the 35 oblique items live under each persona (restrict + union-wrap).
  4. Additivity, invert θ̂ = lstsq(S, Δ−b), per-facet recovery, covariance recovery.

This is exactly the human product's scoring path: a respondent's oblique answers → Δ → θ̂ (27D).
Output → research/loop/oblique_inversion.json.
"""
from __future__ import annotations
import os, json, time, gc
from statistics import mean
import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
from extract_vectors import VEC_DIR
import facets27 as F
import oblique_items as OI
from covariance27 import restrict, fit_S, build_bifactor, UNION_BAND, PRIME

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")


def merged_delta(cals):
    """Merge per-facet/dose item-delta dicts from several calibration files."""
    out = {}
    for cal in cals:
        for facet, by_dose in cal["delta"].items():
            out.setdefault(facet, {})
            for k, row in by_dose.items():
                out.setdefault(facet, {}).setdefault(k, {}).update(row)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--bank", choices=["b1", "pool"], default="b1")
    args = ap.parse_args()
    out_name = "oblique_inversion.json" if args.bank == "b1" else f"oblique_inversion_{args.bank}.json"

    via = json.load(open(os.path.join(OUT, "viability27_best.json")))["facets"]
    if args.bank == "pool":
        OBLIQUE = OI.POOL
        cals = [json.load(open(os.path.join(OUT, "oblique_calibration.json"))),
                json.load(open(os.path.join(OUT, "oblique_calibration_b2.json")))]
        delta = merged_delta(cals)
    else:
        OBLIQUE = OI.OBLIQUE
        delta = json.load(open(os.path.join(OUT, "oblique_calibration.json")))["delta"]
    usable = [f for f in F.FACET_ORDER if via[f]["verdict"] in ("strong", "weak")]
    K = len(usable)
    print(f"bank = {args.bank} ({len(OBLIQUE)} items); K={K}", flush=True)

    # reuse covariance27.fit_S: needs a matrix dict with delta[facet][θ][item] and item_order
    item_ids = [it["id"] for it in OBLIQUE]
    matrix = {"delta": delta, "item_order": item_ids}
    S, b, _ = fit_S(matrix, usable)
    cond = float(np.linalg.cond(S))
    print(f"oblique S {S.shape} from iter-9 calibration; cond(S) = {cond:.1f}; K={K}", flush=True)

    personas, lam = build_bifactor(usable, args.n)

    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    rvecs = {f: restrict(ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")),
                         via[f]["band"]) for f in usable}
    signs = {f: via[f]["sign"] for f in usable}

    id2idx = {iid: j for j, iid in enumerate(item_ids)}
    cmodel = ControlModel(base, UNION_BAND)
    Delta = {}
    t0 = time.time()
    for n, p in enumerate(personas, 1):
        cmodel.reset()
        cv = None
        for f, k in p["theta"].items():
            if k == 0:
                continue
            term = rvecs[f] * float(signs[f] * k)
            cv = term if cv is None else cv + term
        if cv is not None:
            cmodel.set_control(cv, 1.0)
        scorer = PrefixScorer(cmodel, tok, PRIME, n_options=2, device=device)
        row = [0.0] * len(item_ids)
        for it in OBLIQUE:
            s = scorer.score(it["stem"], [it["hi"], it["lo"]])
            row[id2idx[it["id"]]] = s[0]["logprob_sum"] - s[1]["logprob_sum"]
        Delta[p["id"]] = row
        del scorer
        if n % 5 == 0:
            print(f"  [{n}/{len(personas)}] {time.time()-t0:.0f}s", flush=True)
    cmodel.unwrap(); del cmodel; gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    Theta = np.array([[p["theta"][f] for f in usable] for p in personas])
    A = np.array([Delta[p["id"]] for p in personas])
    Pred = b[None, :] + Theta @ S.T
    pooled_r = float(pearsonr(A.ravel(), Pred.ravel())[0])
    item_r2 = [1 - np.sum((A[:, j] - Pred[:, j]) ** 2) / np.sum((A[:, j] - A[:, j].mean()) ** 2)
               for j in range(len(item_ids)) if np.ptp(A[:, j]) > 0]
    add_r2 = float(mean(item_r2))

    That = np.array([np.linalg.lstsq(S, A[i] - b, rcond=None)[0] for i in range(len(personas))])

    def corr(x, y):
        return float("nan") if np.ptp(x) == 0 or np.ptp(y) == 0 else float(pearsonr(x, y)[0])
    rec_corr = {usable[fi]: corr(Theta[:, fi], That[:, fi]) for fi in range(K)}
    Cimp = np.corrcoef(Theta.T); Crec = np.corrcoef(That.T)
    iu = np.triu_indices(K, k=1)
    offcorr = float(pearsonr(Cimp[iu], Crec[iu])[0])
    off_mae = float(np.mean(np.abs(Cimp[iu] - Crec[iu])))
    rc = [v for v in rec_corr.values() if not np.isnan(v)]
    well = sum(1 for v in rc if v >= 0.70)

    print(f"\n=== OBLIQUE INVERSION @ {K}D  ({len(personas)} personas, oblique battery only) ===")
    print(f"[ADDITIVITY]  pooled r = {pooled_r:.3f}   per-item R² = {add_r2:.3f}")
    print(f"[INVERSION]   mean recovery r = {mean(rc):+.3f}  (min {min(rc):+.3f}); "
          f"facets recovered ≥0.70: {well}/{K}")
    for f in sorted(usable, key=lambda f: -(rec_corr[f] if not np.isnan(rec_corr[f]) else -9)):
        v = rec_corr[f]
        print(f"    {f:26} {('  nan' if np.isnan(v) else f'{v:+.3f}')}")
    print(f"[COVARIANCE]  corr-struct r = {offcorr:.3f}  MAE = {off_mae:.3f}")

    out = {"stage": "oblique_inversion", "n_multi": len(personas), "K": K, "usable": usable,
           "S_shape": list(S.shape), "S_cond": cond,
           "additivity": {"pooled_r": round(pooled_r, 3), "mean_item_R2": round(add_r2, 3)},
           "recovery_corr": {f: (None if np.isnan(rec_corr[f]) else round(rec_corr[f], 3)) for f in usable},
           "mean_recovery_r": round(mean(rc), 3), "n_recovered_70": well,
           "corr_struct_recovery": round(offcorr, 3), "off_diag_mae": round(off_mae, 3),
           "theta_true": Theta.tolist(), "theta_hat": That.tolist(),
           "item_order": item_ids, "S": S.tolist(), "b": b.tolist(),
           "A_multi": A.tolist(), "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, out_name), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 10, "stage": "oblique_inversion", "K": K,
                             "additivity_R2": round(add_r2, 3),
                             "mean_recovery_r": round(mean(rc), 3), "n_recovered_70": well,
                             "corr_struct_recovery": round(offcorr, 3)}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, out_name)} + metrics.jsonl  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
