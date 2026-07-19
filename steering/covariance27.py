"""
covariance27.py — breadth 5→27 (D13), Pass 3: covariance recovery at 27D scale.

The iter-6 pipeline (model the linear manifold, invert, recover imposed trait covariance),
scaled from 5 to the full set of STEERABLE SPI facets:

  1. S, b (135×K) — fit per item from the single-facet response matrix (fc27_matrix.json),
     over the K = strong+weak facets from the Tier-0 scan (viability27.json). Broken facets
     (no induction → ~0 column) are excluded so S stays well-conditioned.
  2. Multi-facet personas from a KNOWN bifactor covariance over the K facets: each facet =
     λ_f·G (shared positive-manifold general factor) + σ·U_f (independent specific variance),
     graded λ so pairwise correlations are non-degenerate. Each facet injected at its OWN
     band/sign (restrict the vector to its band, scale by sign·θ, union-wrap once).
  3. ADDITIVITY  Δ(multi) ≈ b + S·θ ?     → pooled r, mean per-item R².
  4. INVERT      θ̂ = lstsq(S, Δ−b)        → recover de-confounded K-vector per persona.
  5. PAYOFF      recovery corr per facet, recovery matrix R, and corr(imposed, recovered)
                 over the K·(K−1)/2 off-diagonal facet correlations.

Output → research/loop/covariance27.json (+ metrics.jsonl).
"""
from __future__ import annotations
import os, json, time, gc, argparse
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
import personas as P

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
PRIME = "I'll be honest about myself.\n\n"
BAND_POS = {"late": set(range(17, 32)), "mid": set(range(12, 21))}
UNION_BAND = list(range(-1, -21, -1))   # layers 12..31


def restrict(vec, band):
    keep = BAND_POS[band]
    dirs = {l: (d if l in keep else np.zeros_like(d)) for l, d in vec.directions.items()}
    return ControlVector(model_type=vec.model_type, directions=dirs)


def fit_S(matrix, usable):
    """S (135×K), b (135) from single-facet forced-choice deltas in fc27_matrix.json.
    Design rows = every (facet, θ) single administration; cols = [1] + [θ_g for g in usable]."""
    delta = matrix["delta"]
    item_ids = matrix["item_order"]
    rows, ys_by_item = [], {iid: [] for iid in item_ids}
    for f in usable:
        for th_str, row in delta[f].items():
            theta = float(th_str)
            rows.append([1.0] + [theta if g == f else 0.0 for g in usable])
            for iid in item_ids:
                ys_by_item[iid].append(row[iid])
    X = np.array(rows)
    S = np.zeros((len(item_ids), len(usable))); b = np.zeros(len(item_ids))
    for j, iid in enumerate(item_ids):
        coef, *_ = np.linalg.lstsq(X, np.array(ys_by_item[iid]), rcond=None)
        b[j] = coef[0]; S[j] = coef[1:]
    return S, b, item_ids


def build_bifactor(usable, n, seed=20260617):
    """Non-degenerate bifactor design over the usable facets. λ_f graded → pairwise corr varied."""
    rng = np.random.default_rng(seed)
    lam = {f: float(rng.uniform(0.2, 0.7)) for f in usable}
    sig = 0.6
    out = []
    for i in range(n):
        G = rng.uniform(-1.3, 1.3)
        theta = {f: float(np.clip(0.7 * (lam[f] * G + sig * rng.uniform(-1.3, 1.3)), -2, 2))
                 for f in usable}
        out.append({"id": f"bifac:{i:03d}", "theta": theta})
    return out, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of multi-facet personas")
    ap.add_argument("--items", choices=["v1", "best", "all"], default="v1")
    ap.add_argument("--include-weak", action="store_true", default=True)
    args = ap.parse_args()
    via_name = "viability27.json" if args.items == "v1" else "viability27_best.json"
    mat_name = "fc27_matrix.json" if args.items == "v1" else f"fc27_matrix_{args.items}.json"
    out_name = "covariance27.json" if args.items == "v1" else f"covariance27_{args.items}.json"
    ITEMS = F.fc_items(args.items)
    print(f"item set = {args.items} ({len(ITEMS)} items)", flush=True)

    via = json.load(open(os.path.join(OUT, via_name)))["facets"]
    matrix = json.load(open(os.path.join(OUT, mat_name)))
    usable = [f for f in F.FACET_ORDER
              if via[f]["verdict"] in (("strong", "weak") if args.include_weak else ("strong",))]
    K = len(usable)
    print(f"usable facets (K={K}): {usable}", flush=True)

    S, b, item_ids = fit_S(matrix, usable)
    cond = float(np.linalg.cond(S))
    print(f"fitted S {S.shape} from single-facet matrix; cond(S) = {cond:.1f}", flush=True)

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
        for it in ITEMS:
            s = scorer.score(it["stem"], [it["hi"], it["lo"]])
            row[id2idx[it["id"]]] = s[0]["logprob_sum"] - s[1]["logprob_sum"]
        Delta[p["id"]] = row
        del scorer
        if n % 5 == 0:
            print(f"  [{n}/{len(personas)}] {time.time()-t0:.0f}s", flush=True)
    cmodel.unwrap(); del cmodel; gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ---- additivity ----
    Theta = np.array([[p["theta"][f] for f in usable] for p in personas])   # n×K
    A = np.array([Delta[p["id"]] for p in personas])                        # n×135
    Pred = b[None, :] + Theta @ S.T
    pooled_r = float(pearsonr(A.ravel(), Pred.ravel())[0])
    item_r2 = []
    for j in range(len(item_ids)):
        a, q = A[:, j], Pred[:, j]
        if np.ptp(a) > 0:
            item_r2.append(1 - np.sum((a - q) ** 2) / np.sum((a - a.mean()) ** 2))
    add_r2 = float(mean(item_r2))

    # ---- invert ----
    That = np.array([np.linalg.lstsq(S, A[i] - b, rcond=None)[0] for i in range(len(personas))])

    def corr(x, y):
        return float("nan") if np.ptp(x) == 0 or np.ptp(y) == 0 else float(pearsonr(x, y)[0])

    rec_corr = {usable[fi]: corr(Theta[:, fi], That[:, fi]) for fi in range(K)}
    Cimp = np.corrcoef(Theta.T)
    Crec = np.corrcoef(That.T)
    iu = np.triu_indices(K, k=1)
    offcorr = float(pearsonr(Cimp[iu], Crec[iu])[0])
    # mean abs error on off-diagonal correlations
    off_mae = float(np.mean(np.abs(Cimp[iu] - Crec[iu])))

    rc = [v for v in rec_corr.values() if not np.isnan(v)]
    print(f"\n=== COVARIANCE RECOVERY @ {K}D  ({len(personas)} multi personas, {time.time()-t0:.0f}s) ===")
    print(f"\n[ADDITIVITY]  pooled r = {pooled_r:.3f}   mean per-item R² = {add_r2:.3f}")
    print(f"\n[INVERSION]  per-facet recovery corr(θ̂_f, θ_f):")
    for f in sorted(usable, key=lambda f: -(rec_corr[f] if not np.isnan(rec_corr[f]) else -9)):
        v = rec_corr[f]
        print(f"    {f:26} {('  nan' if np.isnan(v) else f'{v:+.3f}')}")
    print(f"  mean recovery r = {mean(rc):+.3f}   (min {min(rc):+.3f})")
    print(f"\n[CORRELATION STRUCTURE]  corr(imposed off-diag, recovered off-diag) = {offcorr:.3f}")
    print(f"  mean |Δcorr| over {len(iu[0])} facet pairs = {off_mae:.3f}")

    out = {"stage": f"covariance_recovery_27_{args.items}", "n_multi": len(personas), "K": K,
           "usable": usable, "S_shape": list(S.shape), "S_cond": cond,
           "lambda": {f: round(lam[f], 3) for f in usable},
           "additivity": {"pooled_r": round(pooled_r, 3), "mean_item_R2": round(add_r2, 3)},
           "recovery_corr": {f: (None if np.isnan(rec_corr[f]) else round(rec_corr[f], 3)) for f in usable},
           "mean_recovery_r": round(mean(rc), 3),
           "corr_struct_recovery": round(offcorr, 3), "off_diag_mae": round(off_mae, 3),
           "imposed_corr": Cimp.tolist(), "recovered_corr": Crec.tolist(),
           "theta_true": Theta.tolist(), "theta_hat": That.tolist(),
           "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, out_name), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 7, "stage": f"covariance_recovery_27_{args.items}", "K": K,
                             "additivity_R2": round(add_r2, 3), "pooled_r": round(pooled_r, 3),
                             "corr_struct_recovery": round(offcorr, 3),
                             "off_diag_mae": round(off_mae, 3),
                             "mean_recovery_r": round(mean(rc), 3)}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, out_name)} + metrics.jsonl  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
