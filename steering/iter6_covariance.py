"""
iter6_covariance.py — recover the trait CORRELATION STRUCTURE by modeling the manifold (D12).

iter-5 established the cross-facet entanglement is representational (a positive manifold), not a
fixable artifact. Rather than fight it, MODEL it: characterise the instrument's linear response
and invert it.

Pipeline:
  1. S, b — item response matrix. From the EXISTING single-facet forced-choice data
     (forced_choice.json: 35 personas × 25 items Δ), fit per item  Δ_i = b_i + Σ_g S[i][g]·θ_g
     by least squares (one-hot single-facet design). S (25×5) captures whatever each item
     actually measures, contamination included — sign-agnostic, no re-authoring.
  2. Multi-facet personas drawn from a KNOWN target covariance (two latent factors: f1 →
     Trust+Sociability, f2 → SelfControl+Conformity; Adaptability → 0.4(f1−f2)). Each facet is
     injected at its OWN band (zero the vector outside the band, sum, wrap the union) — matching
     how S's singles were administered.
  3. ADDITIVITY: is Δ(multi) ≈ b + S·θ?  (is the manifold linear?)  → pooled corr / per-item R².
  4. INVERT: θ̂ = lstsq(S, Δ−b) per persona — recover de-confounded trait profiles from all 25
     items jointly.
  5. PAYOFF: recovery matrix R[g][f]=slope(θ̂_f ~ θ_g) (want ≈ I), per-facet recovery corr, and
     cov(θ̂) vs cov(θ_imposed) — does the instrument recover the imposed correlation structure?
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
import personas as P
import forced_choice as FC

from paths import OUT
ITER = 6
FACETS = FC.IB.FACETS
ITEMS = FC.ITEMS

# band → positive layer indices (gguf layers 1..31; Llama-3.1-8B 32 layers).
# late = range(-1,-16) = layers 17..31 ; mid = range(-12,-21) = layers 12..20.
BAND_POS = {"late": set(range(17, 32)), "mid": set(range(12, 21))}
UNION_BAND = list(range(-1, -21, -1))   # layers 12..31 wrapped once


def restrict(vec, facet):
    """Zero the vector outside the facet's band, so a union-wrap injects each facet at its band."""
    keep = BAND_POS[P.FACET_BAND[facet]]
    dirs = {l: (d if l in keep else np.zeros_like(d)) for l, d in vec.directions.items()}
    return ControlVector(model_type=vec.model_type, directions=dirs)


def combined_restricted(theta, rvecs):
    cv = None
    for f, k in theta.items():
        if k == 0:
            continue
        term = rvecs[f] * float(k)
        cv = term if cv is None else cv + term
    return cv


# ---- target covariance ----
FA = [-1.2, -0.6, 0.0, 0.6, 1.0]
# per-facet usable θ ranges (mirror personas.DOSE one-sided saturation)
RANGE = {"SelfControl": (-2, 2), "Trust": (-2, 2), "Sociability": (-2, 2),
         "Conformity": (-2, 1), "Adaptability": (-2, 1)}


def _clip(theta):
    return {f: float(np.clip(v, *RANGE[f])) for f, v in theta.items()}


def build_multi():
    """Two-latent-factor design (Trust≡Socia, SC≡Conf perfectly collinear) — simple but
    degenerate: cannot separate the collinear pairs in the recovery matrix."""
    out = []
    for f1 in FA:
        for f2 in FA:
            theta = {"Trust": 0.8 * f1, "Sociability": 0.8 * f1,
                     "SelfControl": 0.8 * f2, "Conformity": 0.7 * f2,
                     "Adaptability": 0.4 * (f1 - f2)}
            out.append({"id": f"multi:f1{f1:+}:f2{f2:+}", "theta": _clip(theta)})
    return out


def build_multi_bifactor(n=30):
    """NON-degenerate bifactor design: each facet = λ_f·G (shared general/positive-manifold
    factor) + σ·U_f (facet-specific INDEPENDENT variance). Every facet has its own variance so
    the recovery matrix R is identifiable; pairwise correlations are graded (varied λ), none
    perfectly collinear — the bulletproof test of correlation-structure recovery."""
    rng = np.random.default_rng(20260616)
    lam = {"SelfControl": 0.6, "Trust": 0.5, "Sociability": 0.5,
           "Conformity": 0.6, "Adaptability": 0.3}
    sig = 0.6
    out = []
    for i in range(n):
        G = rng.uniform(-1.3, 1.3)
        theta = {f: 0.7 * (lam[f] * G + sig * rng.uniform(-1.3, 1.3)) for f in FACETS}
        out.append({"id": f"bifac:{i:02d}", "theta": _clip(theta)})
    return out


def fit_S():
    """S (n_items×5), b (n_items) from single-facet forced-choice deltas."""
    d = json.load(open(os.path.join(OUT, "forced_choice.json")))["delta"]
    pids = list(d)

    def th(pid):
        return pid.split(":")[1], float(pid.split(":")[2])

    # design: rows = single personas, cols = [1, θ_SC, θ_Tr, θ_So, θ_Co, θ_Ad]
    X = []
    for pid in pids:
        f, k = th(pid)
        row = [1.0] + [k if g == f else 0.0 for g in FACETS]
        X.append(row)
    X = np.array(X)
    S = np.zeros((len(ITEMS), len(FACETS))); b = np.zeros(len(ITEMS))
    for j, it in enumerate(ITEMS):
        y = np.array([d[pid][it["id"]] for pid in pids])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        b[j] = coef[0]; S[j] = coef[1:]
    return S, b


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["grid", "bifactor"], default="grid")
    args = ap.parse_args()
    tag = args.design
    out_name = "iter6_covariance.json" if tag == "grid" else f"iter6_covariance_{tag}.json"

    S, b = fit_S()
    print(f"fitted item response matrix S {S.shape} from single-facet forced-choice data", flush=True)
    print(f"  S condition number = {np.linalg.cond(S):.1f}  (lower = better-conditioned for inversion)")
    print(f"  design = {tag}")

    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in FACETS}
    rvecs = {f: restrict(vectors[f], f) for f in FACETS}

    personas = build_multi() if tag == "grid" else build_multi_bifactor()
    cmodel = ControlModel(base, UNION_BAND)
    Delta = {}   # pid -> [Δ per item]
    t0 = time.time()
    for n, p in enumerate(personas, 1):
        cmodel.reset()
        cv = combined_restricted(p["theta"], rvecs)
        if cv is not None:
            cmodel.set_control(cv, 1.0)
        scorer = PrefixScorer(cmodel, tok, FC.PRIME, n_options=2, device=device)
        d = []
        for it in ITEMS:
            s = scorer.score(it["stem"], [it["hi"], it["lo"]])
            d.append(s[0]["logprob_sum"] - s[1]["logprob_sum"])
        Delta[p["id"]] = d
        del scorer
        if n % 5 == 0:
            print(f"  [{n}/{len(personas)}] {time.time()-t0:.0f}s", flush=True)
    cmodel.unwrap(); del cmodel; gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ---- additivity: actual Δ vs predicted b + S·θ ----
    Theta = np.array([[p["theta"][f] for f in FACETS] for p in personas])   # 25×5 true
    A = np.array([Delta[p["id"]] for p in personas])                        # 25×n_items actual
    Pred = b[None, :] + Theta @ S.T                                         # 25×n_items predicted
    pooled_r = float(pearsonr(A.ravel(), Pred.ravel())[0])
    item_r2 = []
    for j in range(len(ITEMS)):
        a, q = A[:, j], Pred[:, j]
        if np.ptp(a) > 0:
            ss_res = np.sum((a - q) ** 2); ss_tot = np.sum((a - a.mean()) ** 2)
            item_r2.append(1 - ss_res / ss_tot)
    add_r2 = float(mean(item_r2))

    # ---- invert: θ̂ = lstsq(S, Δ−b) ----
    That = np.array([np.linalg.lstsq(S, A[i] - b, rcond=None)[0] for i in range(len(personas))])  # 25×5

    def corr(x, y):
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            return float("nan")
        return float(pearsonr(x, y)[0])

    # recovery matrix R[g][f] = slope(θ̂_f ~ θ_g_true); want ≈ I (after column scaling)
    R = np.zeros((5, 5))
    for gi in range(5):
        for fi in range(5):
            if np.ptp(Theta[:, gi]) > 0:
                R[gi, fi] = np.polyfit(Theta[:, gi], That[:, fi], 1)[0]
    rec_corr = {FACETS[fi]: corr(Theta[:, fi], That[:, fi]) for fi in range(5)}

    # covariance structure recovery
    Cimp = np.corrcoef(Theta.T)
    Crec = np.corrcoef(That.T)
    iu = np.triu_indices(5, k=1)
    offcorr = float(pearsonr(Cimp[iu], Crec[iu])[0])

    print(f"\n=== ITER {ITER} — manifold modeling & correlation-structure recovery (25 multi personas) ===")
    print(f"\n[ADDITIVITY]  pooled corr(actual, b+Sθ) = {pooled_r:.3f}   mean per-item R² = {add_r2:.3f}")
    print(f"  → {'LINEAR manifold: multi = superposition of singles' if add_r2 > 0.5 else 'NONlinear: superposition fails'}")

    print(f"\n[INVERSION]  per-facet recovery corr(θ̂_f, θ_f):")
    for f in FACETS:
        print(f"    {f:12} {rec_corr[f]:+.3f}")
    print(f"  mean = {mean(v for v in rec_corr.values() if not np.isnan(v)):+.3f}")
    print(f"\n  recovery matrix R[true g → recovered f] (want diagonal-dominant):")
    print(f"    {'g\\f':>12} " + " ".join(f"{f[:5]:>7}" for f in FACETS))
    for gi, g in enumerate(FACETS):
        print(f"    {g:>12} " + " ".join(f"{R[gi,fi]:+7.2f}" for fi in range(5)))

    print(f"\n[CORRELATION STRUCTURE]  corr(imposed off-diag, recovered off-diag) = {offcorr:.3f}")
    print(f"    imposed  vs  recovered facet-correlation (upper triangle):")
    names = [f"{FACETS[i][:4]}-{FACETS[j][:4]}" for i, j in zip(*iu)]
    for k, nm in enumerate(names):
        print(f"      {nm:12} imposed={Cimp[iu][k]:+.2f}  recovered={Crec[iu][k]:+.2f}")

    out = {"iter": ITER, "design": tag, "n_multi": len(personas), "S_cond": float(np.linalg.cond(S)),
           "additivity": {"pooled_r": round(pooled_r, 3), "mean_item_R2": round(add_r2, 3)},
           "recovery_corr": {f: (None if np.isnan(rec_corr[f]) else round(rec_corr[f], 3)) for f in FACETS},
           "recovery_matrix": {FACETS[gi]: {FACETS[fi]: round(float(R[gi, fi]), 3) for fi in range(5)} for gi in range(5)},
           "corr_struct_recovery": round(offcorr, 3),
           "imposed_corr": Cimp.tolist(), "recovered_corr": Crec.tolist(),
           "theta_true": Theta.tolist(), "theta_hat": That.tolist(),
           "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, out_name), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": ITER, "stage": f"covariance_recovery_{tag}",
                             "additivity_R2": round(add_r2, 3), "pooled_r": round(pooled_r, 3),
                             "corr_struct_recovery": round(offcorr, 3),
                             "recovery_corr": out["recovery_corr"]}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, out_name)} + metrics.jsonl  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
