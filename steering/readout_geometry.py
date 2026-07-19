"""
readout_geometry.py — iter-14b: the SECOND manifold, and the IDENTIFIABILITY law.

manifold_geometry.py showed the INPUT (steering-vector) manifold is ~16-dim with no positive
manifold, and the resistant facets (Sociability, Irritability) are NOT collinear with neighbors in
vector space. So the readout-invariant wall is not steering-vector collinearity. A first pass also
ruled out the naive culprit: nearest-NEIGHBOR readout cosine doesn't predict recovery either
(r≈0) — well-recovered facets (WellBeing, Order) are just as pairwise-collinear.

The right object is the recovery VARIANCE of the inversion itself. For Δ = b + Sθ + noise,
Var(θ̂_f) ∝ (SᵀS)⁻¹_ff = 1 / ( ‖S_f‖² · (1 − R²_f) ), where R²_f is the (through-origin) multiple
R² of facet f's readout column on ALL other columns. This factors EXACTLY into the iter-8 two gates:

    identifiability_f  =  ‖S_f‖   ×   sqrt(1 − R²_f)
                          └ induction ┘  └ separability ┘   (= INDUCTION GATE × SEPARABILITY GATE)

A facet is recoverable iff BOTH are non-trivial. Nearest-neighbor cosine missed it because a facet
can be far from any single neighbor yet lie in the SPAN of several (high multiple R², low pairwise).

This script computes identifiability per facet WITHIN each battery and correlates it (and each
gate) with that battery's own per-facet recovery:
  * FC      (135×25 usable) — recovery from covariance27_best.recovery_corr.
  * Oblique (75×25)         — recovery from oblique_inversion_pool θ_true/θ_hat.

Output -> research/loop/readout_geometry.json + figs/identifiability_*.png.
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import facets27 as F

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
FIGS = os.path.join(OUT, "figs")
DOSE = [-2.0, -1.0, 0.0, 1.0, 2.0]
DOMAIN_COLORS = {"Agreeableness": "#2ca02c", "Conscientiousness": "#1f77b4",
                 "Neuroticism": "#d62728", "Extraversion": "#ff7f0e", "Openness": "#9467bd"}
RESIST = ("Sociability", "Irritability")


def ols_slope(xs, ys):
    x = np.asarray(xs, float) - np.mean(xs)
    d = float((x * x).sum())
    return float((x * (np.asarray(ys, float) - np.mean(ys))).sum() / d) if d else 0.0


def build_fc_S(usable):
    fc = json.load(open(os.path.join(OUT, "fc27_matrix_best.json")))
    delta, items = fc["delta"], fc["item_order"]
    facets = [f for f in F.FACET_ORDER if f in delta and f in usable]
    S = np.zeros((len(items), len(facets)))
    for j, f in enumerate(facets):
        pd = delta[f]
        for i, it in enumerate(items):
            S[i, j] = ols_slope(DOSE, [pd[f"{d:+}"][it] for d in DOSE])
    return S, facets


def identifiability(S, facets):
    """Per-facet: strength ‖S_f‖, separability sqrt(1−R²_f), identifiability = product.
    R²_f via through-origin projection of column f on the span of the others."""
    K = len(facets)
    out = {}
    for j in range(K):
        sf = S[:, j]
        nf = float(np.linalg.norm(sf))
        others = np.delete(S, j, axis=1)
        # least-squares projection of sf onto span(others), through origin
        coef, *_ = np.linalg.lstsq(others, sf, rcond=None)
        resid = sf - others @ coef
        r2 = 1.0 - float(resid @ resid) / (nf * nf) if nf > 0 else 0.0
        sep = float(np.sqrt(max(0.0, 1.0 - r2)))
        out[facets[j]] = {"strength": round(nf, 3), "separability": round(sep, 3),
                          "identif": round(nf * sep, 3), "R2": round(r2, 3)}
    return out


def corr(xs, ys):
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    r = pearsonr(xs, ys)[0]
    return float(0.0 if r != r else r)


def analyse(name, S, facets, recovery):
    ident = identifiability(S, facets)
    fac = [f for f in facets if f in recovery]
    rec = np.array([recovery[f] for f in fac])
    strength = np.array([ident[f]["strength"] for f in fac])
    sep = np.array([ident[f]["separability"] for f in fac])
    idf = np.array([ident[f]["identif"] for f in fac])
    log_idf = np.log10(idf + 1e-9)
    res = {
        "name": name, "S_shape": list(S.shape), "facets": fac, "ident": ident,
        "r_identif_recovery": corr(log_idf, rec),
        "r_strength_recovery": corr(strength, rec),
        "r_separability_recovery": corr(sep, rec),
        "recovery": {f: round(float(recovery[f]), 3) for f in fac},
    }
    return res, (fac, strength, sep, idf, rec)


def main():
    cov = json.load(open(os.path.join(OUT, "covariance27_best.json")))
    fc_rec = cov["recovery_corr"]
    usable = list(fc_rec.keys())

    S_fc, fac_fc = build_fc_S(usable)
    fc_res, fc_pts = analyse("forced_choice", S_fc, fac_fc, fc_rec)

    o = json.load(open(os.path.join(OUT, "oblique_inversion_pool.json")))
    S_obl = np.array(o["S"]); fac_obl = o["usable"]
    tt, th = np.array(o["theta_true"]), np.array(o["theta_hat"])
    obl_rec = {f: corr(tt[:, j], th[:, j]) for j, f in enumerate(fac_obl)}
    obl_res, obl_pts = analyse("oblique", S_obl, fac_obl, obl_rec)

    for res in (fc_res, obl_res):
        print(f"\n=== IDENTIFIABILITY — {res['name']} battery {res['S_shape']} ===")
        print(f"  corr(recovery, log identifiability) = {res['r_identif_recovery']:+.3f}")
        print(f"  corr(recovery, strength/induction)  = {res['r_strength_recovery']:+.3f}")
        print(f"  corr(recovery, separability)        = {res['r_separability_recovery']:+.3f}")
        ident, rec = res["ident"], res["recovery"]
        print(f"  {'facet':24} {'strength':>8} {'separab':>8} {'identif':>8} {'recov':>7}")
        for f in sorted(res["facets"], key=lambda f: rec[f]):
            flag = "  <-- resistant" if f in RESIST else ""
            print(f"  {f:24} {ident[f]['strength']:8.2f} {ident[f]['separability']:8.2f} "
                  f"{ident[f]['identif']:8.2f} {rec[f]:7.2f}{flag}")

    # ---- figure: identifiability vs recovery (both batteries) + gate decomposition ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (res, pts) in zip(axes, [(fc_res, fc_pts), (obl_res, obl_pts)]):
        fac, strength, sep, idf, rec = pts
        for i, f in enumerate(fac):
            c = DOMAIN_COLORS[F.DOMAIN_OF[f]]
            big = f in RESIST
            ax.scatter(np.log10(idf[i] + 1e-9), rec[i], s=180 if big else 60, color=c,
                       edgecolor="black" if big else "none", linewidth=1.6, zorder=3 if big else 2)
            if big or rec[i] < 0.75:
                ax.annotate(f, (np.log10(idf[i] + 1e-9), rec[i]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.axhline(0.70, color="gray", ls=":", lw=1)
        ax.set_xlabel("log₁₀ identifiability  =  log( ‖S_f‖ · √(1−R²) )")
        ax.set_ylabel("recovery r")
        ax.set_title(f"{res['name']}: recovery vs identifiability "
                     f"(r = {res['r_identif_recovery']:+.2f})")
    for d, c in DOMAIN_COLORS.items():
        axes[0].scatter([], [], color=c, label=d)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "identifiability_vs_recovery.png"), dpi=130)
    plt.close(fig)

    # ---- figure: the two-gate map (strength × separability), recovery as color ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (res, pts) in zip(axes, [(fc_res, fc_pts), (obl_res, obl_pts)]):
        fac, strength, sep, idf, rec = pts
        sc = ax.scatter(strength, sep, c=rec, cmap="viridis", vmin=0.2, vmax=1.0,
                        s=[170 if f in RESIST else 55 for f in fac],
                        edgecolor=["black" if f in RESIST else "gray" for f in fac], linewidth=1.2)
        for i, f in enumerate(fac):
            if f in RESIST or rec[i] < 0.78:
                ax.annotate(f, (strength[i], sep[i]), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        ax.set_xlabel("INDUCTION gate — strength ‖S_f‖"); ax.set_ylabel("SEPARABILITY gate — √(1−R²)")
        ax.set_title(f"{res['name']}: the two gates")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="recovery r")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "two_gates_map.png"), dpi=130); plt.close(fig)

    out = {"stage": "readout_geometry", "forced_choice": fc_res, "oblique": obl_res}
    json.dump(out, open(os.path.join(OUT, "readout_geometry.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 14, "stage": "readout_geometry",
                             "fc_r_identif_recovery": round(fc_res["r_identif_recovery"], 3),
                             "fc_r_strength_recovery": round(fc_res["r_strength_recovery"], 3),
                             "fc_r_separability_recovery": round(fc_res["r_separability_recovery"], 3),
                             "obl_r_identif_recovery": round(obl_res["r_identif_recovery"], 3)}) + "\n")
    print(f"\nsaved -> readout_geometry.json + 3 figures")


if __name__ == "__main__":
    main()
