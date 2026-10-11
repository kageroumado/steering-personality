"""
native_factors.py — iter-15a: the model's NATIVE trait factors from the INPUT geometry.

Kiri's reframe: don't recover the SPI — the SPI is itself an imperfect human artifact. The Big Five
was DISCOVERED bottom-up (factor-analyze the lexicon), not handed down. So derive the model's own
factor structure from the data and THEN compare it to the human taxonomy; convergence and divergence
are both findings.

This is the cheap, no-model route: factor-analyze the 27x27 sign-corrected cosine Gram of the
steering vectors (input geometry) — "how the model encodes trait directions." PCA -> varimax ->
name factors by loadings -> Tucker congruence vs the a-priori Big Five DOMAINS. The behavioral
route (factor-analyze FC-27 scores across a persona population) is iter-15b (`persona_population.py`).

Output -> research/loop/native_factors.json + figs/native_factor_loadings.png.
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from repeng import ControlVector
from extract_vectors import VEC_DIR
import facets27 as F

from paths import OUT, FIGS
HEADLINE_LAYER = 13


def varimax(Phi, gamma=1.0, q=100, tol=1e-7):
    p, k = Phi.shape
    R = np.eye(k); d = 0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vh = np.linalg.svd(Phi.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vh; d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R


def gram(layer):
    via = json.load(open(os.path.join(OUT, "viability27_best.json")))["facets"]
    M = []
    for f in F.FACET_ORDER:
        cv = ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf"))
        v = (via[f]["sign"] * cv.directions[layer]).astype(float)
        M.append(v / np.linalg.norm(v))
    M = np.array(M)
    return M @ M.T


def congruence(a, b):
    """Tucker's phi (sign-insensitive: factors and domain indicators may be flipped)."""
    return abs(float(a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-12))


def main():
    C = gram(HEADLINE_LAYER)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    eig = np.clip(w[idx], 0, None)
    V = V[:, idx]
    K = 5  # compare to the Big Five's 5
    Phi = V[:, :K] * np.sqrt(eig[:K])      # PCA loadings (27 x 5)
    L = varimax(Phi)                        # rotated loadings (27 x 5)
    # order rotated factors by variance (column SS) desc
    ss = (L ** 2).sum(0)
    order = np.argsort(ss)[::-1]
    L = L[:, order]
    var_explained = float(eig[:K].sum() / eig.sum())

    # name each factor by its strongest +/- loadings
    factors = []
    for j in range(K):
        col = L[:, j]
        o = np.argsort(col)
        pos = [(F.FACET_ORDER[i], round(float(col[i]), 2)) for i in o[::-1][:5] if col[i] > 0.15]
        neg = [(F.FACET_ORDER[i], round(float(col[i]), 2)) for i in o[:5] if col[i] < -0.15]
        factors.append({"factor": j + 1, "pos": pos, "neg": neg,
                        "var_share": round(float(ss[order][j] / ss.sum()), 3)})

    # congruence vs a-priori Big Five domain indicators (centered membership vectors)
    dom_names = list(F.DOMAINS.keys())
    domI = np.zeros((27, 5))
    for di, d in enumerate(dom_names):
        for i, f in enumerate(F.FACET_ORDER):
            domI[i, di] = 1.0 if F.DOMAIN_OF[f] == d else 0.0
    domI -= domI.mean(0, keepdims=True)
    Cong = np.zeros((K, 5))
    for j in range(K):
        for di in range(5):
            Cong[j, di] = congruence(L[:, j], domI[:, di])
    # Hungarian match factors -> domains (maximize congruence)
    ri, ci = linear_sum_assignment(-Cong)
    matches = {dom_names[ci[m]]: {"factor": int(ri[m]) + 1, "congruence": round(float(Cong[ri[m], ci[m]]), 3)}
               for m in range(K)}
    mean_cong = float(np.mean([matches[d]["congruence"] for d in matches]))

    print(f"\n=== NATIVE FACTORS — input geometry (layer {HEADLINE_LAYER}, K={K} varimax) ===")
    print(f"  5 factors capture {var_explained*100:.0f}% of vector variance "
          f"(eff-rank ~16 → the model is HIGHER-dimensional than 5)")
    for fa in factors:
        pos = ", ".join(f"{n}({v:+})" for n, v in fa["pos"])
        neg = ", ".join(f"{n}({v:+})" for n, v in fa["neg"])
        print(f"\n  Factor {fa['factor']} ({fa['var_share']*100:.0f}% of rotated var):")
        print(f"     + {pos}")
        if neg:
            print(f"     − {neg}")

    print(f"\n  --- congruence with a-priori Big Five (Tucker φ; >.85 fair, >.95 identical) ---")
    for d in dom_names:
        m = matches[d]
        tag = "identical" if m["congruence"] > 0.95 else "fair" if m["congruence"] > 0.85 else "WEAK"
        print(f"    {d:18} ↔ Factor {m['factor']}   φ = {m['congruence']:.2f}  [{tag}]")
    print(f"\n  mean congruence = {mean_cong:.3f}  → "
          f"{'native ≈ Big Five' if mean_cong > 0.85 else 'native structure DIVERGES from Big Five'}")

    # ---- figure: rotated loading heatmap ----
    fig, ax = plt.subplots(figsize=(7.5, 9))
    im = ax.imshow(L, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(27)); ax.set_yticklabels(F.FACET_ORDER, fontsize=7)
    for i, f in enumerate(F.FACET_ORDER):
        ax.get_yticklabels()[i].set_color(
            {"Agreeableness": "#2ca02c", "Conscientiousness": "#1f77b4", "Neuroticism": "#d62728",
             "Extraversion": "#ff7f0e", "Openness": "#9467bd"}[F.DOMAIN_OF[f]])
    ax.set_xticks(range(K)); ax.set_xticklabels([f"F{j+1}" for j in range(K)])
    ax.set_title(f"Native factors (input geometry, varimax)\nrow color = a-priori Big Five domain")
    fig.colorbar(im, ax=ax, shrink=0.6, label="loading")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "native_factor_loadings.png"), dpi=130)
    plt.close(fig)

    out = {"stage": "native_factors_input", "layer": HEADLINE_LAYER, "K": K,
           "var_explained_5": var_explained, "factors": factors,
           "congruence_matrix": Cong.tolist(), "domain_names": dom_names,
           "matches": matches, "mean_congruence": mean_cong,
           "loadings": L.tolist(), "facet_order": F.FACET_ORDER}
    json.dump(out, open(os.path.join(OUT, "native_factors.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 15, "stage": "native_factors_input",
                             "var_explained_5": round(var_explained, 3),
                             "mean_congruence_bigfive": round(mean_cong, 3)}) + "\n")
    print(f"\nsaved -> native_factors.json + figs/native_factor_loadings.png")


if __name__ == "__main__":
    main()
