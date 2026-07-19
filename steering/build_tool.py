"""
build_tool.py — iter-20: the human-facing tool — scoring engine, confound model, precision, typology.

Turns the formal native model (iter-19, native_model.json: N=200 personas × 27 facets, 27×5 varimax
loadings, per-factor reliabilities) into a *human-facing instrument*:

  1. NORMS        — per-facet calibration (mean/sd over the 200 reference personas) → z-scores.
  2. CONFOUND     — a bifactor decomposition: estimate a general response/desirability factor g
                    (1st PC of the standardized facets), characterize it (desirability correlation,
                    top facets), and produce metatrait scores NET of g ("purified"). Report how much
                    each metatrait shifts when g is removed = confound contamination.
  3. SCORE        — loading-weighted metatrait scale scores on the g-residualized facets, mapped to a
                    1–10 decile readout (the headline number) with the raw z as the secondary value.
  4. PRECISION    — per-metatrait reliability (Cronbach α) → SE of measurement → a 1–10 confidence
                    band, shown per axis. A 3-axis solid core, a 2-axis provisional tail.
  5. TYPOLOGY     — cluster the 200 personas in 5-metatrait space (k chosen by silhouette), name the
                    types by their metatrait signature, list the occupations in each (our personas
                    ARE occupations → intrinsic job→type map), give each type's Big-Five domain
                    profile, and bootstrap cluster stability (adjusted Rand index).

No model run — everything is computed from native_model.json. Output -> research/loop/human_tool.json
+ figs/tool_*.png.
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import facets27 as F

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
FIGS = os.path.join(OUT, "figs")
RNG = np.random.default_rng(20)

METATRAITS = ["Plasticity", "Drive", "Fluidity", "Stability", "Communion"]
CORE = [0, 1, 2]          # the reliable 3 (α ≥ 0.81); 3,4 are the provisional tail
LOAD_THRESH = 0.35

# Rough social-desirability sign of each facet (+ generally seen as desirable, − as undesirable,
# 0 neutral) — used ONLY to characterize what the general factor g resembles, not to score anyone.
DESIRABILITY = {
    "WellBeing": +1, "EmotionalStability": +1, "Honesty": +1, "Compassion": +1, "Trust": +1,
    "Industry": +1, "SelfControl": +1, "Adaptability": +1, "Charisma": +1, "Humor": +1,
    "Creativity": +1, "Intellect": +1, "Introspection": +1, "ArtAppreciation": +1, "Order": +1,
    "Perfectionism": +1, "EmotionalExpressiveness": +1, "Sociability": +1,
    "Anxiety": -1, "Irritability": -1, "Impulsivity": -1, "AttentionSeeking": -1,
    "SensationSeeking": -1, "Authoritarianism": -1,
    "Conformity": 0, "Conservatism": 0,
}


def varimax(Phi, q=200, tol=1e-8):
    p, k = Phi.shape
    R = np.eye(k); d = 0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vh = np.linalg.svd(Phi.T @ (L ** 3 - (1.0 / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vh; d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R


def congruence(a, b):
    return abs(float(a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-12))


def efa(Z, K):
    """varimax EFA on already-standardized Z (n×p). Returns p×K loadings, columns sorted by var."""
    C = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(C); idx = np.argsort(w)[::-1]
    eig = np.clip(w[idx], 0, None); V = V[:, idx]
    L = varimax(V[:, :K] * np.sqrt(eig[:K]))
    return L[:, np.argsort((L ** 2).sum(0))[::-1]]


def cronbach_alpha(items):
    m = items.shape[1]
    if m < 2:
        return float("nan")
    var_sum = items.var(0, ddof=1).sum()
    tot_var = items.sum(1).var(ddof=1)
    return float((m / (m - 1)) * (1 - var_sum / tot_var)) if tot_var > 0 else float("nan")


def scale_scores(Z, L):
    """loading-weighted scale score per factor: (Σ w_i z_i)/Σ|w_i| over |loading|>thresh, sign-kept."""
    n, _ = Z.shape
    K = L.shape[1]
    S = np.zeros((n, K))
    member = []
    for f in range(K):
        idx = np.where(np.abs(L[:, f]) > LOAD_THRESH)[0]
        w = L[idx, f]
        S[:, f] = (Z[:, idx] @ w) / (np.abs(w).sum() + 1e-12)
        member.append(idx)
    return S, member


def adjusted_rand(a, b):
    """Adjusted Rand Index between two integer labelings."""
    a = np.asarray(a); b = np.asarray(b)
    ca = np.unique(a); cb = np.unique(b)
    cont = np.array([[np.sum((a == i) & (b == j)) for j in cb] for i in ca], float)
    ai = cont.sum(1); bj = cont.sum(0); n = cont.sum()
    comb = lambda x: x * (x - 1) / 2.0
    idx = comb(cont).sum()
    exp = comb(ai).sum() * comb(bj).sum() / comb(n)
    mx = 0.5 * (comb(ai).sum() + comb(bj).sum())
    return float((idx - exp) / (mx - exp)) if mx != exp else 1.0


def silhouette(M, labels):
    D = cdist(M, M)
    labs = np.unique(labels)
    sil = np.zeros(len(M))
    for i in range(len(M)):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, labels == c].mean() for c in labs if c != labels[i])
        sil[i] = (b - a) / (max(a, b) + 1e-12)
    return float(sil.mean())


def main():
    TAG = os.environ.get("TOOL_TAG", "")            # e.g. "_n400" to confirm on the stabilized model
    d = json.load(open(os.path.join(OUT, os.environ.get("TOOL_INPUT", "native_model.json"))))
    X = np.array(d["scores"])                       # 200 × 27 raw logprob deltas
    L0 = np.array(d["loadings"])                    # 27 × 5 varimax loadings (raw)
    facets = d["facet_order"]; personas = d["personas"]
    N, P = X.shape; K = L0.shape[1]
    fac_idx = {f: i for i, f in enumerate(facets)}

    # ---- 1. NORMS: per-facet z-scores over the reference population ----
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Z = (X - mu) / sd

    # ---- 2. CONFOUND: bifactor g = 1st PC of the standardized facets ----
    C = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(C); order = np.argsort(w)[::-1]
    eig = w[order]; V = V[:, order]
    vg = V[:, 0]
    if vg.sum() < 0:                                # orient g toward the positive direction
        vg = -vg
    g_var = float(eig[0] / P)                       # share of total variance carried by g
    g_score = Z @ vg                                # per-persona general-factor score
    g_score = (g_score - g_score.mean()) / (g_score.std() + 1e-9)
    # characterize g: correlation of its facet loadings with social-desirability sign
    des = np.array([DESIRABILITY.get(f, 0) for f in facets], float)
    g_des_r = float(np.corrcoef(vg, des)[0, 1])
    g_top = sorted(((facets[i], float(vg[i])) for i in range(P)), key=lambda t: -abs(t[1]))[:8]

    # residualize the rank-1 general component out of every facet
    Z_resid = Z - np.outer(g_score, vg * (Z.std(0).mean()))   # remove g direction
    # re-standardize residuals
    Z_resid = (Z_resid - Z_resid.mean(0)) / (Z_resid.std(0) + 1e-9)

    # ---- purified metatraits: re-derive 5-factor varimax on residuals, match to raw ----
    L_pure = efa(Z_resid, K)
    # align purified factors to raw factors by congruence (Hungarian) + sign
    Cong = np.array([[congruence(L0[:, i], L_pure[:, j]) for j in range(K)] for i in range(K)])
    ri, ci = linear_sum_assignment(-Cong)
    perm = np.zeros(K, int)
    for a, b in zip(ri, ci):
        perm[a] = b
    L_pure = L_pure[:, perm]
    for f in range(K):                              # fix sign to match raw
        if (L0[:, f] @ L_pure[:, f]) < 0:
            L_pure[:, f] *= -1
    metatrait_congruence = [float(congruence(L0[:, f], L_pure[:, f])) for f in range(K)]

    # ---- 3. SCORE: loading-weighted scale scores on purified facets ----
    S_raw, members = scale_scores(Z, L0)                       # raw (g included)
    S_pure, _ = scale_scores(Z_resid, L_pure)                 # purified (g removed)
    # standardize the purified scores → the canonical metatrait space for typing
    M = (S_pure - S_pure.mean(0)) / (S_pure.std(0) + 1e-9)

    # 1–10 decile readout: empirical percentile rank of each persona on each metatrait
    def decile(col):
        r = col.argsort().argsort()                            # ranks 0..N-1
        return np.clip(np.floor(r / N * 10).astype(int) + 1, 1, 10)
    deciles = np.column_stack([decile(M[:, f]) for f in range(K)])

    # ---- 4. PRECISION: per-metatrait reliability → SE → ±band in decile units ----
    rel = []
    for f in range(K):
        idx = members[f]
        signed = Z[:, idx] * np.sign(L0[idx, f])
        rel.append(cronbach_alpha(signed))
    rel = np.array(rel)
    se_z = np.sqrt(np.clip(1 - rel, 0, 1))                     # SE of a standardized score
    # SE in z → decile half-width: the standardized score spans ~rng_z over 10 deciles
    rng_z = M.max(0) - M.min(0)
    band_dec = se_z * (10.0 / (rng_z + 1e-9))

    # ---- 5. TYPOLOGY: cluster the 200 in metatrait space ----
    # choose k by silhouette over 3..8 (use only the reliable CORE axes for stable typing,
    # but keep all 5 in the centroid description)
    Mc = M[:, CORE]
    best_k, best_sil, best_lab, best_cent = None, -1, None, None
    sil_curve = {}
    for k in range(3, 9):
        cent, lab = kmeans2(Mc, k, minit="++", seed=20, iter=50)
        if len(np.unique(lab)) < k:                            # empty cluster → skip
            sil_curve[k] = float("nan"); continue
        s = silhouette(Mc, lab)
        sil_curve[k] = s
        if s > best_sil:
            best_k, best_sil, best_lab, best_cent = k, s, lab, cent
    labels = best_lab

    # bootstrap cluster stability: resample personas, recluster, ARI vs the full labeling on shared pts
    aris = []
    for _ in range(100):
        samp = RNG.choice(N, N, replace=True)
        uniq = np.unique(samp)
        _, lab_b = kmeans2(Mc[uniq], best_k, minit="++", seed=int(RNG.integers(1e6)), iter=50)
        if len(np.unique(lab_b)) < best_k:
            continue
        # map full labels onto uniq points, compare
        aris.append(adjusted_rand(labels[uniq], lab_b))
    stability = float(np.mean(aris)) if aris else float("nan")

    # Big Five domain profile per facet (a-priori DOMAINS) for cross-reference
    dom_names = list(F.DOMAINS.keys())
    domZ = np.zeros((N, len(dom_names)))
    for di, dn in enumerate(dom_names):
        cols = [fac_idx[f] for f in F.DOMAINS[dn]]
        domZ[:, di] = Z[:, cols].mean(1)
    domZ = (domZ - domZ.mean(0)) / (domZ.std(0) + 1e-9)

    # describe each type
    HI = {0: "Expressive", 1: "Driven", 2: "Adaptive", 3: "Serene", 4: "Warm"}
    LO = {0: "Grounded", 1: "Placid", 2: "Exacting", 3: "Turbulent", 4: "Detached"}
    types = []
    for c in range(best_k):
        mask = labels == c
        cent = M[mask].mean(0)                                 # centroid over ALL 5 metatraits
        sig = []
        for f in np.argsort(-np.abs(cent)):
            if abs(cent[f]) > 0.45:
                sig.append((HI if cent[f] > 0 else LO)[f] + f" {METATRAITS[f]}({cent[f]:+.2f})")
        occ = [personas[i] for i in np.where(mask)[0]]
        dom_profile = {dom_names[di]: round(float(domZ[mask, di].mean()), 2) for di in range(len(dom_names))}
        types.append({
            "type": c, "n": int(mask.sum()),
            "centroid": {METATRAITS[f]: round(float(cent[f]), 2) for f in range(K)},
            "signature": sig,
            "bigfive": dom_profile,
            "example_occupations": occ[:8],
        })
    types.sort(key=lambda t: -t["n"])

    # ---------- report ----------
    print(f"=== HUMAN-FACING TOOL (N={N}) ===\n")
    print(f"[confound] general factor g carries {g_var*100:.1f}% of total facet variance")
    print(f"          g resembles social desirability: r(g, desirability-sign) = {g_des_r:+.2f}")
    print(f"          top g facets: " + ", ".join(f"{f}({v:+.2f})" for f, v in g_top[:6]))
    print(f"[confound] metatrait congruence raw→purified (how stable to removing g):")
    for f in range(K):
        print(f"          {METATRAITS[f]:11s} φ={metatrait_congruence[f]:.2f}")
    print(f"\n[precision] per-metatrait reliability and 1–10 confidence band:")
    for f in range(K):
        tier = "CORE" if f in CORE else "prov"
        print(f"          {METATRAITS[f]:11s} α={rel[f]:.2f}  SE±{band_dec[f]:.1f} deciles  [{tier}]")
    print(f"          mean reliability (core 3) = {rel[CORE].mean():.2f}; (all 5) = {rel.mean():.2f}")
    print(f"\n[typology] k={best_k} types (silhouette={best_sil:.3f}, bootstrap ARI={stability:.2f})")
    print(f"          silhouette curve: " + ", ".join(f"k{k}:{sil_curve[k]:.2f}" for k in sorted(sil_curve)))
    for t in types:
        print(f"\n  Type {t['type']} (n={t['n']}): {'  '.join(t['signature']) or 'near-average'}")
        b5 = ", ".join(f"{k[:4]}{v:+.1f}" for k, v in sorted(t['bigfive'].items(), key=lambda x: -x[1])[:3])
        print(f"     Big5: {b5}")
        egs = [o.split('.')[0].replace("I'm ", "").replace("I am ", "")[:46] for o in t['example_occupations'][:4]]
        print("     e.g.: " + " | ".join(egs))

    # example report cards
    print(f"\n=== EXAMPLE REPORT CARDS ===")
    picks = [personas.index(p) for p in personas[:1]] + list(RNG.choice(N, 2, replace=False))
    for pi in picks:
        print(f"\n  « {personas[pi][:70]} »  →  Type {labels[pi]}")
        for f in range(K):
            bar = "█" * deciles[pi, f] + "·" * (10 - deciles[pi, f])
            print(f"     {METATRAITS[f]:11s} {bar} {deciles[pi,f]}/10  ±{band_dec[f]:.0f}")

    # ---------- figures ----------
    # fig 1: confound — g loadings + metatrait stability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    order_g = np.argsort(vg)
    cols = ["#d62728" if des[i] < 0 else "#2ca02c" if des[i] > 0 else "#888" for i in order_g]
    ax1.barh(range(P), vg[order_g], color=cols)
    ax1.set_yticks(range(P)); ax1.set_yticklabels([facets[i] for i in order_g], fontsize=6)
    ax1.set_title(f"General factor g ({g_var*100:.0f}% var)\nr(g, desirability)={g_des_r:+.2f}  "
                  f"(green=desirable, red=undesirable)")
    ax1.axvline(0, color="k", lw=0.5)
    ax2.bar(range(K), metatrait_congruence, color=["#1f77b4" if f in CORE else "#bbb" for f in range(K)])
    ax2.set_xticks(range(K)); ax2.set_xticklabels(METATRAITS, rotation=30, ha="right", fontsize=8)
    ax2.axhline(0.9, color="green", ls="--", lw=1, label="φ=0.90")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("congruence raw→purified")
    ax2.set_title("Metatrait stability to removing g\n(how confound-contaminated each axis is)")
    ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"tool_confound{TAG}.png"), dpi=130); plt.close(fig)

    # fig 2: typology scatter (Plasticity × Drive), colored by type
    fig, ax = plt.subplots(figsize=(9, 7.5))
    palette = plt.cm.tab10(np.linspace(0, 1, best_k))
    for c in range(best_k):
        mask = labels == c
        ax.scatter(M[mask, 0], M[mask, 1], color=palette[c], s=40, alpha=0.75,
                   label=f"T{c} (n={mask.sum()})")
        cx, cy = M[mask, 0].mean(), M[mask, 1].mean()
        sigtxt = " ".join(s.split(" ")[0] for s in next(t["signature"] for t in types if t["type"] == c)[:2])
        ax.annotate(sigtxt or f"T{c}", (cx, cy), fontsize=9, fontweight="bold",
                    ha="center", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("Plasticity (z)"); ax.set_ylabel("Drive (z)")
    ax.set_title(f"Personality types — {best_k} clusters in metatrait space\n"
                 f"silhouette={best_sil:.2f}, bootstrap stability ARI={stability:.2f}")
    ax.legend(fontsize=7, loc="best"); ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"tool_types{TAG}.png"), dpi=130); plt.close(fig)

    # fig 3: type × metatrait centroid heatmap
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * best_k + 2))
    cent_mat = np.array([[types[r]["centroid"][m] for m in METATRAITS] for r in range(best_k)])
    im = ax.imshow(cent_mat, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(K)); ax.set_xticklabels(METATRAITS, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(best_k)); ax.set_yticklabels([f"T{types[r]['type']} (n={types[r]['n']})" for r in range(best_k)])
    for r in range(best_k):
        for f in range(K):
            ax.text(f, r, f"{cent_mat[r,f]:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(cent_mat[r, f]) > 0.8 else "black")
    ax.set_title("Type centroids in metatrait space (z)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"tool_type_centroids{TAG}.png"), dpi=130); plt.close(fig)

    out = {
        "stage": "human_tool", "N": N, "metatraits": METATRAITS, "core_axes": CORE,
        "norms": {"mu": mu.tolist(), "sd": sd.tolist(), "facet_order": facets},
        "g_factor": {"var_share": g_var, "loadings": vg.tolist(), "desirability_r": g_des_r,
                     "top_facets": g_top},
        "metatrait_congruence_raw_to_purified": metatrait_congruence,
        "reliability": rel.tolist(), "se_z": se_z.tolist(), "band_decile": band_dec.tolist(),
        "loadings_purified": L_pure.tolist(),
        "typology": {"k": best_k, "silhouette": best_sil, "stability_ari": stability,
                     "silhouette_curve": sil_curve, "types": types, "labels": labels.tolist()},
        "scores_metatrait_z": M.tolist(), "deciles": deciles.tolist(), "personas": personas,
    }
    json.dump(out, open(os.path.join(OUT, f"human_tool{TAG}.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 20, "stage": "human_tool", "N": N,
                             "g_var": round(g_var, 3), "g_desirability_r": round(g_des_r, 2),
                             "k_types": best_k, "silhouette": round(best_sil, 3),
                             "stability_ari": round(stability, 2),
                             "mean_rel_core": round(float(rel[CORE].mean()), 3)}) + "\n")
    print("\nsaved -> human_tool.json + 3 figures")


if __name__ == "__main__":
    main()
