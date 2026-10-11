"""
manifold_geometry.py — iter-14: the SHAPE of the trait manifold in activation space.

The whole program kept hitting walls that turned out to be geometric (positive manifold, the
two gates, the readout-invariant resistant facets). Instead of probing the geometry through
readouts, look at it directly: the 27 steering vectors live in R^4096 at every layer. Their
sign-corrected cosine (Gram) matrix IS the shape of the trait manifold.

What this measures, per layer (vectors exist at all 31 layers; at a fixed layer they share one
activation basis so cosines are comparable):
  * Positive manifold        — fraction of off-diagonal cosines > 0, mean off-diagonal cosine.
  * General-factor saturation — top eigenvalue share of the Gram (lambda_1 / sum lambda).
  * Effective dimensionality  — participation ratio (sum l)^2 / sum l^2 and effective rank.
  * Big-Five recovery         — agglomerative clustering (cut at k=5) vs the a-priori DOMAINS
                                labeling: do the vectors self-organize into the Big Five?
  * Separability map          — each facet's nearest neighbor by cosine. A facet that is nearly
                                collinear with a neighbor is NOT separably represented — this is
                                the iter-8 wall, read straight off the geometry. Sociability /
                                Irritability are the de-confounding targets.

Sign convention: each pca_diff vector is oriented by the viability sign so it points "more of the
trait". With that, a POSITIVE cosine means two traits are aligned in activation space (a genuine
positive manifold), not a sign artifact.

Figures (research/loop/figs/): cosine heatmap + dendrogram, 3D PCA embedding colored by domain,
eigenspectrum & GF-share vs layer, effective-rank & positive-manifold vs layer.
Numbers -> research/loop/manifold_geometry.json.
"""
from __future__ import annotations
import os, json
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from repeng import ControlVector
from extract_vectors import VEC_DIR
import facets27 as F

from paths import OUT, FIGS
os.makedirs(FIGS, exist_ok=True)
N_LAYERS = 31
HEADLINE_LAYER = None  # chosen automatically = layer with richest Big-Five recovery

DOMAIN_COLORS = {
    "Agreeableness": "#2ca02c", "Conscientiousness": "#1f77b4", "Neuroticism": "#d62728",
    "Extraversion": "#ff7f0e", "Openness": "#9467bd",
}


def load_signed_dirs():
    """Return {facet: {layer: unit vector oriented 'more of trait'}} for the 27 facets."""
    via = json.load(open(os.path.join(OUT, "viability27_best.json")))["facets"]
    dirs = {}
    for f in F.FACET_ORDER:
        cv = ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf"))
        sign = float(via[f]["sign"])
        dirs[f] = {L: (sign * cv.directions[L]).astype(np.float64) for L in cv.directions}
    return dirs, via


def layer_matrix(dirs, layer):
    """27 x 4096 sign-corrected unit-direction matrix at one layer (rows = FACET_ORDER)."""
    M = np.stack([dirs[f][layer] for f in F.FACET_ORDER])
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return M


def cluster_agreement(labels_pred, facets):
    """Best-match accuracy of a k=5 clustering against the a-priori Big Five domains.
    For each predicted cluster, assign it the domain of its plurality; accuracy = fraction of
    facets whose predicted-cluster plurality domain equals their true domain."""
    true = [F.DOMAIN_OF[f] for f in facets]
    clusters = defaultdict(list)
    for i, c in enumerate(labels_pred):
        clusters[c].append(i)
    correct = 0
    cluster_to_domain = {}
    for c, idx in clusters.items():
        doms = [true[i] for i in idx]
        plurality = max(set(doms), key=doms.count)
        cluster_to_domain[c] = plurality
        correct += sum(1 for i in idx if true[i] == plurality)
    return correct / len(facets), cluster_to_domain


def analyse_layer(M):
    """Geometry scalars for one 27x4096 sign-corrected matrix."""
    C = M @ M.T                         # 27x27 cosine (Gram of unit rows)
    n = C.shape[0]
    off = C[~np.eye(n, dtype=bool)]
    eig = np.linalg.eigvalsh(C)[::-1]   # descending
    eig = np.clip(eig, 0, None)
    s = eig.sum()
    gf_share = float(eig[0] / s) if s else 0.0
    part_ratio = float(s * s / (eig * eig).sum()) if (eig * eig).sum() else 0.0
    # effective rank = exp(spectral entropy)
    p = eig / s if s else eig
    p = p[p > 0]
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    # k=5 agglomerative clustering on cosine distance
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=5, criterion="maxclust")
    acc, c2d = cluster_agreement(lab, F.FACET_ORDER)
    return {
        "C": C, "eig": eig, "Z": Z, "labels": lab,
        "gf_share": gf_share, "part_ratio": part_ratio, "eff_rank": eff_rank,
        "mean_offdiag": float(off.mean()), "frac_pos_offdiag": float((off > 0).mean()),
        "max_offdiag": float(off.max()), "bigfive_acc": acc, "cluster_to_domain": c2d,
    }


def neighbor_map(C):
    """For each facet, its most-collinear other facet and that cosine (the separability map)."""
    out = {}
    for i, f in enumerate(F.FACET_ORDER):
        row = C[i].copy(); row[i] = -np.inf
        j = int(np.argmax(row))
        out[f] = {"neighbor": F.FACET_ORDER[j], "cos": round(float(C[i, j]), 3)}
    return out


# ---------------------------------------------------------------------------- figures
def fig_heatmap(C, Z, layer, path):
    order = dendrogram(Z, no_plot=True)["leaves"]
    facets = [F.FACET_ORDER[i] for i in order]
    Cz = C[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(Cz, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(27)); ax.set_yticks(range(27))
    ax.set_xticklabels(facets, rotation=90, fontsize=7)
    ax.set_yticklabels(facets, fontsize=7)
    for i, f in enumerate(facets):
        ax.get_yticklabels()[i].set_color(DOMAIN_COLORS[F.DOMAIN_OF[f]])
        ax.get_xticklabels()[i].set_color(DOMAIN_COLORS[F.DOMAIN_OF[f]])
    ax.set_title(f"Sign-corrected cosine geometry of 27 trait vectors (layer {layer}, "
                 f"clustered)\ncolor = a-priori Big Five domain", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7, label="cosine similarity")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_3d(M, layer, path):
    U, S, Vt = np.linalg.svd(M - M.mean(0, keepdims=True), full_matrices=False)
    coords = (M - M.mean(0, keepdims=True)) @ Vt[:3].T
    var = (S ** 2 / (S ** 2).sum())[:3]
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    for i, f in enumerate(F.FACET_ORDER):
        c = DOMAIN_COLORS[F.DOMAIN_OF[f]]
        ax.scatter(*coords[i], color=c, s=45)
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], "  " + f, fontsize=6.5, color=c)
    for d, c in DOMAIN_COLORS.items():
        ax.scatter([], [], color=c, label=d)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
    ax.set_zlabel(f"PC3 ({var[2]*100:.0f}%)")
    ax.set_title(f"Trait manifold — top-3 PCA of sign-corrected vectors (layer {layer})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return var.tolist(), coords


def fig_spectrum(per_layer, headline, path):
    layers = sorted(per_layer)
    gf = [per_layer[L]["gf_share"] for L in layers]
    er = [per_layer[L]["eff_rank"] for L in layers]
    acc = [per_layer[L]["bigfive_acc"] for L in layers]
    mo = [per_layer[L]["mean_offdiag"] for L in layers]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(layers, gf, "-o", ms=3, label="general-factor share (λ₁/Σλ)")
    ax.plot(layers, mo, "-s", ms=3, label="mean off-diag cosine")
    ax.plot(layers, acc, "-^", ms=3, label="Big Five recovery (k=5 acc)")
    ax.axvline(headline, color="gray", ls="--", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("value"); ax.legend(fontsize=9)
    ax.set_title("Manifold structure across layers")
    ax2 = axes[1]
    ax2.plot(layers, er, "-o", ms=3, color="purple", label="effective rank")
    ax2.axhline(27, color="gray", ls=":", lw=1, label="27 (isotropic)")
    ax2.axvline(headline, color="gray", ls="--", lw=1)
    ax2.set_xlabel("layer"); ax2.set_ylabel("effective rank"); ax2.legend(fontsize=9)
    ax2.set_title("Effective dimensionality across layers")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    dirs, via = load_signed_dirs()
    per_layer = {}
    for L in range(1, N_LAYERS + 1):
        M = layer_matrix(dirs, L)
        per_layer[L] = analyse_layer(M)

    # headline layer = best Big-Five recovery, tie-broken by richest effective rank, among mid/late
    cand = [L for L in range(10, N_LAYERS + 1)]
    headline = max(cand, key=lambda L: (per_layer[L]["bigfive_acc"], per_layer[L]["eff_rank"]))
    H = per_layer[headline]
    M_h = layer_matrix(dirs, headline)

    # figures
    fig_heatmap(H["C"], H["Z"], headline, os.path.join(FIGS, "manifold_cosine_heatmap.png"))
    var3, coords = fig_3d(M_h, headline, os.path.join(FIGS, "manifold_3d.png"))
    fig_spectrum(per_layer, headline, os.path.join(FIGS, "manifold_spectrum.png"))

    nbr = neighbor_map(H["C"])

    # ---- console report ----
    print(f"\n=== MANIFOLD GEOMETRY (27 trait vectors, headline layer {headline}) ===")
    print(f"  positive manifold : {H['frac_pos_offdiag']*100:.0f}% of off-diagonal cosines > 0, "
          f"mean {H['mean_offdiag']:+.3f}")
    print(f"  general factor     : λ₁ = {H['gf_share']*100:.0f}% of total variance")
    print(f"  effective rank     : {H['eff_rank']:.1f} of 27  (participation ratio "
          f"{H['part_ratio']:.1f})")
    print(f"  Big Five recovery  : {H['bigfive_acc']*100:.0f}% (k=5 clustering vs a-priori domains)")
    print(f"  3D PCA captures    : {sum(var3)*100:.0f}% of variance ({', '.join(f'{v*100:.0f}%' for v in var3)})")
    print(f"\n  cluster -> plurality domain: {H['cluster_to_domain']}")
    print(f"\n  separability map — most-collinear neighbor per facet:")
    for f in sorted(F.FACET_ORDER, key=lambda f: -nbr[f]["cos"]):
        flag = "  <-- resistant" if f in ("Sociability", "Irritability") else ""
        print(f"    {f:24} -> {nbr[f]['neighbor']:22} cos {nbr[f]['cos']:+.3f}{flag}")

    out = {
        "stage": "manifold_geometry", "headline_layer": headline,
        "headline": {k: H[k] for k in ("gf_share", "part_ratio", "eff_rank", "mean_offdiag",
                                       "frac_pos_offdiag", "max_offdiag", "bigfive_acc")},
        "pca3_var": var3,
        "cluster_to_domain": {str(k): v for k, v in H["cluster_to_domain"].items()},
        "neighbor_map": nbr,
        "per_layer": {str(L): {k: per_layer[L][k] for k in
                               ("gf_share", "part_ratio", "eff_rank", "mean_offdiag",
                                "frac_pos_offdiag", "bigfive_acc")} for L in per_layer},
        "coords_3d": {f: coords[i].tolist() for i, f in enumerate(F.FACET_ORDER)},
        "domain_of": F.DOMAIN_OF,
    }
    json.dump(out, open(os.path.join(OUT, "manifold_geometry.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 14, "stage": "manifold_geometry",
                             "headline_layer": headline, "gf_share": round(H["gf_share"], 3),
                             "eff_rank": round(H["eff_rank"], 1),
                             "bigfive_acc": round(H["bigfive_acc"], 3),
                             "frac_pos_offdiag": round(H["frac_pos_offdiag"], 3)}) + "\n")
    print(f"\nsaved -> manifold_geometry.json + 3 figures in figs/")


if __name__ == "__main__":
    main()
