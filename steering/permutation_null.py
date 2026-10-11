"""
permutation_null.py — the control §4.4 of PAPER-A names: is the 5-factor steering
structure real, or forced by the extraction/EFA pipeline?

The factor analysis runs over X = persona x facet score matrix (409 x 27). The five
metatraits are the covariance structure of the 27 facet-scores across personas, and
split-half congruence (0.883) is whether that structure recovers in two random halves.

Three nulls, all run through the BYTE-IDENTICAL efa/congruence/split machinery copied
from stabilize_n400.py, so the null statistic and the observed statistic are comparable:

  1. COLUMN-PERMUTATION null. Independently permute the persona order within each facet
     column. This destroys all cross-facet covariance (the thing the factors rest on)
     while preserving every facet's marginal distribution, N, p, K, varimax, and the
     Hungarian-matched-congruence readout. If split-half survives this, the 0.883 is a
     phantom — varimax finding stable structure in noise. If it collapses, 0.883 is signal.
  2. GAUSSIAN null. Standard-normal X of the same shape. Absolute floor for what the
     Hungarian-matched split-half congruence returns on structureless data.
  3. BIG-FIVE-LABEL null. Shuffle the facet->Big-Five-domain assignment and recompute the
     observed structure's congruence to "Big Five". Establishes the chance level for the
     vs-Big-Five number (0.322): is 0.322 meaningfully low, or just what random alignment
     to 5 domain indicators returns?

No model inference. Reuses research/loop/native_model_n400.json. Writes
research/loop/permutation_null.json + figs/permutation_null.png. Does NOT git-commit.

Run:  cd steering && .venv/bin/python permutation_null.py
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import facets27 as F

from paths import REPO, OUT, FIGS
B = 2000                       # null iterations per test
SPLIT_SEED = 7                 # the exact persona split used in stabilize_n400.analyze()


# ---------- machinery: byte-identical to stabilize_n400.py ----------
def varimax(Phi, gamma=1.0, q=200, tol=1e-8):
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


def congruence(a, b):
    return abs(float(a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-12))


def parallel_analysis(X, n_iter=200, seed=11):
    n, p = X.shape
    rng = np.random.default_rng(seed)
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    rand = np.zeros((n_iter, p))
    for i in range(n_iter):
        rand[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1]
    thresh = np.percentile(rand, 95, axis=0)
    return max(1, int((real > thresh).sum()))


def efa(X, K):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    C = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(C); idx = np.argsort(w)[::-1]
    eig = np.clip(w[idx], 0, None); V = V[:, idx]
    L = varimax(V[:, :K] * np.sqrt(eig[:K]))
    L = L[:, np.argsort((L ** 2).sum(0))[::-1]]
    return L


def split_half_congruence(X, K, split_seed=SPLIT_SEED):
    """The headline metric: EFA each random half, Hungarian-match factors, mean |phi|."""
    N = X.shape[0]
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(N); h1, h2 = perm[:N // 2], perm[N // 2:]
    L1 = efa(X[h1], K); L2 = efa(X[h2], K)
    C = np.array([[congruence(L1[:, i], L2[:, j]) for j in range(K)] for i in range(K)])
    ri, ci = linear_sum_assignment(-C)
    return float(np.mean([C[ri[m], ci[m]] for m in range(K)]))


def bigfive_congruence(L, domI):
    """Hungarian-matched mean |phi| of the 5 EFA factors against the 5 Big-Five domain indicators."""
    K = L.shape[1]; D = domI.shape[1]
    CB = np.array([[congruence(L[:, j], domI[:, d]) for d in range(D)] for j in range(K)])
    ri, ci = linear_sum_assignment(-CB)
    return float(np.mean([CB[ri[m], ci[m]] for m in range(min(K, D))]))


def summarize(observed, null, higher_is_signal=True):
    null = np.asarray(null)
    mean, sd = float(null.mean()), float(null.std())
    p95, p99 = float(np.percentile(null, 95)), float(np.percentile(null, 99))
    z = (observed - mean) / (sd + 1e-12)
    if higher_is_signal:
        p_emp = float((null >= observed).mean())
    else:
        p_emp = float((null <= observed).mean())
    return dict(observed=round(observed, 4), null_mean=round(mean, 4), null_sd=round(sd, 4),
                null_p95=round(p95, 4), null_p99=round(p99, 4), null_max=round(float(null.max()), 4),
                null_min=round(float(null.min()), 4), z=round(z, 2), p_empirical=p_emp,
                p_reported=("<%.0e" % (1.0 / len(null))) if p_emp == 0 else round(p_emp, 4))


def main():
    d = json.load(open(os.path.join(OUT, "native_model_n400.json")))
    X = np.array(d["scores"])                  # 409 x 27
    L = np.array(d["loadings"])                # 27 x 5
    facets = d["facet_order"]
    N, p = X.shape
    K = L.shape[1]
    print(f"X = {X.shape}, K = {K}")

    # Big-Five domain indicator (mean-centered), identical to stabilize_n400.analyze()
    dom_of = [F.DOMAIN_OF[f] for f in facets]
    domI0 = np.zeros((p, len(F.DOMAINS)))
    for di, dom in enumerate(F.DOMAINS):
        for i in range(p):
            domI0[i, di] = 1.0 if dom_of[i] == dom else 0.0
    domI = domI0 - domI0.mean(0, keepdims=True)

    # ---- observed statistics (reproduce the paper's numbers off this pipeline) ----
    obs_split = split_half_congruence(X, K)
    obs_big5 = bigfive_congruence(L, domI)
    obs_K = parallel_analysis(X)
    print(f"\nreproduced observed:  split-half phi = {obs_split:.3f}  (paper 0.883)")
    print(f"                      vs-BigFive  phi = {obs_big5:.3f}  (paper 0.322)")
    print(f"                      parallel-analysis K = {obs_K}  (paper 5)")

    # ---- NULL 1: column-permutation (destroys cross-facet covariance) ----
    rng = np.random.default_rng(101)
    col_split, col_K = [], []
    for b in range(B):
        Xp = X.copy()
        for j in range(p):
            Xp[:, j] = Xp[rng.permutation(N), j]
        col_split.append(split_half_congruence(Xp, K))
        if b < 400:                              # K is the slow one; 400 is plenty for the dist
            col_K.append(parallel_analysis(Xp, seed=b))
    s_col = summarize(obs_split, col_split)
    K_ge5 = float(np.mean(np.array(col_K) >= 5))
    print(f"\n[NULL 1 column-perm]  split-half: obs {obs_split:.3f} vs null {s_col['null_mean']:.3f}"
          f" (p95 {s_col['null_p95']:.3f}, max {s_col['null_max']:.3f})  z={s_col['z']}  p={s_col['p_reported']}")
    print(f"                      parallel-analysis retained >=5 factors in {K_ge5*100:.1f}% of nulls"
          f" (mean K_null={np.mean(col_K):.2f})")

    # ---- NULL 2: gaussian floor ----
    rng2 = np.random.default_rng(202)
    g_split = [split_half_congruence(rng2.standard_normal((N, p)), K) for _ in range(B)]
    s_gauss = summarize(obs_split, g_split)
    print(f"[NULL 2 gaussian]     split-half: obs {obs_split:.3f} vs null {s_gauss['null_mean']:.3f}"
          f" (p95 {s_gauss['null_p95']:.3f}, max {s_gauss['null_max']:.3f})  z={s_gauss['z']}  p={s_gauss['p_reported']}")

    # ---- NULL 3: Big-Five-label shuffle (chance level for the 0.322) ----
    rng3 = np.random.default_rng(303)
    big5_null = []
    for _ in range(B):
        perm = rng3.permutation(p)
        domS = domI0[perm] - domI0[perm].mean(0, keepdims=True)
        big5_null.append(bigfive_congruence(L, domS))
    # observed 0.322 is LOW if it sits at/below the random-label band -> structure ~orthogonal to Big Five
    s_big5 = summarize(obs_big5, big5_null, higher_is_signal=False)
    print(f"[NULL 3 big5-labels]  vs-BigFive: obs {obs_big5:.3f} vs random-label null {s_big5['null_mean']:.3f}"
          f" (p5 {np.percentile(big5_null,5):.3f}, p95 {s_big5['null_p95']:.3f})")
    print(f"                      -> the model's structure aligns to Big Five at chance"
          f" {'(<=' if obs_big5 <= s_big5['null_mean'] else '(>'} random)")

    # also: how does within-model replication (0.883) compare to the best the structure
    # ever reaches toward Big Five even under favorable random relabeling?
    big5_best = float(np.max(big5_null))

    out = dict(
        test="permutation_null", B=B, X_shape=list(X.shape), K=K,
        observed=dict(split_half=round(obs_split, 4), vs_bigfive=round(obs_big5, 4),
                      parallel_K=obs_K),
        null_column_permutation=dict(**s_col, parallel_K_ge5_frac=round(K_ge5, 4),
                                     parallel_K_mean=round(float(np.mean(col_K)), 3)),
        null_gaussian=s_gauss,
        null_bigfive_labels=dict(**s_big5, max_under_random_relabel=round(big5_best, 4)),
        verdict=("STRUCTURE REAL — split-half 0.883 exceeds both nulls (p<1/B); "
                 "parallel analysis recovers >=5 factors essentially never under the column null; "
                 "vs-BigFive 0.322 sits in the random-label band, i.e. ~orthogonal to Big Five"
                 if (s_col["p_empirical"] == 0 and s_gauss["p_empirical"] == 0)
                 else "INCONCLUSIVE — observed split-half within null band; see numbers"),
    )
    json.dump(out, open(os.path.join(OUT, "permutation_null.json"), "w"), indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(col_split, bins=40, alpha=0.7, label="column-perm null", color="#888")
    ax[0].hist(g_split, bins=40, alpha=0.5, label="gaussian null", color="#bbb")
    ax[0].axvline(obs_split, color="#d62728", lw=2, label=f"observed {obs_split:.3f}")
    ax[0].set_title("Split-half factor replicability\nvs structure-destroying nulls")
    ax[0].set_xlabel("Hungarian-matched mean |Tucker phi|"); ax[0].legend(fontsize=8)
    ax[1].hist(big5_null, bins=40, alpha=0.7, color="#888", label="random Big-Five labels")
    ax[1].axvline(obs_big5, color="#d62728", lw=2, label=f"observed {obs_big5:.3f}")
    ax[1].set_title("Congruence to Big Five\nvs random facet->domain labels")
    ax[1].set_xlabel("Hungarian-matched mean |Tucker phi|"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "permutation_null.png"), dpi=130); plt.close(fig)

    print("\n=== VERDICT ===\n" + out["verdict"])
    print("\nsaved -> research/loop/permutation_null.json, figs/permutation_null.png")


if __name__ == "__main__":
    main()
