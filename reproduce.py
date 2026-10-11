#!/usr/bin/env python3
"""Recompute the paper's headline numbers from the committed data and check them.

    python3 reproduce.py              # offline checks, no model weights, no GPU
    python3 reproduce.py --quick      # same checks, 200 null draws instead of 2000
    python3 reproduce.py --with-vectors   # also rebuild the factor structure from vectors/

Every number is recomputed by the paper's own functions in `steering/`, not by a
reimplementation, and compared against the value printed in the manuscript. The
exit status is 0 only if every check passes.

What this cannot check without more than a CPU is listed at the end of the run:
direction extraction needs the model weights, and the cross-instrument control
additionally needs the SPI-135 item bank, which is not redistributable here.
"""
from __future__ import annotations
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "steering"))

# claim -> (expected, tolerance, where the manuscript states it)
EXPECTED = {
    "N (personas)":                 (409,    0,      "Table 1"),
    "facets":                       (27,     0,      "Table 1"),
    "split-half congruence":        (0.883,  0.003,  "§4.2"),
    "congruence to the Big Five":   (0.322,  0.003,  "§4.3"),
    "parallel-analysis K":          (5,      0,      "§4.1"),
    "column null: mean":            (0.277,  0.010,  "§4.4"),
    "column null: sd":              (0.041,  0.008,  "§4.4"),
    "column null: z":               (14.6,   0.6,    "§4.4"),
    "column null: P(K>=5)":         (0.03,   0.03,   "§4.4"),
    "label null: mean":             (0.258,  0.010,  "§4.3"),
    "label null: p-value":          (0.923,  0.060,  "§4.3"),
}


class Checks:
    def __init__(self):
        self.rows, self.failed = [], 0

    def add(self, claim, observed, expected=None, tol=None, note="", mc_scale=1.0):
        """mc_scale widens the tolerance for a statistic estimated from random draws,
        by the ratio of Monte Carlo standard errors, so fewer draws is not a failure."""
        if expected is None:
            expected, tol, note = EXPECTED[claim][0], EXPECTED[claim][1], EXPECTED[claim][2]
        tol = float(tol) * mc_scale
        ok = abs(float(observed) - float(expected)) <= float(tol)
        self.failed += (not ok)
        self.rows.append((claim, expected, observed, tol, note, ok))

    def report(self):
        w = max(len(r[0]) for r in self.rows)
        print(f"\n{'check'.ljust(w)}  {'paper':>8}  {'rerun':>8}  {'tol':>6}  where     result")
        print("-" * (w + 46))
        for claim, exp, obs, tol, note, ok in self.rows:
            e = f"{exp:.3f}" if isinstance(exp, float) else str(exp)
            o = f"{obs:.3f}" if isinstance(obs, float) else str(obs)
            t = f"{tol:.3f}" if isinstance(tol, float) else str(tol)
            print(f"{claim.ljust(w)}  {e:>8}  {o:>8}  {t:>6}  {note:<9} {'PASS' if ok else 'FAIL'}")
        print("-" * (w + 46))
        n = len(self.rows)
        print(f"{n - self.failed}/{n} checks passed")
        return self.failed == 0


def offline(checks, B, mc=1.0):
    """Recompute the structure statistics and both nulls from the committed scores."""
    import numpy as np
    import permutation_null as pn
    import facets27 as F

    src = os.path.join(pn.OUT, "native_model_n400.json")
    print(f"reading committed scores from {src}")
    d = json.load(open(src))
    X = np.array(d["scores"])
    L = np.array(d["loadings"])
    facets = d["facet_order"]
    N, p = X.shape
    K = L.shape[1]

    checks.add("N (personas)", N)
    checks.add("facets", p)

    dom_of = [F.DOMAIN_OF[f] for f in facets]
    domI0 = np.zeros((p, len(F.DOMAINS)))
    for di, dom in enumerate(F.DOMAINS):
        for i in range(p):
            domI0[i, di] = 1.0 if dom_of[i] == dom else 0.0
    domI = domI0 - domI0.mean(0, keepdims=True)

    t0 = time.time()
    checks.add("split-half congruence", pn.split_half_congruence(X, K))
    checks.add("congruence to the Big Five", pn.bigfive_congruence(L, domI))
    checks.add("parallel-analysis K", pn.parallel_analysis(X))
    print(f"observed statistics recomputed in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(101)
    col_split, col_K = [], []
    t0 = time.time()
    for b in range(B):
        Xp = X.copy()
        for j in range(p):
            Xp[:, j] = Xp[rng.permutation(N), j]
        col_split.append(pn.split_half_congruence(Xp, K))
        col_K.append(pn.parallel_analysis(Xp))
        if (b + 1) % max(1, B // 10) == 0:
            print(f"  column-permutation null {b + 1}/{B}", end="\r", flush=True)
    col_split = np.array(col_split)
    print(f"\ncolumn-permutation null, {B} draws, {time.time() - t0:.1f}s")
    obs = pn.split_half_congruence(X, K)
    checks.add("column null: mean", float(col_split.mean()), mc_scale=mc)
    checks.add("column null: sd", float(col_split.std(ddof=1)), mc_scale=mc)
    checks.add("column null: z", (obs - col_split.mean()) / col_split.std(ddof=1), mc_scale=mc)
    checks.add("column null: P(K>=5)", float((np.array(col_K) >= 5).mean()), mc_scale=mc)

    rng = np.random.default_rng(303)
    lab = []
    for _ in range(B):
        lab.append(pn.bigfive_congruence(L, domI[rng.permutation(p)]))
    lab = np.array(lab)
    obs_b5 = pn.bigfive_congruence(L, domI)
    # The manuscript reports this null with higher_is_signal=False, so its p is
    # P(null <= observed): the fraction of random relabelings that do no better.
    summary = pn.summarize(obs_b5, lab, higher_is_signal=False)
    checks.add("label null: mean", float(lab.mean()), mc_scale=mc)
    checks.add("label null: p-value", float(summary["p_empirical"]), mc_scale=mc)
    print(f"Big-Five label null, {B} draws: observed {obs_b5:.3f}, null mean "
          f"{lab.mean():.3f}, null max {lab.max():.3f}")
    print(f"  {float((lab >= obs_b5).mean()):.3f} of random relabelings reach or beat it, "
          f"so the alignment is not significantly better than chance labeling")


def with_vectors(checks):
    """Rebuild the 27-direction geometry from the committed .gguf vectors."""
    import numpy as np
    import paths
    missing = [f for f in os.listdir(paths.VEC_DIR) if f.endswith(".gguf")]
    print(f"\nvectors/: {len(missing)} .gguf directions in {paths.VEC_DIR}")
    checks.add("extracted directions", len(missing), 27, 0, "Table 1")
    try:
        from repeng import ControlVector
    except ImportError:
        print("  repeng is not installed, so the directions were counted but not loaded")
        print("  install it with: pip install -r requirements.txt")
        return
    import facets27 as F
    layer, rows = 13, []
    for f in F.FACETS:
        cv = ControlVector.import_gguf(os.path.join(paths.VEC_DIR, f"{f}.gguf"))
        rows.append(np.asarray(cv.directions[layer], dtype=np.float64))
    M = np.vstack(rows)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    C = M @ M.T
    off = C[~np.eye(len(F.FACETS), dtype=bool)]
    print(f"  loaded layer {layer} for {len(rows)} facets; mean |cosine| between "
          f"distinct directions = {np.abs(off).mean():.3f}")
    checks.add("directions loaded", len(rows), 27, 0, "Fig. 6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="200 null draws instead of 2000")
    ap.add_argument("--with-vectors", action="store_true",
                    help="also load the committed steering directions (needs repeng)")
    a = ap.parse_args()

    print(__doc__.splitlines()[0])
    try:
        import numpy, scipy  # noqa: F401
    except ImportError:
        sys.exit("numpy and scipy are required: pip install -r requirements.txt")

    checks = Checks()
    B = 200 if a.quick else 2000
    offline(checks, B, mc=(2000 / B) ** 0.5)
    if a.with_vectors:
        with_vectors(checks)

    ok = checks.report()
    print("\nnot checked here, and why:")
    print("  direction extraction (extract_vectors.py) needs the Llama-3.1-8B weights and a GPU")
    print("  cross-instrument convergence (convergent_validity.py) additionally needs the")
    print("    SPI-135 item bank, which is not redistributable in this repository")
    print("  cross-language replication needs Llama-3.1-Swallow-8B-v0.2")
    if a.quick:
        print("\n--quick used 200 draws instead of 2000, so the tolerance on every"
              "\nnull-derived row was widened by the ratio of Monte Carlo standard"
              "\nerrors. Rerun without --quick before reporting a mismatch.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
