"""
select_oblique.py — iter-11 (D16): greedy item SELECTION on the pooled oblique bank.

The bank is over-saturated with the general factor (iter-9: GF 38%), so a well-chosen SUBSET whose
loading profiles are COMPLEMENTARY can recover the facet space better than the whole pool. This does
D-optimal greedy forward selection on the pooled calibration loading matrix S (n_items × K):

    pick the item that maximally increases  log det(SᵀS + εI)   (D-optimality / info volume)

until a budget is reached. Reports cond(S) and min-singular-value for selected vs full vs random.
If a live multi-persona response matrix is available (oblique_inversion_pool.json A_multi), it ALSO
runs the actual inversion on the selected subset OFFLINE (no model) and reports recovery — the real
payoff. Pure numpy; no model load.

Output → research/loop/oblique_selection.json.
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.stats import pearsonr
import facets27 as F
import oblique_items as OI

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
DOSE_KEYS = None  # inferred


def load_pooled_delta():
    cals = [json.load(open(os.path.join(OUT, "oblique_calibration.json"))),
            json.load(open(os.path.join(OUT, "oblique_calibration_b2.json")))]
    delta = {}
    for cal in cals:
        for facet, by_dose in cal["delta"].items():
            delta.setdefault(facet, {})
            for k, row in by_dose.items():
                delta[facet].setdefault(k, {}).update(row)
    return delta


def fit_S(delta, item_ids, usable):
    rows, ys = [], {i: [] for i in item_ids}
    for f in usable:
        for k, row in delta[f].items():
            rows.append([1.0] + [float(k) if g == f else 0.0 for g in usable])
            for i in item_ids:
                ys[i].append(row[i])
    X = np.array(rows)
    S = np.zeros((len(item_ids), len(usable))); b = np.zeros(len(item_ids))
    for j, i in enumerate(item_ids):
        c, *_ = np.linalg.lstsq(X, np.array(ys[i]), rcond=None)
        b[j] = c[0]; S[j] = c[1:]
    return S, b


def greedy_dopt(S, budget, eps=1e-3):
    """Greedy forward D-optimal selection over rows of S (items)."""
    n, K = S.shape
    chosen, M = [], eps * np.eye(K)
    avail = set(range(n))
    while len(chosen) < budget and avail:
        best, best_gain = None, -np.inf
        sign, logdet = np.linalg.slogdet(M)
        for i in avail:
            s = S[i:i+1]
            sign2, ld2 = np.linalg.slogdet(M + s.T @ s)
            if ld2 - logdet > best_gain:
                best_gain, best = ld2 - logdet, i
        chosen.append(best); avail.discard(best)
        M = M + S[best:best+1].T @ S[best:best+1]
    return chosen


def greedy_eopt(S, budget, eps=1e-3):
    """Greedy forward E-optimal selection: maximize the smallest singular value of S_subset
    (directly targets invertibility / conditioning, unlike D-optimal volume)."""
    n, K = S.shape
    chosen, M = [], eps * np.eye(K)
    avail = set(range(n))
    while len(chosen) < budget and avail:
        best, best_min = None, -np.inf
        for i in avail:
            s = S[i:i+1]
            mineig = float(np.linalg.eigvalsh(M + s.T @ s)[0])
            if mineig > best_min:
                best_min, best = mineig, i
        chosen.append(best); avail.discard(best)
        M = M + S[best:best+1].T @ S[best:best+1]
    return chosen


def cond_stats(S):
    s = np.linalg.svd(S, compute_uv=False)
    return float(s[0] / s[-1]), float(s[-1])


def recovery(delta, A_full, item_order_full, usable, subset_ids):
    """Offline inversion on a subset, reusing the live multi-persona responses A_full."""
    idx = [item_order_full.index(i) for i in subset_ids]
    A = A_full[:, idx]
    S, b = fit_S(delta, subset_ids, usable)
    That = np.array([np.linalg.lstsq(S, A[r] - b, rcond=None)[0] for r in range(A.shape[0])])
    return S, That


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=30)
    args = ap.parse_args()

    via = json.load(open(os.path.join(OUT, "viability27_best.json")))["facets"]
    usable = [f for f in F.FACET_ORDER if via[f]["verdict"] in ("strong", "weak")]
    K = len(usable)
    delta = load_pooled_delta()
    item_ids = [it["id"] for it in OI.POOL]
    S_full, b_full = fit_S(delta, item_ids, usable)
    cond_full, mineig_full = cond_stats(S_full)

    sel = greedy_dopt(S_full, args.budget)
    sel_ids = [item_ids[i] for i in sel]
    S_sel = S_full[sel]
    cond_sel, mineig_sel = cond_stats(S_sel)

    selE = greedy_eopt(S_full, args.budget)
    selE_ids = [item_ids[i] for i in selE]
    cond_selE, mineig_selE = cond_stats(S_full[selE])
    print(f"  E-optimal subset: cond={cond_selE:.1f}  σ_min={mineig_selE:.3f}")

    rng = np.random.default_rng(7)
    rand_conds = []
    for _ in range(20):
        r = rng.choice(len(item_ids), args.budget, replace=False)
        rand_conds.append(cond_stats(S_full[r])[0])
    cond_rand = float(np.mean(rand_conds))

    print(f"=== OBLIQUE SELECTION (D-optimal, budget {args.budget} of {len(item_ids)}) ===")
    print(f"  cond(S):   full({len(item_ids)})={cond_full:.1f}   selected={cond_sel:.1f}   "
          f"random-mean={cond_rand:.1f}")
    print(f"  σ_min(S):  full={mineig_full:.3f}   selected={mineig_sel:.3f}")
    print(f"  selected items: {sel_ids}")

    out = {"stage": "oblique_selection", "budget": args.budget, "K": K, "usable": usable,
           "selected": sel_ids, "cond_full": round(cond_full, 1), "cond_selected": round(cond_sel, 1),
           "cond_random_mean": round(cond_rand, 1),
           "sigma_min_full": round(mineig_full, 4), "sigma_min_selected": round(mineig_sel, 4)}

    # ---- offline recovery, if the pooled live responses are saved ----
    pool_inv = os.path.join(OUT, "oblique_inversion_pool.json")
    if os.path.exists(pool_inv):
        pj = json.load(open(pool_inv))
        if "A_multi" in pj:
            A_full = np.array(pj["A_multi"]); item_order_full = pj["item_order"]
            Theta = np.array(pj["theta_true"])

            def report(name, ids):
                _, That = recovery(delta, A_full, item_order_full, usable, ids)
                rc = [pearsonr(Theta[:, fi], That[:, fi])[0] for fi in range(K)
                      if np.ptp(Theta[:, fi]) > 0 and np.ptp(That[:, fi]) > 0]
                Cimp = np.corrcoef(Theta.T); Crec = np.corrcoef(That.T)
                iu = np.triu_indices(K, 1)
                cs = pearsonr(Cimp[iu], Crec[iu])[0]
                print(f"  [{name:9}] mean recovery={np.mean(rc):+.3f}  ≥0.70: "
                      f"{sum(v>=0.7 for v in rc)}/{K}  corr-struct={cs:.3f}")
                return {"mean_recovery": round(float(np.mean(rc)), 3),
                        "n70": int(sum(v >= 0.7 for v in rc)), "corr_struct": round(float(cs), 3)}

            print("\n[OFFLINE RECOVERY]  same live responses, different item subsets:")
            out["recovery_full"] = report("full75", item_order_full)
            out["recovery_dopt"] = report(f"Dopt{args.budget}", sel_ids)
            out["recovery_eopt"] = report(f"Eopt{args.budget}", selE_ids)
            rng2 = np.random.default_rng(3)
            rid = [item_ids[i] for i in rng2.choice(len(item_ids), args.budget, replace=False)]
            out["recovery_random"] = report(f"rand{args.budget}", rid)

    json.dump(out, open(os.path.join(OUT, "oblique_selection.json"), "w"), indent=2)
    print(f"\nsaved → {os.path.join(OUT, 'oblique_selection.json')}")


if __name__ == "__main__":
    main()
