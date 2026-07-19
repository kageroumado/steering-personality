"""
stabilize_n400.py — iter-22: stabilize the formal native model at N≈400.

iter-19 gave a moderate split-half factor replicability of 0.77 at N=200 — the 3-core
(Plasticity/Drive/Fluidity, α 0.81–0.87) is solid but the F4/F5 tail (Stability/Communion,
α 0.60/0.65) is under-powered. The fix is more personas, not a new method. This reuses the 200
already-scored personas in native_model.json (byte-identical procedure) and administers ~200 NEW
naturalistic personas from personas_extra200.json → N≈409, then re-runs the exact iter-19 EFA +
reliability and reports whether the tail stabilized.

Built to run AUTONOMOUSLY and RESUME-SAFE:
  * Per-persona checkpointing to n400_checkpoint.jsonl (append+flush). Re-running skips done personas,
    so a crash / sleep / kill loses at most one persona.
  * Periodic `git` checkpoint commits of the jsonl (COMMIT_EVERY), so progress is durable in history.
  * On completion writes the full native_model_n400.json AND a compact n400_result.json summary
    (read that, not the 1 MB model file) + appends metrics.jsonl + regenerates figs.

Run:  cd steering && .venv/bin/python stabilize_n400.py     (≈2–2.5 h on MPS; resumable)
Output -> research/loop/native_model_n400.json, n400_result.json, figs/native_model_n400_*.png
"""
from __future__ import annotations
import os, json, time, gc, subprocess
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
import facets27 as F

REPO = os.path.expanduser("~/Developer/personality-test-27d")
OUT = os.path.join(REPO, "research/loop")
FIGS = os.path.join(OUT, "figs")
CKPT = os.path.join(OUT, "n400_checkpoint.jsonl")
ITEMS = F.fc_items("best")
FACETS = F.FACET_ORDER
COMMIT_EVERY = 50           # git-commit the checkpoint every N newly-scored personas
PREFIX_SUFFIX = " This is me being honest about who I really am.\n\n"   # iter-15/19 byte-identical


# ---------- analysis (identical to formalize_native.py, kept local so this script is standalone) ----------
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


def parallel_analysis(X, n_iter=300, seed=11):
    n, p = X.shape
    rng = np.random.default_rng(seed)
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    rand = np.zeros((n_iter, p))
    for i in range(n_iter):
        rand[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1]
    thresh = np.percentile(rand, 95, axis=0)
    return max(1, int((real > thresh).sum())), real, thresh


def efa(X, K):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    C = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(C); idx = np.argsort(w)[::-1]
    eig = np.clip(w[idx], 0, None); V = V[:, idx]
    L = varimax(V[:, :K] * np.sqrt(eig[:K]))
    L = L[:, np.argsort((L ** 2).sum(0))[::-1]]
    return L, Z


def cronbach_alpha(items):
    m = items.shape[1]
    if m < 2:
        return float("nan")
    var_sum = items.var(0, ddof=1).sum()
    tot_var = items.sum(1).var(ddof=1)
    return float((m / (m - 1)) * (1 - var_sum / tot_var)) if tot_var > 0 else float("nan")


def mcdonald_omega(load):
    s = load.sum() ** 2
    err = (1 - load ** 2).sum()
    return float(s / (s + err)) if (s + err) > 0 else float("nan")


# ---------- resume-safe administration ----------
def git(*args):
    """Best-effort git in the repo; never fatal to the run."""
    try:
        subprocess.run(["git", "-C", REPO, *args], check=True,
                       capture_output=True, timeout=120)
        return True
    except Exception as e:
        print(f"  [git] skipped: {' '.join(args)} ({type(e).__name__})", flush=True)
        return False


def load_checkpoint():
    done = {}
    if os.path.exists(CKPT):
        for line in open(CKPT):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[int(r["idx"])] = r
    return done


def administer_new(new_personas):
    device = pick_device()
    done = load_checkpoint()
    todo = [i for i in range(len(new_personas)) if i not in done]
    print(f"resume: {len(done)}/{len(new_personas)} already scored, {len(todo)} to go", flush=True)
    if not todo:
        return done

    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()

    by_facet = defaultdict(list)
    for it in ITEMS:
        by_facet[it["facet"]].append(it)

    t0 = time.time(); n_new = 0
    with open(CKPT, "a") as ck:
        for k, pi in enumerate(todo):
            desc = new_personas[pi]
            scorer = PrefixScorer(model, tok, f"{desc}{PREFIX_SUFFIX}", n_options=2, device=device)
            row = np.zeros(len(FACETS))
            for fj, facet in enumerate(FACETS):
                vals = [scorer.score(it["stem"], [it["hi"], it["lo"]]) for it in by_facet[facet]]
                row[fj] = float(np.mean([s[0]["logprob_sum"] - s[1]["logprob_sum"] for s in vals]))
            del scorer
            rec = {"idx": pi, "persona": desc, "scores": row.tolist()}
            ck.write(json.dumps(rec) + "\n"); ck.flush()
            done[pi] = rec; n_new += 1
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            if n_new % 5 == 0:
                el = time.time() - t0; rate = el / n_new
                eta = rate * (len(todo) - n_new)
                print(f"  [{len(done)}/{len(new_personas)}] {el:.0f}s elapsed, "
                      f"{rate:.1f}s/persona, ETA {eta/60:.0f}min", flush=True)
            if n_new % COMMIT_EVERY == 0:
                git("add", CKPT)
                git("commit", "-m", f"research: n400 checkpoint {len(done)}/{len(new_personas)} personas")
    print(f"administered {n_new} new personas in {(time.time()-t0)/60:.0f}min", flush=True)
    return done


# ---------- full N≈400 model ----------
def analyze(X, personas):
    N = X.shape[0]
    nm200 = json.load(open(os.path.join(OUT, "native_model.json")))
    L200 = np.array(nm200["loadings"])                       # 27 x 5 (the iter-19 structure)

    K, real_eig, pa_thresh = parallel_analysis(X)
    Kuse = max(K, 5)
    print(f"\nN={N}: parallel analysis → K={K} (eigs>1: {(real_eig>1).sum()}); using K={Kuse}", flush=True)
    L, Z = efa(X, Kuse)

    factors = []
    for j in range(Kuse):
        col = L[:, j]; o = np.argsort(col)
        pos = [(FACETS[i], round(float(col[i]), 2)) for i in o[::-1] if col[i] > 0.35][:6]
        neg = [(FACETS[i], round(float(col[i]), 2)) for i in o if col[i] < -0.35][:6]
        idx = [i for i in range(27) if abs(col[i]) > 0.35]
        signed = Z[:, idx] * np.sign(col[idx])
        alpha = cronbach_alpha(signed) if len(idx) >= 2 else float("nan")
        omega = mcdonald_omega(np.abs(col[idx])) if len(idx) >= 2 else float("nan")
        factors.append({"factor": j + 1, "var_share": round(float((col ** 2).sum() / (L ** 2).sum()), 3),
                        "n_facets": len(idx), "alpha": round(alpha, 3), "omega": round(omega, 3),
                        "pos": pos, "neg": neg})

    # split-half factor replicability (the headline stabilization metric)
    rng = np.random.default_rng(7)
    perm = rng.permutation(N); h1, h2 = perm[:N // 2], perm[N // 2:]
    L1, _ = efa(X[h1], Kuse); L2, _ = efa(X[h2], Kuse)
    C = np.array([[congruence(L1[:, i], L2[:, j]) for j in range(Kuse)] for i in range(Kuse)])
    ri, ci = linear_sum_assignment(-C)
    split_half = float(np.mean([C[ri[m], ci[m]] for m in range(Kuse)]))

    # congruence vs Big Five
    domI = np.zeros((27, 5))
    for di, d in enumerate(F.DOMAINS):
        for i, f in enumerate(FACETS):
            domI[i, di] = 1.0 if F.DOMAIN_OF[f] == d else 0.0
    domI -= domI.mean(0, keepdims=True)
    CB = np.array([[congruence(L[:, j], domI[:, di]) for di in range(5)] for j in range(5)])
    ri2, ci2 = linear_sum_assignment(-CB)
    big5 = float(np.mean([CB[ri2[m], ci2[m]] for m in range(5)]))

    # did the structure HOLD vs the N=200 solution? (per-factor best-match congruence)
    CN = np.array([[congruence(L[:, j], L200[:, k]) for k in range(L200.shape[1])]
                   for j in range(Kuse)])
    ri3, ci3 = linear_sum_assignment(-CN)
    vs200_pairs = sorted([float(CN[ri3[m], ci3[m]]) for m in range(min(Kuse, L200.shape[1]))], reverse=True)
    vs200_mean = float(np.mean(vs200_pairs))

    return dict(N=N, K=K, Kuse=Kuse, factors=factors, split_half=split_half, big5=big5,
                vs200_mean=vs200_mean, vs200_pairs=vs200_pairs,
                L=L, real_eig=real_eig, pa_thresh=pa_thresh, nm200=nm200)


def make_figs(L, real_eig, pa_thresh, K, N, split_half, big5, factors):
    Kuse = L.shape[1]
    fig, ax = plt.subplots(figsize=(8.5, 9))
    im = ax.imshow(L, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(27)); ax.set_yticklabels(FACETS, fontsize=7)
    for i, f in enumerate(FACETS):
        ax.get_yticklabels()[i].set_color(
            {"Agreeableness": "#2ca02c", "Conscientiousness": "#1f77b4", "Neuroticism": "#d62728",
             "Extraversion": "#ff7f0e", "Openness": "#9467bd"}[F.DOMAIN_OF[f]])
    ax.set_xticks(range(Kuse))
    ax.set_xticklabels([f"F{j+1}\nα={factors[j]['alpha']:.2f}" for j in range(Kuse)], fontsize=8)
    ax.set_title(f"Native model N={N} (varimax)\nsplit-half φ={split_half:.2f}, vs Big Five φ={big5:.2f}")
    fig.colorbar(im, ax=ax, shrink=0.6, label="loading")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "native_model_n400_loadings.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(1, 28)
    ax.plot(xs, real_eig, "-o", ms=4, label="observed eigenvalues")
    ax.plot(xs, pa_thresh, "--", color="gray", label="parallel-analysis 95th pct")
    ax.axhline(1, color="lightgray", ls=":")
    ax.axvline(K, color="red", ls="--", lw=1, label=f"retain K={K}")
    ax.set_xlabel("factor"); ax.set_ylabel("eigenvalue"); ax.legend(fontsize=8)
    ax.set_title(f"Native model dimensionality (N={N})")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "native_model_n400_scree.png"), dpi=130); plt.close(fig)


def main():
    extra = json.load(open(os.path.join(OUT, "personas_extra200.json")))["personas"]
    done = administer_new(extra)

    nm200 = json.load(open(os.path.join(OUT, "native_model.json")))
    old_scores = np.array(nm200["scores"]); old_personas = nm200["personas"]
    new_idx = sorted(done)
    new_scores = np.array([done[i]["scores"] for i in new_idx])
    new_personas = [done[i]["persona"] for i in new_idx]
    X = np.vstack([old_scores, new_scores])
    personas = old_personas + new_personas

    R = analyze(X, personas)
    L = R["L"]
    mean_alpha = float(np.nanmean([f["alpha"] for f in R["factors"]]))
    mean_omega = float(np.nanmean([f["omega"] for f in R["factors"]]))
    core_alpha = [f["alpha"] for f in R["factors"][:3]]
    tail_alpha = [f["alpha"] for f in R["factors"][3:5]]

    print(f"\n=== NATIVE MODEL N={R['N']} (K={R['Kuse']}) ===", flush=True)
    for fa in R["factors"]:
        pos = ", ".join(f"{n}({v:+})" for n, v in fa["pos"])
        neg = ", ".join(f"{n}({v:+})" for n, v in fa["neg"])
        print(f"\n  F{fa['factor']} ({fa['var_share']*100:.0f}% | α={fa['alpha']:.2f} ω={fa['omega']:.2f} | {fa['n_facets']} facets):")
        print(f"     + {pos}")
        if neg:
            print(f"     − {neg}")
    print(f"\n  split-half φ = {R['split_half']:.3f}  (N=200 was 0.770)")
    print(f"  vs Big Five φ = {R['big5']:.3f}  (N=200 was 0.346)")
    print(f"  structure-held vs N=200 (per-factor best-match φ) = {R['vs200_mean']:.3f}  pairs={[round(x,2) for x in R['vs200_pairs']]}")
    print(f"  3-core α = {[round(a,2) for a in core_alpha]}   tail α (F4/F5) = {[round(a,2) for a in tail_alpha]}  (N=200 tail was 0.65/0.60)")

    make_figs(L, R["real_eig"], R["pa_thresh"], R["K"], R["N"], R["split_half"], R["big5"], R["factors"])

    full = {"stage": "native_model_n400", "N": R["N"], "K_parallel": R["K"], "K_used": R["Kuse"],
            "factors": R["factors"], "split_half_congruence": R["split_half"], "vs_bigfive": R["big5"],
            "structure_vs_n200": R["vs200_mean"], "structure_vs_n200_pairs": R["vs200_pairs"],
            "mean_alpha": mean_alpha, "mean_omega": mean_omega,
            "loadings": L.tolist(), "facet_order": FACETS, "scores": X.tolist(),
            "real_eig": R["real_eig"].tolist(), "personas": personas}
    json.dump(full, open(os.path.join(OUT, "native_model_n400.json"), "w"), indent=2)

    # compact summary — READ THIS, not the 1 MB model file
    summary = {"iter": 22, "stage": "native_model_n400", "N": R["N"], "K_parallel": R["K"], "K_used": R["Kuse"],
               "split_half_n200": 0.77, "split_half_n400": round(R["split_half"], 3),
               "vs_bigfive_n200": 0.346, "vs_bigfive_n400": round(R["big5"], 3),
               "structure_held_vs_n200": round(R["vs200_mean"], 3),
               "structure_held_pairs": [round(x, 3) for x in R["vs200_pairs"]],
               "core_alpha_n400": [round(a, 3) for a in core_alpha],
               "tail_alpha_n200": [0.65, 0.60], "tail_alpha_n400": [round(a, 3) for a in tail_alpha],
               "mean_alpha_n400": round(mean_alpha, 3),
               "factor_top_facets": [{"F": f["factor"], "var": f["var_share"], "alpha": f["alpha"],
                                      "pos": [p[0] for p in f["pos"][:4]],
                                      "neg": [n[0] for n in f["neg"][:3]]} for f in R["factors"]],
               "verdict": ("STABILIZED — 5-factor replicable" if R["split_half"] > 0.85
                           else "PARTIAL — tail firmer but <0.85; lead with 3-core"
                           if R["split_half"] > 0.77 else "NO GAIN — lead with 3-core")}
    json.dump(summary, open(os.path.join(OUT, "n400_result.json"), "w"), indent=2)

    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({k: summary[k] for k in
                 ("iter", "stage", "N", "split_half_n400", "vs_bigfive_n400",
                  "structure_held_vs_n200", "tail_alpha_n400", "verdict")}) + "\n")

    # final durable commit of the result
    git("add", os.path.join(OUT, "native_model_n400.json"), os.path.join(OUT, "n400_result.json"),
        os.path.join(OUT, "metrics.jsonl"), CKPT,
        os.path.join(FIGS, "native_model_n400_loadings.png"),
        os.path.join(FIGS, "native_model_n400_scree.png"))
    git("commit", "-m", f"research: iter-22 native model stabilized at N={R['N']} (split-half {R['split_half']:.2f}) — {summary['verdict']}")
    print(f"\nsaved -> native_model_n400.json, n400_result.json, metrics.jsonl, 2 figs. verdict: {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
