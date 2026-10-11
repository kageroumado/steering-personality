"""
e1_instruct_manifold.py — iter-23 / EXPERIMENT-PLAN E1: the base↔instruct natural experiment.

Same base weights, post-training the only difference. Administer the IDENTICAL 409-persona forced-choice
instrument (raw-completion protocol, byte-identical to iter-22 — only the model differs) to
Llama-3.1-8B-Instruct, derive its native behavioral structure, and compare to the stabilized N=409 BASE
model. Tests the pre-registered predictions (EXPERIMENT-PLAN.md §E1):
  P1 structure INVARIANT (base↔instruct φ ≥ 0.85): post-training moves location, not shape.
  P2 means SHIFT toward the Assistant pole (↑desirable facets, ↓volatility/disinhibition).
  P3 measurability ≥ base (instruct follows the test at least as well) → measurability ⊥ governance.

Resume-safe: per-persona checkpoint to e1_instruct_checkpoint.jsonl + git commits every 50.
Output -> research/loop/e1_instruct.json, e1_result.json, figs/e1_*.png.

Run:  cd steering && PT_MODEL=Meta-Llama-3.1-8B-Instruct .venv/bin/python e1_instruct_manifold.py
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
from common import pick_device
from fastscore import PrefixScorer
import facets27 as F

from paths import REPO, OUT, FIGS
CKPT = os.path.join(OUT, "e1_instruct_checkpoint.jsonl")
MODEL = os.path.expanduser(os.environ.get("PT_MODEL", "Meta-Llama-3.1-8B-Instruct"))
ITEMS = F.fc_items("best")
FACETS = F.FACET_ORDER
COMMIT_EVERY = 50
PREFIX_SUFFIX = " This is me being honest about who I really am.\n\n"   # byte-identical to iter-22

# rough social-desirability sign per facet (the "Assistant pole" prediction): instruct should drift +
DESIRABILITY = {
    "WellBeing": +1, "EmotionalStability": +1, "Honesty": +1, "Compassion": +1, "Trust": +1,
    "Industry": +1, "SelfControl": +1, "Adaptability": +1, "Charisma": +1, "Humor": +1,
    "Creativity": +1, "Intellect": +1, "Introspection": +1, "ArtAppreciation": +1, "Order": +1,
    "Perfectionism": +1, "EmotionalExpressiveness": +1, "Sociability": +1,
    "Anxiety": -1, "Irritability": -1, "Impulsivity": -1, "AttentionSeeking": -1,
    "SensationSeeking": -1, "Authoritarianism": -1, "Conformity": 0, "Conservatism": 0,
}


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
    return max(1, int((real > np.percentile(rand, 95, axis=0)).sum())), real


def efa(X, K):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    C = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(C); idx = np.argsort(w)[::-1]
    eig = np.clip(w[idx], 0, None); V = V[:, idx]
    L = varimax(V[:, :K] * np.sqrt(eig[:K]))
    return L[:, np.argsort((L ** 2).sum(0))[::-1]], Z


def cronbach_alpha(items):
    m = items.shape[1]
    if m < 2:
        return float("nan")
    vs = items.var(0, ddof=1).sum(); tv = items.sum(1).var(ddof=1)
    return float((m / (m - 1)) * (1 - vs / tv)) if tv > 0 else float("nan")


def git(*args):
    try:
        subprocess.run(["git", "-C", REPO, *args], check=True, capture_output=True, timeout=120)
        return True
    except Exception as e:
        print(f"  [git] skipped: {' '.join(args)} ({type(e).__name__})", flush=True)
        return False


def load_checkpoint():
    done = {}
    if os.path.exists(CKPT):
        for line in open(CKPT):
            line = line.strip()
            if line:
                r = json.loads(line); done[int(r["idx"])] = r
    return done


def administer(personas):
    device = pick_device()
    done = load_checkpoint()
    todo = [i for i in range(len(personas)) if i not in done]
    print(f"resume: {len(done)}/{len(personas)} scored, {len(todo)} to go on {MODEL}", flush=True)
    if todo:
        print(f"loading {MODEL} on {device} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
        by_facet = defaultdict(list)
        for it in ITEMS:
            by_facet[it["facet"]].append(it)
        t0 = time.time(); n = 0
        with open(CKPT, "a") as ck:
            for pi in todo:
                desc = personas[pi]
                scorer = PrefixScorer(model, tok, f"{desc}{PREFIX_SUFFIX}", n_options=2, device=device)
                row = np.zeros(len(FACETS))
                for fj, facet in enumerate(FACETS):
                    vals = [scorer.score(it["stem"], [it["hi"], it["lo"]]) for it in by_facet[facet]]
                    row[fj] = float(np.mean([s[0]["logprob_sum"] - s[1]["logprob_sum"] for s in vals]))
                del scorer
                ck.write(json.dumps({"idx": pi, "persona": desc, "scores": row.tolist()}) + "\n"); ck.flush()
                done[pi] = {"idx": pi, "persona": desc, "scores": row.tolist()}; n += 1
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
                if n % 5 == 0:
                    el = time.time() - t0; eta = el / n * (len(todo) - n)
                    print(f"  [{len(done)}/{len(personas)}] {el:.0f}s, {el/n:.1f}s/p, ETA {eta/60:.0f}min", flush=True)
                if n % COMMIT_EVERY == 0:
                    git("add", CKPT); git("commit", "-m", f"research: E1 instruct checkpoint {len(done)}/{len(personas)}")
        print(f"administered {n} on instruct in {(time.time()-t0)/60:.0f}min", flush=True)
    return np.array([done[i]["scores"] for i in range(len(personas))])


def main():
    base = json.load(open(os.path.join(OUT, "native_model_n400.json")))
    personas = base["personas"]
    Xb = np.array(base["scores"]); Lb = np.array(base["loadings"])      # base N=409 structure
    assert base["facet_order"] == FACETS

    Xi = administer(personas)                                            # instruct, same personas
    N = Xi.shape[0]
    K, _ = parallel_analysis(Xi); Kuse = max(K, 5)
    Li, Zi = efa(Xi, Kuse)

    # P1 — structure invariance base↔instruct (per-factor best-match congruence, Hungarian)
    Cb = np.array([[congruence(Li[:, j], Lb[:, k]) for k in range(Lb.shape[1])] for j in range(Kuse)])
    ri, ci = linear_sum_assignment(-Cb)
    pairs = sorted([float(Cb[ri[m], ci[m]]) for m in range(min(Kuse, Lb.shape[1]))], reverse=True)
    struct_congruence = float(np.mean(pairs))

    # P2 — mean shift toward the Assistant pole. z-score each model's facets within itself, compare means.
    Zb_mean = (Xb - Xb.mean(0)).mean(0)        # ~0 by construction; use raw-unit shift instead
    raw_shift = Xi.mean(0) - Xb.mean(0)         # per-facet mean shift in logprob-delta units
    des = np.array([DESIRABILITY.get(f, 0) for f in FACETS], float)
    # does the shift align with desirability? positive r => instruct drifts toward the desirable pole
    shift_des_r = float(np.corrcoef(raw_shift, des)[0, 1])
    top_up = sorted(((FACETS[i], float(raw_shift[i])) for i in range(27)), key=lambda t: -t[1])[:6]
    top_dn = sorted(((FACETS[i], float(raw_shift[i])) for i in range(27)), key=lambda t: t[1])[:6]

    # P3 — measurability on instruct: per-factor alpha + split-half replicability
    facs = []
    for j in range(Kuse):
        col = Li[:, j]; idx = [i for i in range(27) if abs(col[i]) > 0.35]
        signed = Zi[:, idx] * np.sign(col[idx])
        facs.append({"factor": j + 1, "n_facets": len(idx),
                     "alpha": round(cronbach_alpha(signed), 3) if len(idx) >= 2 else None,
                     "pos": [FACETS[i] for i in np.argsort(col)[::-1] if col[i] > 0.35][:5],
                     "neg": [FACETS[i] for i in np.argsort(col) if col[i] < -0.35][:4]})
    rng = np.random.default_rng(7); perm = rng.permutation(N); h1, h2 = perm[:N // 2], perm[N // 2:]
    L1, _ = efa(Xi[h1], Kuse); L2, _ = efa(Xi[h2], Kuse)
    Csh = np.array([[congruence(L1[:, i], L2[:, j]) for j in range(Kuse)] for i in range(Kuse)])
    r2, c2 = linear_sum_assignment(-Csh)
    split_half_i = float(np.mean([Csh[r2[m], c2[m]] for m in range(Kuse)]))

    # vs Big Five (instruct)
    domI = np.zeros((27, 5))
    for di, d in enumerate(F.DOMAINS):
        for i, f in enumerate(FACETS):
            domI[i, di] = 1.0 if F.DOMAIN_OF[f] == d else 0.0
    domI -= domI.mean(0, keepdims=True)
    CB = np.array([[congruence(Li[:, j], domI[:, di]) for di in range(5)] for j in range(5)])
    rb, cb = linear_sum_assignment(-CB)
    big5_i = float(np.mean([CB[rb[m], cb[m]] for m in range(5)]))

    # degeneracy guard: are instruct readouts non-flat? (assistant-pole collapse would crush variance)
    facet_sd_base = float(Xb.std(0).mean()); facet_sd_inst = float(Xi.std(0).mean())

    print(f"\n=== E1 BASE↔INSTRUCT (N={N}) ===", flush=True)
    print(f"  P1 structure congruence base↔instruct = {struct_congruence:.3f}  pairs={[round(x,2) for x in pairs]}")
    print(f"     {'INVARIANT (≥0.85) ✓' if struct_congruence>=0.85 else 'PARTIAL/SHIFTED ✗'}")
    print(f"  P2 mean-shift↔desirability r = {shift_des_r:+.3f}  ({'drifts to Assistant pole ✓' if shift_des_r>0.2 else 'no clear desirability drift'})")
    print(f"     up:   {', '.join(f'{n}{v:+.2f}' for n,v in top_up)}")
    print(f"     down: {', '.join(f'{n}{v:+.2f}' for n,v in top_dn)}")
    print(f"  P3 instruct split-half φ = {split_half_i:.3f} (base 0.883); mean α = "
          f"{np.nanmean([f['alpha'] for f in facs if f['alpha']]):.3f}")
    print(f"  vs Big Five (instruct) φ = {big5_i:.3f} (base 0.322)")
    print(f"  facet SD: base {facet_sd_base:.2f} vs instruct {facet_sd_inst:.2f}  "
          f"({'instruct compressed' if facet_sd_inst < 0.7*facet_sd_base else 'comparable signal'})")

    # figure: side-by-side loadings
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    for ax, L, ttl in [(axes[0], Lb, f"BASE N=409"), (axes[1], Li, f"INSTRUCT N={N}")]:
        im = ax.imshow(L, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(27)); ax.set_yticklabels(FACETS, fontsize=6)
        for i, f in enumerate(FACETS):
            ax.get_yticklabels()[i].set_color(
                {"Agreeableness": "#2ca02c", "Conscientiousness": "#1f77b4", "Neuroticism": "#d62728",
                 "Extraversion": "#ff7f0e", "Openness": "#9467bd"}[F.DOMAIN_OF[f]])
        ax.set_xticks(range(L.shape[1])); ax.set_xticklabels([f"F{j+1}" for j in range(L.shape[1])])
        ax.set_title(ttl)
    fig.suptitle(f"E1: native structure base↔instruct (congruence φ={struct_congruence:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e1_base_vs_instruct.png"), dpi=130); plt.close(fig)

    full = {"stage": "e1_instruct", "model": MODEL, "N": N, "K_parallel": K, "K_used": Kuse,
            "structure_congruence_base_instruct": struct_congruence, "congruence_pairs": pairs,
            "mean_shift": raw_shift.tolist(), "shift_desirability_r": shift_des_r,
            "top_up": top_up, "top_down": top_dn, "factors_instruct": facs,
            "split_half_instruct": split_half_i, "vs_bigfive_instruct": big5_i,
            "facet_sd_base": facet_sd_base, "facet_sd_instruct": facet_sd_inst,
            "loadings_instruct": Li.tolist(), "scores_instruct": Xi.tolist(), "facet_order": FACETS}
    json.dump(full, open(os.path.join(OUT, "e1_instruct.json"), "w"), indent=2)

    verdict = ("P1✓ structure invariant; " if struct_congruence >= 0.85 else "P1✗ structure shifted; ") + \
              ("P2✓ assistant-pole drift; " if shift_des_r > 0.2 else "P2~ no clear drift; ") + \
              ("P3✓ measurable ≥ base" if split_half_i >= 0.85 else "P3~ less stable than base")
    summary = {"iter": 23, "stage": "e1_instruct", "N": N,
               "structure_congruence_base_instruct": round(struct_congruence, 3),
               "congruence_pairs": [round(x, 3) for x in pairs],
               "shift_desirability_r": round(shift_des_r, 3),
               "top_up": [[n, round(v, 2)] for n, v in top_up[:4]],
               "top_down": [[n, round(v, 2)] for n, v in top_dn[:4]],
               "split_half_instruct": round(split_half_i, 3), "split_half_base": 0.883,
               "vs_bigfive_instruct": round(big5_i, 3),
               "facet_sd_base": round(facet_sd_base, 2), "facet_sd_instruct": round(facet_sd_inst, 2),
               "verdict": verdict}
    json.dump(summary, open(os.path.join(OUT, "e1_result.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps(summary) + "\n")

    git("add", os.path.join(OUT, "e1_instruct.json"), os.path.join(OUT, "e1_result.json"),
        os.path.join(OUT, "metrics.jsonl"), CKPT, os.path.join(FIGS, "e1_base_vs_instruct.png"))
    git("commit", "-m", f"research: iter-23 E1 base↔instruct (struct φ {struct_congruence:.2f}) — {verdict}")
    print(f"\nsaved -> e1_instruct.json, e1_result.json. verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
