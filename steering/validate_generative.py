"""
validate_generative.py — iter-21: external validation by quiz-mode administration.

Everything so far reads traits via isolated-item logprob deltas (each item scored alone behind a
fixed persona prefix). A human takes a test differently: a few questions at a time, picking an
answer, with their own prior answers sitting in context. This script administers the reduced
human instrument in exactly that mode and asks whether the tool still recovers the iter-19 ground
truth:

  * QUIZ MODE   — persona prefix + a running batch of A/B questions. The persona PICKS one
                  (read as the logprob margin of the answer letter), the CHOSEN option is appended
                  to the transcript, and the next question is asked with that history in context.
                  Batches of BATCH items reset the context (a few at a time, no unbounded drift).
  * POSITION-DEBIAS — A/B pole order randomized per item; the hi-pole margin is sign-corrected, so
                  the base model's letter prior cancels in expectation.
  * RECOVERY    — quiz-recovered metatrait z vs the iter-19/iter-20 ground-truth metatrait z
                  (per-axis Pearson r = precision), per-persona profile correlation, and type
                  agreement (does quiz mode land each persona in the same cluster?).

Reduced instrument = every facet with |loading|>0.40 on any metatrait, 2 items each. A balanced
subset of personas (by iter-20 type). Output -> research/loop/validation_generative.json + fig.

env: SMOKE=1 → 2 personas × 1 batch (parse/format check). N_PER_TYPE overrides subset size.
"""
from __future__ import annotations
import os, json, time, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
import facets27 as F

from paths import OUT, FIGS
METATRAITS = ["Plasticity", "Drive", "Fluidity", "Stability", "Communion"]
CORE = [0, 1, 2]
LOAD_THRESH = 0.40
BATCH = 6
SMOKE = os.environ.get("SMOKE", "") == "1"
N_PER_TYPE = int(os.environ.get("N_PER_TYPE", "8"))
RNG = np.random.default_rng(21)


def build_instrument(L, facets):
    """facets with |loading|>thresh on any metatrait; 2 items each (from fc_items 'best')."""
    keep = [facets[i] for i in range(len(facets)) if np.abs(L[i]).max() > LOAD_THRESH]
    by_facet = {}
    for it in F.fc_items("best"):
        by_facet.setdefault(it["facet"], []).append(it)
    items = []
    for f in keep:
        items.extend(by_facet[f][:2])
    return items, keep


def administer_quiz(model, tok, device, desc, items):
    """Return per-item hi-pole logprob margins (sign-corrected), quiz mode, batched."""
    prefix = (f"{desc} This is me being honest about who I really am.\n\n"
              "I take a short questionnaire about myself. For each, I pick the option that fits "
              "me better — A or B.\n\n")
    margins = {}
    for b0 in range(0, len(items), BATCH):
        batch = items[b0:b0 + BATCH]
        scorer = PrefixScorer(model, tok, prefix, n_options=2, device=device)
        transcript = ""
        for it in batch:
            flip = bool(RNG.integers(2))
            optA, optB = (it["lo"], it["hi"]) if flip else (it["hi"], it["lo"])
            q = (f"{transcript}Q. {it['stem']}...\n  A){optA}\n  B){optB}\n  Answer:")
            res = scorer.score(q, [" A", " B"])
            lpA, lpB = res[0]["logprob_sum"], res[1]["logprob_sum"]
            hi_margin = (lpB - lpA) if flip else (lpA - lpB)
            margins[it["id"]] = (it["facet"], hi_margin)
            chosen = "A" if lpA > lpB else "B"
            chosen_text = optA if chosen == "A" else optB
            transcript = q + f" {chosen}){chosen_text}\n\n"
        del scorer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    return margins


def main():
    tool = json.load(open(os.path.join(OUT, "human_tool.json")))
    nm = json.load(open(os.path.join(OUT, "native_model.json")))
    facets = nm["facet_order"]; L = np.array(nm["loadings"])
    personas = tool["personas"]; labels = np.array(tool["typology"]["labels"])
    M_truth = np.array(tool["scores_metatrait_z"])           # ground-truth metatrait z (N×5)
    k = tool["typology"]["k"]
    # type centroids in metatrait space (for type-agreement test)
    cent = np.array([M_truth[labels == c].mean(0) for c in range(k)])

    items, keep = build_instrument(L, facets)
    fac_idx = {f: i for i, f in enumerate(facets)}
    print(f"instrument: {len(items)} items over {len(keep)} facets; {len(personas)} personas total")

    # balanced subset by type
    if SMOKE:
        subset = [int(np.where(labels == c)[0][0]) for c in range(2)]
    else:
        subset = []
        for c in range(k):
            pool = np.where(labels == c)[0]
            subset.extend(RNG.choice(pool, min(N_PER_TYPE, len(pool)), replace=False).tolist())
    subset = sorted(set(subset))
    print(f"subset: {len(subset)} personas" + (" [SMOKE]" if SMOKE else ""))

    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()

    # administer
    t0 = time.time()
    quiz_facet = np.full((len(subset), len(facets)), np.nan)
    for si, pi in enumerate(subset):
        margins = administer_quiz(model, tok, device, personas[pi], items)
        per_facet = {}
        for fid, (fac, m) in margins.items():
            per_facet.setdefault(fac, []).append(m)
        for fac, vals in per_facet.items():
            quiz_facet[si, fac_idx[fac]] = float(np.mean(vals))
        if (si + 1) % 5 == 0 or SMOKE:
            print(f"  [{si+1}/{len(subset)}] {time.time()-t0:.0f}s", flush=True)
    print(f"administered {len(subset)} personas in {time.time()-t0:.0f}s")

    # standardize quiz facet scores over the administered facets (per facet, across subset)
    cols = [fac_idx[f] for f in keep]
    Q = quiz_facet[:, cols]
    Qz = (Q - np.nanmean(Q, 0)) / (np.nanstd(Q, 0) + 1e-9)
    Lk = L[cols]                                            # facets × 5 loadings (administered only)

    def scale_scores(Z, Lm):
        S = np.zeros((Z.shape[0], Lm.shape[1]))
        for f in range(Lm.shape[1]):
            idx = np.where(np.abs(Lm[:, f]) > 0.35)[0]
            w = Lm[idx, f]
            S[:, f] = (Z[:, idx] @ w) / (np.abs(w).sum() + 1e-12)
        return S

    S_quiz = scale_scores(Qz, Lk)
    M_quiz = (S_quiz - S_quiz.mean(0)) / (S_quiz.std(0) + 1e-9)
    M_sub = M_truth[subset]                                 # ground truth for the same personas

    # per-axis recovery r
    recov = [float(np.corrcoef(M_quiz[:, f], M_sub[:, f])[0, 1]) for f in range(5)]
    # per-persona profile correlation (all 5 axes)
    prof_r = [float(np.corrcoef(M_quiz[i], M_sub[i])[0, 1]) for i in range(len(subset))]
    # type agreement: nearest centroid in quiz space vs truth label
    from scipy.spatial.distance import cdist
    quiz_type = cdist(M_quiz, cent).argmin(1)
    truth_type = labels[subset]
    type_agree = float(np.mean(quiz_type == truth_type))
    core_recov = float(np.mean([recov[f] for f in CORE]))

    print(f"\n=== QUIZ-MODE VALIDATION (n={len(subset)}) ===")
    for f in range(5):
        tier = "CORE" if f in CORE else "prov"
        print(f"  {METATRAITS[f]:11s} recovery r={recov[f]:+.2f}  [{tier}]")
    print(f"  core-3 mean recovery r = {core_recov:.2f}")
    print(f"  median per-persona profile r = {np.median(prof_r):.2f}")
    print(f"  type agreement (same cluster) = {type_agree:.2f}  (chance≈{1/k:.2f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.bar(range(5), recov, color=["#1f77b4" if f in CORE else "#bbb" for f in range(5)])
    ax1.set_xticks(range(5)); ax1.set_xticklabels(METATRAITS, rotation=25, ha="right", fontsize=9)
    ax1.axhline(0, color="k", lw=0.5); ax1.set_ylim(-0.2, 1.0)
    ax1.set_ylabel("quiz vs ground-truth r"); ax1.set_title(
        f"Per-axis recovery, quiz mode (n={len(subset)})\ncore-3 r={core_recov:.2f}, "
        f"type agreement={type_agree:.2f}")
    # scatter of the strongest core axis
    fbest = int(np.argmax(recov))
    ax2.scatter(M_sub[:, fbest], M_quiz[:, fbest], s=40, alpha=0.75)
    ax2.set_xlabel(f"ground-truth {METATRAITS[fbest]} (z)")
    ax2.set_ylabel(f"quiz-recovered {METATRAITS[fbest]} (z)")
    ax2.set_title(f"{METATRAITS[fbest]}: r={recov[fbest]:.2f}")
    lim = [min(M_sub[:, fbest].min(), M_quiz[:, fbest].min()), max(M_sub[:, fbest].max(), M_quiz[:, fbest].max())]
    ax2.plot(lim, lim, "--", color="gray", lw=1)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "validation_generative.png"), dpi=130); plt.close(fig)

    out = {"stage": "validation_generative", "n": len(subset), "n_items": len(items),
           "n_facets": len(keep), "batch": BATCH, "recovery_r": recov, "core_recovery": core_recov,
           "median_profile_r": float(np.median(prof_r)), "type_agreement": type_agree,
           "chance_type": 1 / k, "subset": subset, "keep_facets": keep,
           "M_quiz": M_quiz.tolist(), "M_truth_sub": M_sub.tolist()}
    json.dump(out, open(os.path.join(OUT, "validation_generative.json"), "w"), indent=2)
    if not SMOKE:
        with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
            fh.write(json.dumps({"iter": 21, "stage": "validation_generative", "n": len(subset),
                                 "core_recovery": round(core_recov, 2),
                                 "type_agreement": round(type_agree, 2),
                                 "median_profile_r": round(float(np.median(prof_r)), 2)}) + "\n")
    print("\nsaved -> validation_generative.json + figure")


if __name__ == "__main__":
    main()
