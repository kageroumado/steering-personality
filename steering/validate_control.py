"""
validate_control.py — iter-21 control: isolate WHY quiz-mode recovery is only moderate.

Quiz-mode recovery (validate_generative.py) was core-3 r=0.42 — but that conflates three changes vs
the iter-19 ground truth at once: (a) fewer items (54 vs 135), (b) a binary single-letter readout vs
graded hi/lo completion logprob, (c) conversational running context + pole-order randomization.

This control administers the SAME 54 items to the SAME 40 personas with the GOLD readout (isolated
item, full hi/lo completion logprob — exactly the iter-19 procedure, just the reduced item set, no
quiz context). Recovery here = the instrument's CEILING. The gap between this and quiz-mode = the cost
of (binary readout + conversation), the part that's about HOW the test is taken, not the items.

Output -> research/loop/validation_control.json.
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
import facets27 as F

from paths import OUT
METATRAITS = ["Plasticity", "Drive", "Fluidity", "Stability", "Communion"]
CORE = [0, 1, 2]
LOAD_THRESH = 0.40


def main():
    val = json.load(open(os.path.join(OUT, "validation_generative.json")))
    tool = json.load(open(os.path.join(OUT, "human_tool.json")))
    nm = json.load(open(os.path.join(OUT, "native_model.json")))
    facets = nm["facet_order"]; L = np.array(nm["loadings"])
    personas = tool["personas"]; M_truth = np.array(tool["scores_metatrait_z"])
    subset = val["subset"]; keep = val["keep_facets"]
    fac_idx = {f: i for i, f in enumerate(facets)}

    # same reduced instrument as the quiz (2 items/facet over the kept facets)
    by_facet = defaultdict(list)
    for it in F.fc_items("best"):
        by_facet[it["facet"]].append(it)
    items = []
    for f in keep:
        items.extend(by_facet[f][:2])
    print(f"control: {len(items)} items over {len(keep)} facets; {len(subset)} personas (isolated gold readout)")

    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()

    by_facet_items = defaultdict(list)
    for it in items:
        by_facet_items[it["facet"]].append(it)

    t0 = time.time()
    ctrl_facet = np.full((len(subset), len(facets)), np.nan)
    for si, pi in enumerate(subset):
        scorer = PrefixScorer(model, tok, f"{personas[pi]} This is me being honest about who I really am.\n\n",
                              n_options=2, device=device)
        for fac in keep:
            vals = [scorer.score(it["stem"], [it["hi"], it["lo"]]) for it in by_facet_items[fac]]
            ctrl_facet[si, fac_idx[fac]] = float(np.mean([s[0]["logprob_sum"] - s[1]["logprob_sum"] for s in vals]))
        del scorer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        if (si + 1) % 10 == 0:
            print(f"  [{si+1}/{len(subset)}] {time.time()-t0:.0f}s", flush=True)
    print(f"administered {len(subset)} personas in {time.time()-t0:.0f}s")

    cols = [fac_idx[f] for f in keep]
    C = ctrl_facet[:, cols]
    Cz = (C - C.mean(0)) / (C.std(0) + 1e-9)
    Lk = L[cols]

    def scale_scores(Z, Lm):
        S = np.zeros((Z.shape[0], Lm.shape[1]))
        for f in range(Lm.shape[1]):
            idx = np.where(np.abs(Lm[:, f]) > 0.35)[0]
            w = Lm[idx, f]
            S[:, f] = (Z[:, idx] @ w) / (np.abs(w).sum() + 1e-12)
        return S

    M_ctrl = (lambda S: (S - S.mean(0)) / (S.std(0) + 1e-9))(scale_scores(Cz, Lk))
    M_sub = M_truth[subset]
    recov = [float(np.corrcoef(M_ctrl[:, f], M_sub[:, f])[0, 1]) for f in range(5)]
    core = float(np.mean([recov[f] for f in CORE]))
    quiz = val["recovery_r"]; quiz_core = val["core_recovery"]

    print(f"\n=== CONTROL (54-item, ISOLATED gold readout) vs QUIZ-MODE ===")
    print(f"  {'axis':11s} {'ceiling':>8s} {'quiz':>7s} {'gap':>7s}")
    for f in range(5):
        print(f"  {METATRAITS[f]:11s} {recov[f]:+8.2f} {quiz[f]:+7.2f} {recov[f]-quiz[f]:+7.2f}  "
              f"[{'CORE' if f in CORE else 'prov'}]")
    print(f"  {'CORE-3 mean':11s} {core:+8.2f} {quiz_core:+7.2f} {core-quiz_core:+7.2f}")
    print(f"\n  reading: ceiling−truth gap = cost of the 54-item REDUCTION; "
          f"ceiling−quiz gap = cost of binary readout + conversation (HOW it's taken).")

    json.dump({"stage": "validation_control", "n": len(subset), "n_items": len(items),
               "recovery_r_ceiling": recov, "core_ceiling": core,
               "recovery_r_quiz": quiz, "core_quiz": quiz_core},
              open(os.path.join(OUT, "validation_control.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 21, "stage": "validation_control",
                             "core_ceiling": round(core, 2), "core_quiz": round(quiz_core, 2)}) + "\n")
    print("\nsaved -> validation_control.json")


if __name__ == "__main__":
    main()
