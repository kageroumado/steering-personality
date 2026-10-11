"""
reauthor_test.py — iter-8 (D14): test the item-authoring hypothesis on the weak 5.

iter-7 flagged SelfControl, Sociability, Honesty, Introspection, EasyGoingness weak/broken on
the forced-choice SELF-slope, while their vectors steer fine on the Likert readout — i.e. the
ITEMS, not the vectors, are the bottleneck. This re-runs the Tier-0 self-slope for those 5
facets with BOTH the original v1 items and the re-authored v2 items (facets27.FC_ITEMS_V2),
in the same pass, so the comparison is apples-to-apples. If v2 self-slope >> v1, the
item-vector-misalignment diagnosis is confirmed.

Output → research/loop/reauthor_test.json.
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
from statistics import mean
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
from extract_vectors import VEC_DIR
import facets27 as F
import personas as P

from paths import OUT
PRIME = "I'll be honest about myself.\n\n"
DOSE = [-2.0, -1.0, 0.0, 1.0, 2.0]
WEAK5 = ["SelfControl", "Sociability", "Honesty", "Introspection", "EasyGoingness"]


def ols_slope(xs, ys):
    x = np.asarray(xs, float) - np.mean(xs)
    d = float((x * x).sum())
    return float((x * (np.asarray(ys, float) - np.mean(ys))).sum() / d) if d else 0.0


def mean_delta(scorer, items):
    vals = []
    for it in items:
        s = scorer.score(it["stem"], [it["hi"], it["lo"]])
        vals.append(s[0]["logprob_sum"] - s[1]["logprob_sum"])
    return mean(vals)


def main():
    v1 = F.items_by_facet()
    v2 = F.FC_ITEMS_V2
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in WEAK5}

    # series[band][facet][version][k]
    series = {b: {f: {"v1": {}, "v2": {}} for f in WEAK5} for b in P.BANDS}
    t0 = time.time()
    for band in P.BANDS:
        cmodel = ControlModel(base, P.BANDS[band])
        for facet in WEAK5:
            for k in DOSE:
                cmodel.reset()
                if k != 0.0:
                    cmodel.set_control(vectors[facet], float(k))
                scorer = PrefixScorer(cmodel, tok, PRIME, n_options=2, device=device)
                series[band][facet]["v1"][k] = mean_delta(scorer, v1[facet])
                series[band][facet]["v2"][k] = mean_delta(scorer, v2[facet])
                del scorer
                cmodel.reset()
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        print(f"  band {band} done ({time.time()-t0:.0f}s)", flush=True)

    def metrics(d):
        ys = [d[k] for k in DOSE]
        slope = ols_slope(DOSE, ys)
        rho = spearmanr(DOSE, ys)[0]
        return slope, (0.0 if rho != rho else float(rho))

    report = {}
    print(f"\n=== ITER-8 RE-AUTHOR TEST — v1 vs v2 forced-choice self-slope ===")
    print(f"  {'facet':16} {'ver':3} {'band':5} {'slope':>8} {'rho':>6}  verdict")
    for facet in WEAK5:
        report[facet] = {}
        for ver in ("v1", "v2"):
            best = max(P.BANDS, key=lambda b: abs(metrics(series[b][facet][ver])[0]))
            slope, rho = metrics(series[best][facet][ver])
            aslope = abs(slope)
            verdict = "strong" if (aslope >= 0.30 and abs(rho) >= 0.80) else \
                      "weak" if (aslope >= 0.10 and abs(rho) >= 0.80) else "broken"
            report[facet][ver] = {"band": best, "slope": round(slope, 4),
                                  "rho": round(rho, 3), "verdict": verdict,
                                  "series": {f"{k:+}": round(series[best][facet][ver][k], 4) for k in DOSE}}
            print(f"  {facet:16} {ver:3} {best:5} {slope:+8.3f} {rho:+6.2f}  {verdict}")
        print()

    improved = [f for f in WEAK5
                if abs(report[f]["v2"]["slope"]) > abs(report[f]["v1"]["slope"]) + 0.05]
    fixed = [f for f in WEAK5 if report[f]["v2"]["verdict"] in ("strong", "weak")
             and report[f]["v1"]["verdict"] == "broken"]
    print(f"v2 slope clearly larger for: {improved}")
    print(f"broken→usable under v2: {fixed}")

    out = {"stage": "reauthor_test_iter8", "dose": DOSE, "weak5": WEAK5,
           "report": report, "improved": improved, "fixed_broken_to_usable": fixed,
           "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, "reauthor_test.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 8, "stage": "reauthor_test",
                             "v1_slope": {f: report[f]["v1"]["slope"] for f in WEAK5},
                             "v2_slope": {f: report[f]["v2"]["slope"] for f in WEAK5},
                             "improved": improved, "fixed": fixed}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, 'reauthor_test.json')}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
