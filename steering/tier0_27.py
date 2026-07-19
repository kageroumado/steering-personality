"""
tier0_27.py — breadth 5→27 (D13), Pass 1: automated Tier-0 viability scan.

For each of the 27 facets, decide — empirically, not by hand — three things the original 5 had
hand-tuned (FACET_BAND / DOSE / sign in personas.py):
    * BAND  — late (layers 17-31) vs mid (12-20): which injection band steers this facet harder
    * SIGN  — pca_diff's principal direction is arbitrary; sign = direction that RAISES the trait
    * VERDICT — strong / weak / broken, from the forced-choice SELF-response

Readout: the iter-4 pole-contrast Delta on the facet's OWN 5 forced-choice items
(mean over items of lp(hi) - lp(lo)). Steer the facet's vector at raw coeff k over a symmetric
dose grid; a steerable facet's mean-self-Delta moves monotonically with k.

    self_slope = OLS slope(mean_self_Delta ~ k)      # magnitude of induction
    self_rho   = Spearman(k, mean_self_Delta)         # monotonicity
    band = argmax_band |self_slope|, sign = sign(self_slope at that band)

Memory rule (D6): wrap each band ONCE, iterate all facets/doses inside it, then unwrap.
Output → research/loop/viability27.json (consumed by fc27_matrix.py to administer the full
135-item response per facet at its chosen band/sign).
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
import personas as P  # BANDS

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
PRIME = "I'll be honest about myself.\n\n"   # same register-setter as forced_choice.py
DOSE = [-2.0, -1.0, 0.0, 1.0, 2.0]
STRONG_SLOPE, WEAK_SLOPE, MONO = 0.30, 0.10, 0.80


def ols_slope(xs, ys):
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    x = x - x.mean()
    denom = float((x * x).sum())
    return float((x * (y - y.mean())).sum() / denom) if denom else 0.0


def main():
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()

    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf"))
               for f in F.FACET_ORDER}
    own_items = F.items_by_facet()

    # self_delta[band][facet][k] = mean over facet's own FC items of lp(hi)-lp(lo)
    self_delta = {b: defaultdict(dict) for b in P.BANDS}
    t0 = time.time(); done = 0; total = len(P.BANDS) * len(F.FACET_ORDER) * len(DOSE)
    for band in P.BANDS:
        cmodel = ControlModel(base, P.BANDS[band])
        for facet in F.FACET_ORDER:
            items = own_items[facet]
            for k in DOSE:
                cmodel.reset()
                if k != 0.0:
                    cmodel.set_control(vectors[facet], float(k))
                scorer = PrefixScorer(cmodel, tok, PRIME, n_options=2, device=device)
                vals = []
                for it in items:
                    s = scorer.score(it["stem"], [it["hi"], it["lo"]])
                    vals.append(s[0]["logprob_sum"] - s[1]["logprob_sum"])
                self_delta[band][facet][k] = mean(vals)
                del scorer
                cmodel.reset()
                done += 1
                if done % 20 == 0:
                    print(f"  [{done}/{total}] {time.time()-t0:.0f}s", flush=True)
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # ---- per-facet band/sign/verdict ----
    report = {}
    for facet in F.FACET_ORDER:
        per_band = {}
        for band in P.BANDS:
            ys = [self_delta[band][facet][k] for k in DOSE]
            slope = ols_slope(DOSE, ys)
            rho = spearmanr(DOSE, ys)[0]
            rho = 0.0 if rho != rho else float(rho)   # nan guard (flat)
            per_band[band] = {"slope": round(slope, 4), "rho": round(rho, 3),
                              "series": {f"{k:+}": round(self_delta[band][facet][k], 4) for k in DOSE}}
        band = max(P.BANDS, key=lambda b: abs(per_band[b]["slope"]))
        slope = per_band[band]["slope"]
        rho = per_band[band]["rho"]
        sign = 1 if slope >= 0 else -1
        aslope = abs(slope)
        if aslope >= STRONG_SLOPE and abs(rho) >= MONO:
            verdict = "strong"
        elif aslope >= WEAK_SLOPE and abs(rho) >= MONO:
            verdict = "weak"
        else:
            verdict = "broken"
        report[facet] = {"band": band, "sign": sign, "self_slope": round(slope, 4),
                         "self_rho": round(rho, 3), "verdict": verdict, "bands": per_band}

    # ---- summary ----
    order = {"strong": 0, "weak": 1, "broken": 2}
    print(f"\n=== TIER-0 VIABILITY SCAN (27 facets, {time.time()-t0:.0f}s) ===")
    print(f"  {'facet':24} {'band':5} {'sign':>4} {'slope':>8} {'rho':>6}  verdict")
    for facet in sorted(F.FACET_ORDER, key=lambda f: (order[report[f]['verdict']], -abs(report[f]['self_slope']))):
        r = report[facet]
        flag = " *CORE" if facet in F.CORE5 else ""
        print(f"  {facet:24} {r['band']:5} {r['sign']:+4d} {r['self_slope']:+8.3f} "
              f"{r['self_rho']:+6.2f}  {r['verdict']}{flag}")
    counts = defaultdict(int)
    for f in F.FACET_ORDER:
        counts[report[f]["verdict"]] += 1
    print(f"\n  strong={counts['strong']}  weak={counts['weak']}  broken={counts['broken']}  "
          f"(usable = strong+weak = {counts['strong']+counts['weak']}/27)")

    os.makedirs(OUT, exist_ok=True)
    out = {"stage": "tier0_viability_27", "dose": DOSE, "prime": PRIME,
           "thresholds": {"strong_slope": STRONG_SLOPE, "weak_slope": WEAK_SLOPE, "mono": MONO},
           "facets": report, "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, "viability27.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 7, "stage": "tier0_viability_27",
                             "counts": dict(counts),
                             "verdicts": {f: report[f]["verdict"] for f in F.FACET_ORDER}}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, 'viability27.json')} + metrics.jsonl")


if __name__ == "__main__":
    main()
