"""
costeer.py — iteration-005b: co-steer the Being stance vector with each facet.

Tests Kiri's self-attractor hypothesis at the activation level: does adding a "being"
(first-person inhabiting) push alongside the facet vector make the steered self COMMIT, so its
forced-choice answers become content-specific (discriminant) rather than generic-affirmation?

For each facet's single-facet dose personas, administer the forced-choice pole-contrast items
under control = facet_vec·θ + Being·(λ·SIGN), sweeping λ. The Being push is applied to EVERY
administration (it's a readout-time stance, independent of which facet is steered), so the 5×5
slope matrix at each λ is comparable. Report, per λ:
    M3_f   = corr(θ_f, form_f)                              — sensitivity (own-facet recovery)
    DR_f   = |slope(f→f items)| / max_{g≠f}|slope(g→f items)| — specificity (own dominates?)
PREDICTION (self-attractor): DR rises with λ (self commits → specific answers) without
collapsing M3. NULL: DR flat/falls → stance is not the lever; entanglement is representational.

SIGN: set BEING_SIGN from stance_vector.py's first-person sanity check (+ if k>0 is more
first-person, else -1).
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
from statistics import mean
import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
from extract_vectors import VEC_DIR
import personas as P
import forced_choice as FC

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
ITER = 5
FACETS = FC.IB.FACETS
LAMBDAS = [0.0, 0.5, 1.0, 2.0]
BEING_SIGN = +1.0   # overwritten from CLI; set by stance sanity check


def theta_of(pid):
    _, facet, k = pid.split(":")
    return facet, float(k)


def slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if np.ptp(xs) == 0:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def build_singles():
    out = []
    for f in FACETS:
        for k in P.DOSE[f]:
            out.append({"id": f"single:{f}:{k:+}", "band": P.FACET_BAND[f],
                        "theta": {f: k}, "facet": f, "k": k})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", type=float, default=BEING_SIGN)
    args = ap.parse_args()
    sign = args.sign

    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...; Being sign={sign:+.0f}", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in FACETS}
    being = ControlVector.import_gguf(os.path.join(VEC_DIR, "Being.gguf"))

    personas = build_singles()
    by_band = defaultdict(list)
    for p in personas:
        by_band[p["band"]].append(p)

    # delta[lam][pid][item_id]
    delta = {lam: {} for lam in LAMBDAS}
    t0 = time.time(); done = 0; total = len(personas) * len(LAMBDAS)
    for band, plist in by_band.items():
        cmodel = ControlModel(base, P.BANDS[band])
        for p in plist:
            facet_cv = P.combined_vector(p["theta"], vectors)
            for lam in LAMBDAS:
                cmodel.reset()
                cv = facet_cv
                if lam != 0.0:
                    term = being * (lam * sign)
                    cv = term if cv is None else cv + term
                if cv is not None:
                    cmodel.set_control(cv, 1.0)
                scorer = PrefixScorer(cmodel, tok, FC.PRIME, n_options=2, device=device)
                d = {}
                for it in FC.ITEMS:
                    s = scorer.score(it["stem"], [it["hi"], it["lo"]])
                    d[it["id"]] = s[0]["logprob_sum"] - s[1]["logprob_sum"]
                delta[lam][p["id"]] = d
                del scorer
                cmodel.reset()
                done += 1
                if done % 20 == 0:
                    print(f"  [{done}/{total}] {time.time()-t0:.0f}s", flush=True)
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # ---- metrics per lambda ----
    singles = {f: [p for p in personas if p["facet"] == f] for f in FACETS}
    fbi = {f: [it for it in FC.ITEMS if it["facet"] == f] for f in FACETS}

    def corr(xs, ys):
        if len({round(x, 9) for x in xs}) < 2 or len({round(y, 9) for y in ys}) < 2:
            return None
        return float(pearsonr(xs, ys)[0])

    results = {}
    for lam in LAMBDAS:
        D = delta[lam]
        # 5x5 |slope| matrix
        L = {g: {} for g in FACETS}
        for g in FACETS:
            ps = singles[g]; ks = [p["k"] for p in ps]
            for f in FACETS:
                L[g][f] = mean(abs(slope(ks, [D[p["id"]][it["id"]] for p in ps])) for it in fbi[f])
        per = {}
        for f in FACETS:
            ps = singles[f]
            # M3: z-scored form
            zforms = []
            for p in ps:
                vals = []
                for it in fbi[f]:
                    col = [D[q["id"]][it["id"]] for q in ps]
                    m, sd = mean(col), (np.std(col) or 1.0)
                    vals.append((D[p["id"]][it["id"]] - m) / sd)
                zforms.append(mean(vals))
            m3 = corr([p["k"] for p in ps], zforms)
            off = max(abs(L[g][f]) for g in FACETS if g != f)
            dr = (abs(L[f][f]) / off) if off > 0 else float("inf")
            per[f] = {"M3": None if m3 is None else round(m3, 3),
                      "own_slope": round(abs(L[f][f]), 3), "max_off": round(off, 3),
                      "DR": round(dr, 3) if dr != float("inf") else 1e9}
        mDR = mean(per[f]["DR"] for f in FACETS if per[f]["DR"] < 1e9)
        mM3 = mean(abs(per[f]["M3"]) for f in FACETS if per[f]["M3"] is not None)
        n_dom = sum(1 for f in FACETS if per[f]["DR"] > 1)
        results[lam] = {"per_facet": per, "mean_DR": round(mDR, 3),
                        "mean_absM3": round(mM3, 3), "own_dominant": n_dom,
                        "L": {g: {f: round(L[g][f], 3) for f in FACETS} for g in FACETS}}

    print(f"\n=== ITER {ITER} — co-steer Being (sign {sign:+.0f}); DR & |M3| vs λ ===")
    print(f"  {'λ':>5} {'meanDR':>7} {'own>1':>6} {'mean|M3|':>9}   per-facet DR")
    for lam in LAMBDAS:
        r = results[lam]
        drs = " ".join(f"{f[:4]}={r['per_facet'][f]['DR'] if r['per_facet'][f]['DR']<1e9 else 'inf'}" for f in FACETS)
        print(f"  {lam:>5} {r['mean_DR']:>7} {r['own_dominant']:>6} {r['mean_absM3']:>9}   {drs}")
    base_dr = results[0.0]["mean_DR"]; best_lam = max(LAMBDAS, key=lambda l: results[l]["mean_DR"])
    print(f"\nλ=0 meanDR={base_dr} → best λ={best_lam} meanDR={results[best_lam]['mean_DR']}  "
          f"({'Being SHARPENS specificity' if results[best_lam]['mean_DR'] > base_dr + 0.05 else 'no specificity gain — stance is not the lever'})")

    out = {"iter": ITER, "being_sign": sign, "lambdas": LAMBDAS, "results": results,
           "delta": {str(l): delta[l] for l in LAMBDAS}, "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, "costeer.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": ITER, "stage": "costeer_being",
                             "being_sign": sign,
                             "by_lambda": {str(l): {"mean_DR": results[l]["mean_DR"],
                                                    "mean_absM3": results[l]["mean_absM3"],
                                                    "own_dominant": results[l]["own_dominant"]}
                                           for l in LAMBDAS}}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, 'costeer.json')} + metrics.jsonl  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
