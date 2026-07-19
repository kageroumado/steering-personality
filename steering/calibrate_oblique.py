"""
calibrate_oblique.py — iter-9 (D15→product): can OBLIQUE questions be calibrated on the model?

Administer the oblique product items (`oblique_items.OBLIQUE`) as the readout while steering each
of the 27 facets one at a time (best band/sign from viability27_best.json), across a dose grid.
Measure the loading

    L[item][facet] = slope( Δ_item ~ θ_facet ),   Δ_item = lp(hi) - lp(lo)

i.e. how much each fun, trait-unnamed question's answer shifts when we dial a given trait. A
usable oblique item loads strongly and CLEANLY (single-facet-dominant) on some facet.

Reports: per-item dominant facet + purity + whether it matches the a-priori guess; per-facet
coverage (does any oblique item reach this trait?); the obliqueness/signal picture.

Memory rule (D6): group facets by band, wrap once. Output → research/loop/oblique_calibration.json.
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
import oblique_items as OI

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
PRIME = "I'll be honest about myself.\n\n"
DOSE = [-2.0, -1.0, 0.0, 1.0, 2.0]


def ols_slope(xs, ys):
    x = np.asarray(xs, float) - np.mean(xs)
    d = float((x * x).sum())
    return float((x * (np.asarray(ys, float) - np.mean(ys))).sum() / d) if d else 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=["b1", "b2", "pool"], default="b1")
    args = ap.parse_args()
    OBLIQUE = OI.BANKS[args.bank]
    out_name = "oblique_calibration.json" if args.bank == "b1" else f"oblique_calibration_{args.bank}.json"
    print(f"bank = {args.bank} ({len(OBLIQUE)} items)", flush=True)

    via = json.load(open(os.path.join(OUT, "viability27_best.json")))["facets"]
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in F.FACET_ORDER}

    by_band = defaultdict(list)
    for f in F.FACET_ORDER:
        by_band[via[f]["band"]].append(f)

    # delta[facet][k][item_id]
    delta = {f: {} for f in F.FACET_ORDER}
    t0 = time.time(); done = 0; total = len(F.FACET_ORDER) * len(DOSE)
    for band, facets in by_band.items():
        cmodel = ControlModel(base, P.BANDS[band])
        for facet in facets:
            sign = via[facet]["sign"]
            for theta in DOSE:
                cmodel.reset()
                coeff = sign * theta
                if coeff != 0.0:
                    cmodel.set_control(vectors[facet], float(coeff))
                scorer = PrefixScorer(cmodel, tok, PRIME, n_options=2, device=device)
                row = {}
                for it in OBLIQUE:
                    s = scorer.score(it["stem"], [it["hi"], it["lo"]])
                    row[it["id"]] = s[0]["logprob_sum"] - s[1]["logprob_sum"]
                delta[facet][f"{theta:+}"] = row
                del scorer
                cmodel.reset()
                done += 1
                if done % 10 == 0:
                    print(f"  [{done}/{total}] {time.time()-t0:.0f}s", flush=True)
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # ---- loading matrix L[item][facet] = slope, plus rho (monotonicity) ----
    usable = [f for f in F.FACET_ORDER if via[f]["verdict"] in ("strong", "weak")]
    L = {it["id"]: {} for it in OBLIQUE}
    Rho = {it["id"]: {} for it in OBLIQUE}
    for it in OBLIQUE:
        for f in F.FACET_ORDER:
            ys = [delta[f][f"{k:+}"][it["id"]] for k in DOSE]
            L[it["id"]][f] = ols_slope(DOSE, ys)
            r = spearmanr(DOSE, ys)[0]
            Rho[it["id"]][f] = 0.0 if r != r else float(r)

    # ---- per-item analysis (restrict dominant search to USABLE facets) ----
    item_report = {}
    guess_hits = 0
    for it in OBLIQUE:
        loads = {f: L[it["id"]][f] for f in usable}
        ranked = sorted(loads, key=lambda f: -abs(loads[f]))
        top, second = ranked[0], ranked[1]
        top_s, sec_s = loads[top], loads[second]
        purity = abs(top_s) / (abs(sec_s) + 1e-9)         # >1.5 ≈ single-facet-dominant
        clean = abs(top_s) >= 0.30 and abs(Rho[it["id"]][top]) >= 0.80
        guess_hit = (top == it["guess"])
        guess_hits += guess_hit
        item_report[it["id"]] = {"guess": it["guess"], "dominant": top,
                                 "dom_slope": round(top_s, 3), "dom_rho": round(Rho[it["id"]][top], 3),
                                 "second": second, "sec_slope": round(sec_s, 3),
                                 "purity": round(purity, 2), "clean": clean, "guess_hit": guess_hit}

    # ---- per-facet coverage: best oblique item reaching each usable facet ----
    facet_cov = {}
    for f in usable:
        best_it = max(OBLIQUE, key=lambda it: abs(L[it["id"]][f]))
        s = L[best_it["id"]][f]
        facet_cov[f] = {"best_item": best_it["id"], "slope": round(s, 3),
                        "rho": round(Rho[best_it["id"]][f], 3),
                        "reached": abs(s) >= 0.30 and abs(Rho[best_it["id"]][f]) >= 0.80}

    n_clean = sum(1 for v in item_report.values() if v["clean"])
    n_reached = sum(1 for v in facet_cov.values() if v["reached"])
    print(f"\n=== OBLIQUE CALIBRATION ({len(OBLIQUE)} items × {len(F.FACET_ORDER)} facets, {time.time()-t0:.0f}s) ===")
    print(f"\n[PER ITEM]  {'id':5} {'guess':18} {'dominant':18} {'slope':>7} {'pure':>5} {'rho':>5}  hit clean")
    for it in OBLIQUE:
        r = item_report[it["id"]]
        print(f"  {it['id']:5} {r['guess']:18} {r['dominant']:18} {r['dom_slope']:+7.2f} "
              f"{r['purity']:5.1f} {r['dom_rho']:+5.2f}  {'Y' if r['guess_hit'] else '.'}  "
              f"{'Y' if r['clean'] else '.'}")
    print(f"\n  guess hit-rate: {guess_hits}/{len(OBLIQUE)}  |  clean items: {n_clean}/{len(OBLIQUE)}")
    print(f"\n[FACET COVERAGE]  facets reached by ≥1 clean oblique item: {n_reached}/{len(usable)}")
    miss = [f for f in usable if not facet_cov[f]["reached"]]
    print(f"  not reached: {miss}")

    out = {"stage": "oblique_calibration", "n_items": len(OBLIQUE), "dose": DOSE,
           "usable_facets": usable, "guess_hit_rate": round(guess_hits / len(OBLIQUE), 3),
           "n_clean_items": n_clean, "n_facets_reached": n_reached,
           "item_report": item_report, "facet_coverage": facet_cov,
           "L": L, "rho": Rho, "delta": delta, "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, out_name), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 9, "stage": "oblique_calibration",
                             "guess_hit_rate": round(guess_hits / len(OBLIQUE), 3),
                             "n_clean_items": n_clean, "n_facets_reached": n_reached,
                             "n_usable": len(usable)}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, out_name)} + metrics.jsonl  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
