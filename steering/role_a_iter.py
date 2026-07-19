"""
role_a_iter.py — iteration-002: role-A item loop on the 5 validated facets (§7).

Pipeline:
  1. Administer the role-A candidate items (itembank.py) + the SPI-25 subset to the SAME steered
     persona battery (personas.py), band-grouped, fast KV-cache readout.
  2. Per-item θ-loading (Pearson of imposed θ_f vs keyed item EV over single-facet-f TRAIN personas)
     and M4 cross-loading (max |Pearson(θ_g, EV)| over off-target g).
  3. Greedy-select top-N items/facet by  loading − w_disc·cross_loading.
  4. Held-out evaluation on TEST personas:
       M2_f = Pearson(candidate-form_f, SPI-135_f)   — convergent validity (the Tier-1 metric)
       M3_f = Pearson(θ_f, candidate-form_f)          — recovery of the imposed trait
     reported for the SELECTED form and the FULL bank (to show selection's effect).
  5. Log metrics.jsonl (iter 2) + best-itembank.json. Iteration prose written separately.
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
from statistics import mean
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from administer import (administer_spi, score_scales, load_spi, keyed_items_for,
                        _make_scorer, calibration_cached)
from extract_vectors import VEC_DIR
import personas as P
import itembank as IB

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
ITER = 2
N_SELECT = 5          # items kept per facet (matches SPI scale length)
W_DISC = 0.4          # discriminant penalty in selection (fitness w_disc)


def keyed(ev, key):
    """Sign-correct an item EV: forward = EV, reverse = 7 - EV."""
    return ev if key == +1 else 7.0 - ev


def main():
    spi, spi_items = load_spi()
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    calib = calibration_cached(_make_scorer(base, tok, device))
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in IB.FACETS}

    # combined administration set: SPI-25 subset (for the backbone score) + 50 candidate items
    spi_qids = keyed_items_for(spi["keys"], IB.FACETS)
    cand = IB.items_dict()
    combined = {**{q: spi_items[q] for q in spi_qids}, **cand}
    print(f"administer set: {len(spi_qids)} SPI + {len(cand)} candidate = {len(combined)} items", flush=True)

    personas = P.build_battery()
    train, test = P.split(personas)
    by_band = defaultdict(list)
    for p in personas:
        by_band[p["band"]].append(p)

    rows = {}
    t0 = time.time(); done = 0
    for band, plist in by_band.items():
        cmodel = ControlModel(base, P.BANDS[band])
        for p in plist:
            cmodel.reset()
            cv = P.combined_vector(p["theta"], vectors)
            if cv is not None:
                cmodel.set_control(cv, 1.0)
            ev = administer_spi(cmodel, tok, spi, combined, device=device, log_every=0,
                                calib=calib, subset=None, fast=True)
            cmodel.reset()
            spi_scores = score_scales({q: ev[q] for q in spi_qids}, spi["keys"])
            cand_ev = {it["id"]: ev[it["id"]]["ev"] for it in IB.ITEMS}
            rows[p["id"]] = {"persona": p, "spi": spi_scores, "cand": cand_ev}
            done += 1
            if done % 5 == 0:
                print(f"  [{done}/{len(personas)}] {time.time()-t0:.0f}s", flush=True)
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # ---- per-item θ-loading + M4 cross-loading on TRAIN single personas ----
    def single(pool, facet):
        return [p for p in pool if p["kind"] == "single" and facet in p["theta"]]

    train_single = {f: single(train, f) for f in IB.FACETS}
    item_stats = {}
    for it in IB.ITEMS:
        f = it["facet"]
        own = train_single[f]
        load = pearsonr([p["theta"][f] for p in own],
                        [keyed(rows[p["id"]]["cand"][it["id"]], it["key"]) for p in own])[0]
        cross = 0.0
        for g in IB.FACETS:
            if g == f:
                continue
            og = train_single[g]
            r = pearsonr([p["theta"][g] for p in og],
                         [keyed(rows[p["id"]]["cand"][it["id"]], it["key"]) for p in og])[0]
            cross = max(cross, abs(r))
        item_stats[it["id"]] = {"facet": f, "key": it["key"], "loading": round(float(load), 3),
                                "cross": round(float(cross), 3),
                                "score": round(float(load - W_DISC * cross), 3)}

    # ---- greedy select top-N per facet ----
    selected = {}
    for f in IB.FACETS:
        ranked = sorted([it for it in IB.ITEMS if it["facet"] == f],
                        key=lambda it: item_stats[it["id"]]["score"], reverse=True)
        selected[f] = [it["id"] for it in ranked[:N_SELECT]]

    # ---- form facet score per persona (mean keyed EV over a facet's item set) ----
    def form_score(pid, ids, f):
        ks = {it["id"]: it["key"] for it in IB.ITEMS if it["facet"] == f}
        vals = [keyed(rows[pid]["cand"][i], ks[i]) for i in ids]
        return mean(vals) if vals else None

    def evaluate(pool, item_sets, tag):
        res = {}
        for f in IB.FACETS:
            forms = [form_score(p["id"], item_sets[f], f) for p in pool]
            spis = [rows[p["id"]]["spi"].get(f) for p in pool]
            thetas = [p["theta"].get(f, 0.0) for p in pool]
            m2 = pearsonr(forms, spis)[0] if len(set(spis)) > 1 else None
            m3 = pearsonr(thetas, forms)[0] if len(set(thetas)) > 1 else None
            res[f] = {"M2": None if m2 is None else round(float(m2), 3),
                      "M3": None if m3 is None else round(float(m3), 3)}
        return res

    full_sets = {f: [it["id"] for it in IB.ITEMS if it["facet"] == f] for f in IB.FACETS}
    sel_te = evaluate(test, selected, "selected/test")
    full_te = evaluate(test, full_sets, "full/test")
    sel_all = evaluate(personas, selected, "selected/all")

    def meanM(d, k):
        xs = [d[f][k] for f in IB.FACETS if d[f][k] is not None]
        return round(mean(xs), 3) if xs else None

    print(f"\n=== ITER {ITER} — role-A item loop (5 facets), held-out TEST ===")
    print(f"{'facet':12} {'M2_sel':>7} {'M3_sel':>7} | {'M2_full':>7} {'M3_full':>7}  kept")
    for f in IB.FACETS:
        kept = ",".join(selected[f])
        print(f"{f:12} {str(sel_te[f]['M2']):>7} {str(sel_te[f]['M3']):>7} | "
              f"{str(full_te[f]['M2']):>7} {str(full_te[f]['M3']):>7}  {kept}")
    print(f"{'MEAN':12} {str(meanM(sel_te,'M2')):>7} {str(meanM(sel_te,'M3')):>7} | "
          f"{str(meanM(full_te,'M2')):>7} {str(meanM(full_te,'M3')):>7}")
    print(f"\nTier-1 check: mean(M2_sel)={meanM(sel_te,'M2')} (target ≥0.60); "
          f"facets M2≥0.40: {sum(1 for f in IB.FACETS if (sel_te[f]['M2'] or 0) >= 0.4)}/5")

    out = {"iter": ITER, "n_select": N_SELECT, "w_disc": W_DISC,
           "selected": selected, "item_stats": item_stats,
           "eval": {"selected_test": sel_te, "full_test": full_te, "selected_all": sel_all},
           "mean": {"M2_sel_test": meanM(sel_te, "M2"), "M3_sel_test": meanM(sel_te, "M3"),
                    "M2_full_test": meanM(full_te, "M2")},
           "seconds": round(time.time() - t0)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "best-itembank.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": ITER, "stage": "roleA_items_5facets",
                             "mean_M2_sel_test": meanM(sel_te, "M2"),
                             "mean_M3_sel_test": meanM(sel_te, "M3"),
                             "facets": {f: sel_te[f] for f in IB.FACETS}}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, 'best-itembank.json')} + metrics.jsonl")


if __name__ == "__main__":
    main()
