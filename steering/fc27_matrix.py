"""
fc27_matrix.py — breadth 5→27 (D13), Pass 2: the master single-facet response matrix.

For every facet, at its Tier-0-chosen BAND and SIGN (from viability27.json), administer ALL
135 forced-choice items across a dose grid. This is the raw data the iter-6 modeling needs:
    Δ_i(facet, θ) = lp(hi_i) - lp(lo_i)      for every item i in the full 135-item bank
where θ is the SIGN-CORRECTED trait dose (θ>0 ⇒ more of the trait), realized by injecting
the facet's vector at coeff = sign·θ.

From this single pass we fit the linear item-response matrix S (135×27) and test additivity /
covariance recovery in covariance27.py — exactly the iter-6 pipeline, at 27D scale.

Memory rule (D6): group facets by band, wrap each band ONCE, iterate facets/doses inside.
Output → research/loop/fc27_matrix.json : delta[facet][θ][item_id].
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
from extract_vectors import VEC_DIR
import facets27 as F
import personas as P

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
PRIME = "I'll be honest about myself.\n\n"
DOSE = [-2.0, -1.0, 0.0, 1.0, 2.0]   # sign-corrected trait dose θ


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", choices=["v1", "best", "all"], default="v1")
    args = ap.parse_args()
    via_name = "viability27.json" if args.items == "v1" else "viability27_best.json"
    out_name = "fc27_matrix.json" if args.items == "v1" else f"fc27_matrix_{args.items}.json"
    ITEMS = F.fc_items(args.items)
    print(f"item set = {args.items} ({len(ITEMS)} items); viability = {via_name}", flush=True)

    via = json.load(open(os.path.join(OUT, via_name)))["facets"]
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf"))
               for f in F.FACET_ORDER}

    by_band = defaultdict(list)
    for f in F.FACET_ORDER:
        by_band[via[f]["band"]].append(f)

    # delta[facet][θstr][item_id]
    delta = {f: {} for f in F.FACET_ORDER}
    t0 = time.time(); done = 0
    total = len(F.FACET_ORDER) * len(DOSE)
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
                for it in ITEMS:
                    s = scorer.score(it["stem"], [it["hi"], it["lo"]])
                    row[it["id"]] = s[0]["logprob_sum"] - s[1]["logprob_sum"]
                delta[facet][f"{theta:+}"] = row
                del scorer
                cmodel.reset()
                done += 1
                if done % 5 == 0:
                    print(f"  [{done}/{total}] {time.time()-t0:.0f}s", flush=True)
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    out = {"stage": f"fc27_matrix_{args.items}", "dose": DOSE, "prime": PRIME,
           "bands": {f: via[f]["band"] for f in F.FACET_ORDER},
           "signs": {f: via[f]["sign"] for f in F.FACET_ORDER},
           "item_order": [it["id"] for it in ITEMS],
           "item_facet": {it["id"]: it["facet"] for it in ITEMS},
           "delta": delta, "seconds": round(time.time() - t0)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, out_name), "w"), indent=2)
    print(f"\nsaved → {os.path.join(OUT, out_name)}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
