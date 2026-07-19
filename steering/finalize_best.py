"""
finalize_best.py — merge iter-7 viability with iter-8 v2 winners → viability27_best.json.

For the BEST_V2 facets (where re-authored items won), override band/sign/verdict with the v2
profile measured in reauthor_test.json; every other facet keeps its iter-7 v1 profile.
"""
import os, json
import facets27 as F

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
via = json.load(open(os.path.join(OUT, "viability27.json")))
rea = json.load(open(os.path.join(OUT, "reauthor_test.json")))["report"]

facets = via["facets"]
for f in F.BEST_V2:
    v2 = rea[f]["v2"]
    facets[f] = {**facets[f], "band": v2["band"], "sign": 1 if v2["slope"] >= 0 else -1,
                 "self_slope": v2["slope"], "self_rho": v2["rho"], "verdict": v2["verdict"],
                 "item_version": "v2"}
for f in facets:
    facets[f].setdefault("item_version", "v1")

via["stage"] = "viability27_best"
via["best_v2"] = F.BEST_V2
json.dump(via, open(os.path.join(OUT, "viability27_best.json"), "w"), indent=2)

usable = [f for f in F.FACET_ORDER if facets[f]["verdict"] in ("strong", "weak")]
print(f"viability27_best.json written. usable = {len(usable)}/27")
print("broken:", [f for f in F.FACET_ORDER if facets[f]["verdict"] == "broken"])
for f in F.BEST_V2:
    print(f"  {f}: band={facets[f]['band']} sign={facets[f]['sign']:+d} "
          f"slope={facets[f]['self_slope']} verdict={facets[f]['verdict']} (v2)")
