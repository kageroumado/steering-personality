"""
discriminant.py — slope-based discriminant analysis (post-hoc on saved iter-3/4 readouts).

WHY: the cross-loading metric used since iter-1 is |Pearson(theta_g, readout)| over a per-facet
DOSE SERIES (7 collinear points). Pearson is scale-invariant — it measures monotonic TIGHTNESS,
not effect SIZE — so any low-noise monotonic nudge saturates to |r|~1. That is why iter-4 shows
own-loading 0.09 yet cross 1.0 for SelfControl: nonsense under correlation, diagnostic of the
wrong metric. Discriminant validity needs SLOPES (readout change per unit theta), not correlations.

This recomputes, from the saved JSONs (no model), the 5x5 influence matrix
    L[g][f] = mean over facet-f items of  slope(readout_item ~ theta_g)   (single-g personas)
and a self-potency-normalised version
    Lhat[g][f] = L[g][f] / |L[g][g]|     (g's pull on f's items, in units of g's pull on its own)
so Lhat[g][g] = 1. Discriminant validity for facet f = own row dominates column f:
    DR_f = |L[f][f]| / max_{g!=f} |L[g][f]|   (>1 = own vector moves f's items most).
Reported for forced-choice (delta) and being-frame Likert (raw EV, keyed).
"""
from __future__ import annotations
import os, json
from statistics import mean
import numpy as np
import itembank as IB
import forced_choice as FC

from paths import OUT
FACETS = IB.FACETS


def theta_of(pid):
    # "single:Facet:+1.0" -> (facet, k)
    _, facet, k = pid.split(":")
    return facet, float(k)


def slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if np.ptp(xs) == 0:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


def influence_matrix(readout_items, item_facet, personas_by_facet):
    """readout_items[pid][item_id] -> scalar. Returns L[g][f] mean item-slope, and per-item."""
    items_of = {f: [i for i in item_facet if item_facet[i] == f] for f in FACETS}
    L = {g: {} for g in FACETS}
    for g in FACETS:
        ps = personas_by_facet[g]
        ks = [theta_of(p)[1] for p in ps]
        for f in FACETS:
            sl = []
            for it in items_of[f]:
                ys = [readout_items[p][it] for p in ps]
                sl.append(slope(ks, ys))
            L[g][f] = mean(abs(s) for s in sl) * (1 if f != g else 1)
            # keep sign for diagonal interpretability via a separate signed pass
    return L, items_of


def signed_diag(readout_items, items_of, personas_by_facet):
    d = {}
    for f in FACETS:
        ps = personas_by_facet[f]
        ks = [theta_of(p)[1] for p in ps]
        sl = [slope(ks, [readout_items[p][it] for p in ps]) for it in items_of[f]]
        d[f] = mean(sl)
    return d


def report(name, readout_items, item_facet, personas_by_facet):
    L, items_of = influence_matrix(readout_items, item_facet, personas_by_facet)
    diag = signed_diag(readout_items, items_of, personas_by_facet)
    print(f"\n=== {name} — |slope| influence matrix  L[steer g][items of f] ===")
    print(f"{'g\\f':>12} " + " ".join(f"{f[:5]:>7}" for f in FACETS) + "   selfpot")
    for g in FACETS:
        row = " ".join(f"{L[g][f]:7.3f}" for f in FACETS)
        print(f"{g:>12} {row}   {abs(diag[g]):7.3f}")
    # normalised: Lhat[g][f] = L[g][f]/|L[g][g]|
    print(f"\n--- self-potency-normalised  Lhat[g][f] = L[g][f]/|L[g][g]|  (diag=1) ---")
    print(f"{'g\\f':>12} " + " ".join(f"{f[:5]:>7}" for f in FACETS))
    DRnum = {f: abs(L[f][f]) for f in FACETS}
    for g in FACETS:
        denom = abs(L[g][g]) or 1e-9
        row = " ".join(f"{L[g][f]/denom:7.2f}" for f in FACETS)
        print(f"{g:>12} {row}")
    print(f"\n--- discriminant ratio  DR_f = |L[f][f]| / max_(g!=f)|L[g][f]|  (>1 = own dominates) ---")
    drs = {}
    for f in FACETS:
        off = max(abs(L[g][f]) for g in FACETS if g != f)
        dr = (abs(L[f][f]) / off) if off > 0 else float("inf")
        drs[f] = dr
        print(f"  {f:12} own={abs(L[f][f]):.3f}  max_off={off:.3f}  DR={dr:.2f}  "
              f"{'OWN DOMINATES' if dr > 1 else 'LEAKS'}")
    print(f"  MEAN DR = {mean(v for v in drs.values() if v != float('inf')):.2f}; "
          f"facets own-dominant: {sum(1 for v in drs.values() if v > 1)}/5")
    return {"L": L, "diag": diag, "DR": drs}


def keyed(ev, key):
    return ev if key == +1 else 7.0 - ev


def main():
    out = {}

    # ---- forced-choice (delta) ----
    fc = json.load(open(os.path.join(OUT, "forced_choice.json")))
    delta = fc["delta"]
    fc_item_facet = {it["id"]: it["facet"] for it in FC.ITEMS}
    pbf = {f: [pid for pid in delta if theta_of(pid)[0] == f] for f in FACETS}
    out["forced_choice"] = report("FORCED-CHOICE (pole-contrast Δ)", delta, fc_item_facet, pbf)

    # ---- being-frame Likert (keyed raw EV) ----
    ft = json.load(open(os.path.join(OUT, "frame_test.json")))
    for frame in ("being", "narrate"):
        raw = ft["raw_ev"][frame]
        keymap = {it["id"]: it["key"] for it in IB.ITEMS}
        keyed_items = {pid: {iid: keyed(ev, keymap[iid]) for iid, ev in d.items()}
                       for pid, d in raw.items()}
        lik_item_facet = {it["id"]: it["facet"] for it in IB.ITEMS}
        pbf2 = {f: [pid for pid in raw if theta_of(pid)[0] == f] for f in FACETS}
        out[f"likert_{frame}"] = report(f"LIKERT keyed EV — {frame} frame", keyed_items,
                                        lik_item_facet, pbf2)

    def jsonable(d):
        return {k: ({kk: (vv if vv != float("inf") else 1e9) for kk, vv in v.items()}
                    if isinstance(v, dict) else v) for k, v in d.items()}
    dump = {k: {"DR": jsonable(v["DR"]),
                "diag": v["diag"],
                "L": v["L"]} for k, v in out.items()}
    json.dump(dump, open(os.path.join(OUT, "discriminant.json"), "w"), indent=2)
    print(f"\nsaved → {os.path.join(OUT, 'discriminant.json')}")


if __name__ == "__main__":
    main()
