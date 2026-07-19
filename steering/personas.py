"""
personas.py — synthetic persona battery + first loop measurement (§5, §7).

A persona = a target z-profile over facets realized by adding sign-aligned steering vectors
at magnitudes k (the ground-truth θ). v1 covers the 3 Tier-0-passing facets, grouped by their
per-facet injection band (D6): SelfControl@late, Trust@mid, Sociability@mid.

  * single-facet dose personas  — one facet at k ∈ {-2,-1,-0.5,0,0.5,1,2} (M1 dose-response)
  * within-band multi-facet      — Trust×Sociability at random k's (held-out validity; same
                                    band so vectors just sum — no cross-band key-mapping needed)

Cross-band multi-facet (SelfControl+mid) is deferred to v2: it needs a validated
directions-key→layer restriction to inject each facet only at its own band.

Memory rule (D6 caveat): wrap each band ONCE and iterate its personas — never re-wrap per
persona. Administration uses the verified fast KV-cache path.
"""
from __future__ import annotations
import os, json, time
import torch
from scipy.stats import spearmanr, pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from administer import administer_spi, score_scales, load_spi, keyed_items_for
from extract_vectors import VEC_DIR

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
ITER = 1
BANDS = {"late": list(range(-1, -16, -1)), "mid": list(range(-12, -21, -1))}
# All 5 robust facets (4 pass Tier-0 cleanly; Conformity recovered on late asymmetric sweep).
FACET_BAND = {"SelfControl": "late", "Trust": "mid", "Sociability": "mid",
              "Conformity": "late", "Adaptability": "late"}
# Per-facet USABLE dose grids (respect one-sided saturation found in build_steering /
# recover_facets): symmetric where the response is two-sided; asymmetric where one direction
# saturates so an out-of-range dose would just inject readout-ceiling noise into M3.
#   * Adaptability — up-steer saturates at the readout ceiling ~3.9; probe the clean down side.
#   * Conformity   — sign-aligned vector has sign=-1, so θ = -(raw coeff). Recovery found the
#                    clean RAW range [-1,+2]; negated that is θ ∈ [-2,+1]. The high-conformity
#                    over-steer kink is at raw -2 = θ=+2, so θ must stop at +1.
DOSE = {
    "SelfControl":  [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "Trust":        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "Sociability":  [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "Conformity":   [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
    "Adaptability": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
}
# Held-out dose indices (3 interior, evenly spread) → valid M1_test (Spearman needs ≥3).
TEST_IDX = {1, 3, 5}
# leakage / discriminant probes administered alongside the targets (M4)
LEAK = ["Anxiety", "Industry", "Compassion"]


def build_battery():
    """Deterministic battery (no RNG; reproducible). Returns list of persona dicts.

    single-facet dose personas tag their dose-index so split() can stratify within each
    facet (every facet's series lands in BOTH halves → held-out M1 is always computable).
    Multi-facet personas are within-band sums (no cross-band key-mapping needed): Trust×
    Sociability @ mid, SelfControl×Conformity @ late.
    """
    personas = []
    for f in FACET_BAND:
        for idx, k in enumerate(DOSE[f]):
            personas.append({"id": f"single:{f}:{k:+}", "band": FACET_BAND[f],
                             "theta": {f: k}, "kind": "single", "dose_idx": idx})
    # within-band multi: mid Trust×Sociability, late SelfControl×Conformity (modest k, clean range)
    for kt in (-1.5, 0.0, 1.5):
        for ks in (-1.5, 0.0, 1.5):
            if kt == 0.0 and ks == 0.0:
                continue
            personas.append({"id": f"multi:T{kt:+}:S{ks:+}", "band": "mid",
                             "theta": {"Trust": kt, "Sociability": ks}, "kind": "multi"})
    for kc in (-1.0, 0.0, 1.0):
        for ka in (-1.0, 0.0, 1.0):
            if kc == 0.0 and ka == 0.0:
                continue
            personas.append({"id": f"multi:Sc{kc:+}:Cf{ka:+}", "band": "late",
                             "theta": {"SelfControl": kc, "Conformity": ka}, "kind": "multi"})
    return personas


def split(personas):
    """Stratified deterministic split: single personas by within-facet dose index (each facet's
    series in both halves); multi personas by stride. Metrics reported on held-out test."""
    test, train = [], []
    for i, p in enumerate(personas):
        if p["kind"] == "single":
            is_test = p["dose_idx"] in TEST_IDX
        else:
            is_test = i % 3 == 0
        (test if is_test else train).append(p)
    return train, test


def combined_vector(theta, vectors):
    vec = None
    for f, k in theta.items():
        if k == 0:
            continue
        term = vectors[f] * float(k)
        vec = term if vec is None else vec + term
    return vec


def main():
    spi, items = load_spi()
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in FACET_BAND}

    # calibration on the UNSTEERED base, frozen. Bands overlap (late 17-31, mid 12-20),
    # so they CANNOT be wrapped simultaneously — process band-grouped: wrap, run, unwrap.
    from collections import defaultdict
    from administer import _make_scorer, calibration_cached
    calib = calibration_cached(_make_scorer(base, tok, device))

    subset = keyed_items_for(spi["keys"], list(FACET_BAND) + LEAK)
    personas = build_battery()
    train, test = split(personas)
    print(f"battery: {len(personas)} personas ({len(train)} train / {len(test)} test); "
          f"{len(subset)} items/persona", flush=True)

    by_band = defaultdict(list)
    for p in personas:
        by_band[p["band"]].append(p)

    rows = {}
    t0 = time.time()
    done = 0
    for band, plist in by_band.items():
        cmodel = ControlModel(base, BANDS[band])
        for p in plist:
            cmodel.reset()
            cv = combined_vector(p["theta"], vectors)
            if cv is not None:
                cmodel.set_control(cv, 1.0)
            ev = administer_spi(cmodel, tok, spi, items, device=device, log_every=0,
                                calib=calib, subset=subset, fast=True)
            cmodel.reset()
            rows[p["id"]] = {"persona": p, "scores": score_scales(ev, spi["keys"])}
            done += 1
            if done % 5 == 0:
                print(f"  [{done}/{len(personas)}] {time.time()-t0:.0f}s", flush=True)
        cmodel.unwrap()

    # ---- metrics on HELD-OUT test (and full, for reference) ----
    def facet_metrics(pool, facet):
        single = [(p["theta"][facet], rows[p["id"]]["scores"][facet])
                  for p in pool if p["kind"] == "single" and facet in p["theta"]]
        m1 = spearmanr([k for k, _ in single], [s for _, s in single])[0] if len(single) >= 3 else None
        allp = [(p["theta"].get(facet, 0.0), rows[p["id"]]["scores"][facet]) for p in pool]
        xs = [x for x, _ in allp]
        m3 = pearsonr(xs, [s for _, s in allp])[0] if len(set(xs)) >= 2 else None
        return m1, m3

    report = {"n_personas": len(personas), "n_train": len(train), "n_test": len(test),
              "facets": {}, "seconds": time.time() - t0}
    print(f"\n=== LOOP ITER {ITER} — M1/M3 on SPI backbone (held-out test) ===")
    for f in FACET_BAND:
        m1_te, m3_te = facet_metrics(test, f)
        m1_all, m3_all = facet_metrics(personas, f)
        report["facets"][f] = {"M1_test": m1_te, "M3_test": m3_te,
                               "M1_all": m1_all, "M3_all": m3_all}
        def fmt(x): return "  n/a" if x is None else f"{x:+.2f}"
        print(f"  {f:12} M1(test)={fmt(m1_te)} M3(test)={fmt(m3_te)} | "
              f"M1(all)={fmt(m1_all)} M3(all)={fmt(m3_all)}")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"report": report,
               "rows": {k: {"theta": v["persona"]["theta"], "scores": v["scores"]}
                        for k, v in rows.items()}},
              open(os.path.join(OUT, f"loop_iter{ITER}.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": ITER, "stage": "spi_backbone_5facets",
                             "facets": report["facets"]}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, f'loop_iter{ITER}.json')} + metrics.jsonl")


if __name__ == "__main__":
    main()
