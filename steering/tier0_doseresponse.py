"""
tier0_doseresponse.py — the Tier-0 gate (§6, §15 step 4).

Steer each robust facet at k ∈ sweep, administer SPI-135 each time, and test whether the
target facet's SPI readout moves monotonically with k. GATE: Spearman ρ(k, target_score)
≥ 0.80 on ≥3 facets. Also logs OFF-target leakage (does steering Adaptability move Trust?)
as a first look at M4 / trait coupling (Bhandari 2026).

If it passes → linear steering works on this stack; build the persona battery and loop.
If it fails on all → STOP, write up the null (§15 guardrail). Don't fake progress.
"""
from __future__ import annotations
import os, json, time, argparse
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from administer import administer_spi, score_scales, load_spi, keyed_items_for, compute_calibration
from extract_vectors import LAYER_IDS, VEC_DIR

from paths import OUT as OUT_DIR
# Sweep sits inside the validated linear regime: SelfControl calib (RUN -03) is monotonic
# on [-2.5,+2] and saturates/collapses beyond (k=-4→floor, k≥3→blowup). §14.1.
DEFAULT_SWEEP = [-2.0, -1.0, 0.0, 1.0, 2.0]
ROBUST = ["Adaptability", "Trust", "SelfControl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facets", nargs="*", default=ROBUST)
    ap.add_argument("--sweep", nargs="*", type=float, default=DEFAULT_SWEEP)
    ap.add_argument("--leak-facets", nargs="*",
                    default=["Sociability", "Anxiety"],  # cross-domain leakage probes
                    help="extra facets to administer for off-target leakage (M4)")
    args = ap.parse_args()

    spi, items = load_spi()
    subset = keyed_items_for(spi["keys"], list(args.facets) + list(args.leak_facets))
    print(f"focused administration: {len(subset)} items "
          f"({len(args.facets)} targets + {len(args.leak_facets)} leakage facets)")
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device)
    base.eval()
    cmodel = ControlModel(base, LAYER_IDS)

    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf"))
               for f in args.facets}

    # calibration must be measured UNSTEERED once, then reused across all coeffs so the
    # readout baseline is fixed and only steering moves the scores.
    calib = compute_calibration(cmodel, tok, device=device)

    results = {}   # facet -> {k -> {scale -> score}}
    for facet in args.facets:
        vec = vectors[facet]
        results[facet] = {}
        for k in args.sweep:
            cmodel.reset()
            if k != 0.0:
                cmodel.set_control(vec, k)
            t0 = time.time()
            ev = administer_spi(cmodel, tok, spi, items, device=device, log_every=0,
                                calib=calib, subset=subset)
            scales = score_scales(ev, spi["keys"])
            results[facet][k] = scales
            print(f"[{facet}] k={k:+6.1f}  {facet}={scales[facet]:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        cmodel.reset()

    # ---- metrics ----
    report = {"model": DEFAULT_MODEL, "layer_ids": LAYER_IDS, "sweep": args.sweep,
              "facets": {}}
    print("\n=== TIER-0 DOSE-RESPONSE ===")
    n_pass = 0
    for facet in args.facets:
        ks = args.sweep
        target = [results[facet][k][facet] for k in ks]
        rho, _ = spearmanr(ks, target)
        # max off-target leakage: largest |spearman| onto a non-target facet
        leak = {}
        for other in spi["keys"]:
            if other == facet or other in spi["factorNames"]:
                continue
            ovals = [results[facet][k].get(other) for k in ks]
            if all(v is not None for v in ovals):
                lr, _ = spearmanr(ks, ovals)
                leak[other] = lr
        worst = max(leak.items(), key=lambda x: abs(x[1])) if leak else (None, 0.0)
        passed = abs(rho) >= 0.80
        n_pass += passed
        report["facets"][facet] = {
            "target_scores": dict(zip(map(str, ks), target)),
            "spearman_rho": rho, "effect_range": max(target) - min(target),
            "pass": bool(passed), "worst_leak_facet": worst[0], "worst_leak_rho": worst[1]}
        flag = "✅" if passed else "❌"
        print(f"{flag} {facet:14} ρ={rho:+.3f}  Δ={max(target)-min(target):+.3f}  "
              f"curve={[round(t,2) for t in target]}  "
              f"worst-leak={worst[0]}({worst[1]:+.2f})")

    report["n_pass"] = n_pass
    report["gate_pass"] = n_pass >= 3
    verdict = ("PASS — ≥3 facets monotonic; proceed to persona battery + loop"
               if n_pass >= 3 else
               "FAIL — <3 facets clear ρ≥0.80; investigate layers/coeff/vectors before looping")
    print(f"\nGATE: {n_pass}/{len(args.facets)} facets pass (ρ≥0.80).  {verdict}")

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(report, open(os.path.join(OUT_DIR, "tier0-doseresponse.json"), "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else float(o))
    write_report_md(report, verdict)


def write_report_md(report, verdict):
    p = os.path.join(OUT_DIR, "tier0-report.md")
    lines = ["# Tier-0 Report — Steering Feasibility (dose-response)\n",
             f"Model: `{report['model']}`  ·  layers `{report['layer_ids']}`  ·  "
             f"sweep `{report['sweep']}`\n",
             f"**GATE: {report['n_pass']}/{len(report['facets'])} facets pass (Spearman ρ≥0.80).** {verdict}\n",
             "| Facet | ρ(k,score) | Δ effect | pass | worst off-target leak |",
             "|---|---|---|---|---|"]
    for f, d in report["facets"].items():
        lines.append(f"| {f} | {d['spearman_rho']:+.3f} | {d['effect_range']:+.3f} | "
                     f"{'✅' if d['pass'] else '❌'} | {d['worst_leak_facet']} "
                     f"({d['worst_leak_rho']:+.2f}) |")
    lines += ["", "## Dose-response curves (target facet SPI score vs k)", ""]
    for f, d in report["facets"].items():
        lines.append(f"- **{f}**: " + "  ".join(
            f"k={k}→{v:.2f}" for k, v in d["target_scores"].items()))
    open(p, "w").write("\n".join(lines) + "\n")
    print(f"saved → {p}")


if __name__ == "__main__":
    main()
