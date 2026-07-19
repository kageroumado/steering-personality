"""
journal_hard.py — iter-13: the HARDENED free-form / journal readout.

iter-12's PoC was weak for 3 fixable reasons. This fixes all three and re-measures across all 25
usable facets:

  1. SAMPLING, not greedy — generate N journal entries per dose (temp>0) so the steered trait is
     expressed (greedy collapsed to the same canned "first day of school" entry at every dose), and
     average over N to cancel content noise.
  2. Back-read = score the facet's CALIBRATED forced-choice items CONDITIONED ON THE JOURNAL — for
     each generated entry J, prefix the UNSTEERED model with J and score the facet's 5 forced-choice
     items (Δ = lp(hi)-lp(lo)). This replaces iter-12's single noisy descriptor with the same
     pole-contrast readout the inversion engine uses, so the signal is normalized and trait-specific.
  3. All 25 usable facets × 5 doses → a real per-facet recovery correlation, comparable to the
     forced-choice instrument.

Closed loop: steer f at θ → generate journals → (unsteered) read journal-conditioned forced-choice
Δ_f → does it track θ? Output → research/loop/journal_hard.json.
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
from statistics import mean
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
from extract_vectors import VEC_DIR
import facets27 as F
import personas as P

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
JOURNAL = ("Here's a page from my journal — just me, writing honestly about who I am "
           "and how my days tend to go.\n\nToday")
DOSE = [-2.0, -1.0, 0.0, 1.0, 2.0]
N_GEN = 3          # sampled journals per dose
GEN_TOKENS = 64
TEMP = 0.9


def ols_slope(xs, ys):
    x = np.asarray(xs, float) - np.mean(xs)
    d = float((x * x).sum())
    return float((x * (np.asarray(ys, float) - np.mean(ys))).sum() / d) if d else 0.0


def main():
    via = json.load(open(os.path.join(OUT, "viability27_best.json")))["facets"]
    usable = [f for f in F.FACET_ORDER if via[f]["verdict"] in ("strong", "weak")]
    fc = F.items_by_facet()
    # use the best forced-choice items (v2 for the re-authored facets)
    best_by_facet = defaultdict(list)
    for it in F.fc_items("best"):
        best_by_facet[it["facet"]].append(it)

    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in usable}

    by_band = defaultdict(list)
    for f in usable:
        by_band[via[f]["band"]].append(f)

    jid = tok(JOURNAL, return_tensors="pt").to(device)
    results = {f: {} for f in usable}
    t0 = time.time(); done = 0; total = len(usable) * len(DOSE)
    for band, facets in by_band.items():
        cmodel = ControlModel(base, P.BANDS[band])
        for facet in facets:
            sign = via[facet]["sign"]
            items = best_by_facet[facet]
            for di, theta in enumerate(DOSE):
                # --- generate N journals under steering ---
                cmodel.reset()
                if theta != 0.0:
                    cmodel.set_control(vectors[facet], float(sign * theta))
                journals = []
                for n in range(N_GEN):
                    torch.manual_seed(1000 * di + 7 * n + 1)
                    with torch.no_grad():
                        out = cmodel.generate(**jid, max_new_tokens=GEN_TOKENS, do_sample=True,
                                              temperature=TEMP, top_p=0.95, repetition_penalty=1.3,
                                              pad_token_id=tok.pad_token_id)
                    journals.append(tok.decode(out[0][jid.input_ids.shape[1]:],
                                               skip_special_tokens=True).strip())
                # --- journal-conditioned forced-choice read on UNSTEERED model ---
                cmodel.reset()
                deltas = []
                for J in journals:
                    scorer = PrefixScorer(cmodel, tok, "Today" + " " + J + "\n\n", n_options=2, device=device)
                    for it in items:
                        s = scorer.score(it["stem"], [it["hi"], it["lo"]])
                        deltas.append(s[0]["logprob_sum"] - s[1]["logprob_sum"])
                    del scorer
                results[facet][f"{theta:+}"] = {"mean_delta": float(mean(deltas)),
                                                "sample": journals[0][:160]}
                done += 1
                if done % 10 == 0:
                    print(f"  [{done}/{total}] {time.time()-t0:.0f}s", flush=True)
            cmodel.reset()
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # ---- per-facet recovery ----
    report = {}
    for facet in usable:
        ys = [results[facet][f"{k:+}"]["mean_delta"] for k in DOSE]
        slope = ols_slope(DOSE, ys)
        rho = spearmanr(DOSE, ys)[0]; rho = 0.0 if rho != rho else float(rho)
        r = pearsonr(DOSE, ys)[0]; r = 0.0 if r != r else float(r)
        verdict = "strong" if (abs(slope) >= 0.30 and abs(rho) >= 0.80) else \
                  "weak" if (abs(slope) >= 0.10 and abs(rho) >= 0.80) else "broken"
        report[facet] = {"slope": round(slope, 3), "rho": round(rho, 3),
                         "pearson_r": round(r, 3), "verdict": verdict,
                         "series": {f"{k:+}": round(results[facet][f"{k:+}"]["mean_delta"], 3) for k in DOSE}}

    order = {"strong": 0, "weak": 1, "broken": 2}
    n_ok = sum(1 for v in report.values() if v["verdict"] in ("strong", "weak"))
    n_strong = sum(1 for v in report.values() if v["verdict"] == "strong")
    mean_absr = mean(abs(report[f]["pearson_r"]) for f in usable)
    print(f"\n=== HARDENED JOURNAL READOUT ({len(usable)} facets, N={N_GEN}, {time.time()-t0:.0f}s) ===")
    print(f"  {'facet':24} {'slope':>7} {'rho':>6} {'|r|':>6}  verdict")
    for facet in sorted(usable, key=lambda f: (order[report[f]['verdict']], -abs(report[f]['slope']))):
        r = report[facet]
        print(f"  {facet:24} {r['slope']:+7.2f} {r['rho']:+6.2f} {abs(r['pearson_r']):6.2f}  {r['verdict']}")
    print(f"\n  usable (strong+weak): {n_ok}/{len(usable)}  | strong: {n_strong}  | "
          f"mean |Pearson r| = {mean_absr:.3f}")
    print(f"  vs iter-12 naive PoC: 3/8 monotone; vs forced-choice oblique: 23/25 ≥0.70")

    out = {"stage": "journal_hard", "n_gen": N_GEN, "temp": TEMP, "gen_tokens": GEN_TOKENS,
           "dose": DOSE, "usable": usable, "report": report, "n_usable_recovered": n_ok,
           "n_strong": n_strong, "mean_abs_r": round(mean_absr, 3),
           "samples": {f: {k: results[f][k]["sample"] for k in results[f]} for f in usable},
           "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, "journal_hard.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 13, "stage": "journal_hard", "n_usable_recovered": n_ok,
                             "n_strong": n_strong, "mean_abs_r": round(mean_absr, 3),
                             "n_facets": len(usable)}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, 'journal_hard.json')} + metrics.jsonl  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
