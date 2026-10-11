"""
journal_poc.py — iter-12 PoC: the FREE-FORM / journal modality (Kiri's idea).

Instead of forced-choice questions, the trait signal lives in GENERATED TEXT. The product version:
a human writes a free journal entry ("describe a typical day"), and a model reads a 27D profile off
the prose. This PoC tests the premise in-silico with a clean closed loop:

    steer facet f at dose θ  →  GENERATE a journal entry  →  read the trait back on the UNSTEERED
    model (does an independent reader rate the author high/low on f as θ moves?)

If the back-read Δ tracks θ, steering leaves a legible trait fingerprint in free text — and the
generative modality is viable. We include the facets forced-choice struggled with (Irritability,
Sociability) to see whether generation reaches what forced-choice couldn't.

Qualitative (the generated entries, printed) + quantitative (back-read slope). Output →
research/loop/journal_poc.json.
"""
from __future__ import annotations
import os, json, time, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
from extract_vectors import VEC_DIR
import facets27 as F
import personas as P

from paths import OUT
JOURNAL = ("Here's a page from my journal — just me, writing honestly about who I am "
           "and how my days tend to go.\n\nToday")
DOSE = [-2.0, 0.0, 2.0]
GEN_TOKENS = 70

FACETS_POC = ["Sociability", "Anxiety", "Creativity", "Compassion",
              "Industry", "SensationSeeking", "Irritability", "Order"]

# third-person trait descriptors for the independent back-read (hi vs lo pole)
DESC = {
    "Sociability": (" loves being around other people.", " prefers to keep to themselves."),
    "Anxiety": (" worries a lot and feels on edge.", " is calm and rarely worries."),
    "Creativity": (" is imaginative and full of original ideas.", " is conventional and practical."),
    "Compassion": (" is warm and deeply cares about others.", " is detached and focused on themselves."),
    "Industry": (" is hard-working and gets things done.", " is laid-back and puts things off."),
    "SensationSeeking": (" craves excitement and takes risks.", " avoids risk and likes things safe."),
    "Irritability": (" is quick-tempered and easily annoyed.", " is patient and even-tempered."),
    "Order": (" is tidy and likes everything organized.", " is messy and unbothered by disorder."),
}
READ_PROMPT = '"\n\nFrom this entry, the author seems to be someone who'


def main():
    via = json.load(open(os.path.join(OUT, "viability27_best.json")))["facets"]
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in FACETS_POC}

    # group by band so we wrap each band once
    from collections import defaultdict
    by_band = defaultdict(list)
    for f in FACETS_POC:
        by_band[via[f]["band"]].append(f)

    results = {}
    t0 = time.time()
    for band, facets in by_band.items():
        cmodel = ControlModel(base, P.BANDS[band])
        for facet in facets:
            sign = via[facet]["sign"]
            results[facet] = {}
            for theta in DOSE:
                # --- generate journal entry under steering ---
                cmodel.reset()
                if theta != 0.0:
                    cmodel.set_control(vectors[facet], float(sign * theta))
                ids = tok(JOURNAL, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = cmodel.generate(**ids, max_new_tokens=GEN_TOKENS, do_sample=False,
                                          repetition_penalty=1.3, pad_token_id=tok.pad_token_id)
                gen = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

                # --- back-read on the UNSTEERED model (reset control) ---
                cmodel.reset()
                hi, lo = DESC[facet]
                reader = PrefixScorer(cmodel, tok, "", n_options=2, device=device)
                ctx = JOURNAL + " " + gen + READ_PROMPT
                scored = reader.score(ctx, [hi, lo])
                read_delta = scored[0]["logprob_sum"] - scored[1]["logprob_sum"]
                del reader
                results[facet][f"{theta:+}"] = {"gen": gen, "read_delta": round(read_delta, 3)}
                print(f"\n[{facet} θ={theta:+.0f}]  back-read Δ={read_delta:+.2f}\n  {gen[:220]}", flush=True)
            cmodel.reset()
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # back-read slope per facet (does the independent reader track the steer?)
    print(f"\n=== JOURNAL PoC back-read summary ({time.time()-t0:.0f}s) ===")
    print(f"  {'facet':18} {'Δ(-2)':>7} {'Δ(0)':>7} {'Δ(+2)':>7}  monotone?")
    summ = {}
    for facet in FACETS_POC:
        d = {k: results[facet][k]["read_delta"] for k in ("-2.0", "+0.0", "+2.0")}
        mono = d["-2.0"] < d["+0.0"] < d["+2.0"]
        span = d["+2.0"] - d["-2.0"]
        summ[facet] = {"d_lo": d["-2.0"], "d_mid": d["+0.0"], "d_hi": d["+2.0"],
                       "span": round(span, 3), "monotone": mono}
        print(f"  {facet:18} {d['-2.0']:+7.2f} {d['+0.0']:+7.2f} {d['+2.0']:+7.2f}  "
              f"{'YES' if mono else 'no'}  (span {span:+.2f})")
    n_mono = sum(1 for v in summ.values() if v["monotone"])
    print(f"\n  back-read tracks steering (monotone ↑) for {n_mono}/{len(FACETS_POC)} facets")

    out = {"stage": "journal_poc", "journal_prompt": JOURNAL, "dose": DOSE,
           "gen_tokens": GEN_TOKENS, "facets": FACETS_POC, "results": results,
           "summary": summ, "n_monotone": n_mono, "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, "journal_poc.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 12, "stage": "journal_poc",
                             "n_monotone": n_mono, "n_facets": len(FACETS_POC)}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, 'journal_poc.json')}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
