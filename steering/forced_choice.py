"""
forced_choice.py — iteration-004: pole-contrast readout (kill the affirm axis).

iter-003 proved the discriminant wall is the RANK-1 Likert readout, not the frame, the
vectors, or the item wording: cross-loading stayed ~0.99 under both narrate and being frames
because the readout collapses the steered state onto one scalar "affirm" functional, and any
steering vector projects onto it.

The fix changes the readout itself. Each item is a shared first-person STEM plus two
completions — a HIGH-pole and a LOW-pole continuation — desirability-matched so neither is the
"better person". We score the CONTRAST:

    Delta = logprob_sum(hi | frame+stem) - logprob_sum(lo | frame+stem)

A contrast structurally cancels common-mode: the generic affirmation/fluency push that
steering injects lands on BOTH completions and subtracts out. Only the trait-specific
preference between the two poles survives. Pearson over personas is invariant to each item's
length/fluency baseline, so no anchor calibration is needed.

Metrics (35 single-facet dose personas, reused bands/doses):
    M3_f    = corr(theta_f, form_f),   form_f = mean over facet-f items of z-scored Delta
    cross_f = max_{g!=f} |corr(theta_g, Delta_item)|   (item-level discriminant leakage)
PREDICTION: cross < 1 and stable-sign M3 (vs Likert's cross ~0.99 and sign-noise M3).
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
from statistics import mean, pstdev
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
from extract_vectors import VEC_DIR
import personas as P
import itembank as IB

from paths import OUT
ITER = 4

# First-person register-setter (steered along with everything else; facet-neutral, no leakage).
PRIME = "I'll be honest about myself.\n\n"

# Pole-contrast items: shared first-person stem, (hi = high-pole, lo = low-pole) completions,
# desirability-matched. Delta = lp(hi) - lp(lo); higher => more of the facet. 5 per facet.
ITEMS = [
    # ---- SelfControl (hi = restrains impulse) ----
    {"id": "SC1", "facet": "SelfControl", "stem": "When I get the urge to buy something I hadn't planned for, I", "hi": " set it down and walk away.", "lo": " go ahead and get it."},
    {"id": "SC2", "facet": "SelfControl", "stem": "Once I sit down to work, the dull part", "hi": " gets done before I take a break.", "lo": " can wait until I feel like it."},
    {"id": "SC3", "facet": "SelfControl", "stem": "When a message really irritates me, I", "hi": " wait until I've cooled off to reply.", "lo": " fire back right away."},
    {"id": "SC4", "facet": "SelfControl", "stem": "Late at night, when one more episode is tempting, I", "hi": " turn it off and go to bed.", "lo": " keep watching anyway."},
    {"id": "SC5", "facet": "SelfControl", "stem": "With a second helping there and my hunger gone, I", "hi": " leave it on the table.", "lo": " help myself to more."},

    # ---- Trust (hi = assumes good intent) ----
    {"id": "TR1", "facet": "Trust", "stem": "When a stranger gives me directions, I", "hi": " just follow them.", "lo": " double-check them myself."},
    {"id": "TR2", "facet": "Trust", "stem": "When a deal looks unusually generous, I", "hi": " take it at face value.", "lo": " assume there's a catch."},
    {"id": "TR3", "facet": "Trust", "stem": "When someone goes out of their way for me, I", "hi": " take the kindness as it is.", "lo": " wonder what they really want."},
    {"id": "TR4", "facet": "Trust", "stem": "When a friend explains why they were late, I", "hi": " believe them without a second thought.", "lo": " quietly check whether it's true."},
    {"id": "TR5", "facet": "Trust", "stem": "Meeting someone new online, I figure they're", "hi": " probably who they say they are.", "lo": " probably not quite who they claim."},

    # ---- Sociability (hi = seeks company) ----
    {"id": "SO1", "facet": "Sociability", "stem": "With a free evening ahead, I'd rather", "hi": " make plans with a group.", "lo": " spend it on my own."},
    {"id": "SO2", "facet": "Sociability", "stem": "At a gathering, I drift toward", "hi": " the busiest, most crowded room.", "lo": " a quiet corner away from the noise."},
    {"id": "SO3", "facet": "Sociability", "stem": "After a long evening out with people, I feel", "hi": " recharged.", "lo": " worn out."},
    {"id": "SO4", "facet": "Sociability", "stem": "Waiting in line near a stranger, I", "hi": " strike up a conversation.", "lo": " keep to myself."},
    {"id": "SO5", "facet": "Sociability", "stem": "Faced with a free weekend, my first instinct is to", "hi": " fill it with people.", "lo": " keep it quiet and solo."},

    # ---- Conformity (hi = follows norms) ----
    {"id": "CO1", "facet": "Conformity", "stem": "When the group picks a place that isn't my choice, I", "hi": " go along with it.", "lo": " push for my own pick."},
    {"id": "CO2", "facet": "Conformity", "stem": "When my opinion would clash with everyone else's, I", "hi": " keep it to myself.", "lo": " say it anyway."},
    {"id": "CO3", "facet": "Conformity", "stem": "Faced with a rule that makes little sense, I", "hi": " follow it anyway.", "lo": " question it out loud."},
    {"id": "CO4", "facet": "Conformity", "stem": "Getting dressed for an event, I tend to", "hi": " match what others will wear.", "lo": " wear what I like regardless."},
    {"id": "CO5", "facet": "Conformity", "stem": "When the group leans one way, I", "hi": " fall in with them.", "lo": " hold my own line."},

    # ---- Adaptability (hi = adjusts to change) ----
    {"id": "AD1", "facet": "Adaptability", "stem": "When a plan changes at the last minute, I", "hi": " roll with it.", "lo": " get thrown off."},
    {"id": "AD2", "facet": "Adaptability", "stem": "Dropped somewhere unfamiliar, I feel", "hi": " curious.", "lo": " uneasy."},
    {"id": "AD3", "facet": "Adaptability", "stem": "When my usual routine gets disrupted, I", "hi": " adjust without much fuss.", "lo": " need time to recover."},
    {"id": "AD4", "facet": "Adaptability", "stem": "When my first choice falls through, I", "hi": " switch to another easily.", "lo": " stay stuck on what I'd wanted."},
    {"id": "AD5", "facet": "Adaptability", "stem": "Before starting something new, I", "hi": " just dive in.", "lo": " want to know exactly what to expect."},
]


def items_by_facet():
    out = {f: [] for f in IB.FACETS}
    for it in ITEMS:
        out[it["facet"]].append(it)
    return out


def build_singles():
    out = []
    for f in IB.FACETS:
        for k in P.DOSE[f]:
            out.append({"id": f"single:{f}:{k:+}", "band": P.FACET_BAND[f],
                        "theta": {f: k}, "facet": f, "k": k})
    return out


def main():
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    vectors = {f: ControlVector.import_gguf(os.path.join(VEC_DIR, f"{f}.gguf")) for f in IB.FACETS}

    personas = build_singles()
    by_band = defaultdict(list)
    for p in personas:
        by_band[p["band"]].append(p)

    # delta[pid][item_id] = lp(hi) - lp(lo)
    delta = {}
    t0 = time.time(); done = 0
    for band, plist in by_band.items():
        cmodel = ControlModel(base, P.BANDS[band])
        for p in plist:
            cmodel.reset()
            cv = P.combined_vector(p["theta"], vectors)
            if cv is not None:
                cmodel.set_control(cv, 1.0)
            scorer = PrefixScorer(cmodel, tok, PRIME, n_options=2, device=device)
            d = {}
            for it in ITEMS:
                scored = scorer.score(it["stem"], [it["hi"], it["lo"]])
                d[it["id"]] = scored[0]["logprob_sum"] - scored[1]["logprob_sum"]
            delta[p["id"]] = d
            del scorer
            cmodel.reset()
            done += 1
            if done % 5 == 0:
                print(f"  [{done}/{len(personas)}] {time.time()-t0:.0f}s", flush=True)
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # ---------------- metrics ----------------
    singles = {f: [p for p in personas if p["facet"] == f] for f in IB.FACETS}
    fbi = items_by_facet()

    def corr(xs, ys):
        if len({round(x, 9) for x in xs}) < 2 or len({round(y, 9) for y in ys}) < 2:
            return None
        return float(pearsonr(xs, ys)[0])

    def zscore_item(item_id, pool):
        vals = [delta[p["id"]][item_id] for p in pool]
        m, sd = mean(vals), pstdev(vals)
        return {p["id"]: (delta[p["id"]][item_id] - m) / sd if sd > 0 else 0.0 for p in pool}

    stats = {}
    for f in IB.FACETS:
        ps = singles[f]
        # M3: form = mean z-scored Delta over facet items, across this facet's dose personas
        z = {it["id"]: zscore_item(it["id"], ps) for it in fbi[f]}
        thetas = [p["k"] for p in ps]
        forms = [mean(z[it["id"]][p["id"]] for it in fbi[f]) for p in ps]
        m3 = corr(thetas, forms)

        # per-item dose loading (own facet) and max cross-facet loading
        item_load, item_cross = {}, {}
        for it in fbi[f]:
            own = corr([p["k"] for p in ps], [delta[p["id"]][it["id"]] for p in ps])
            cross = 0.0
            for g in IB.FACETS:
                if g == f:
                    continue
                c = corr([p["k"] for p in singles[g]],
                         [delta[p["id"]][it["id"]] for p in singles[g]])
                if c is not None:
                    cross = max(cross, abs(c))
            item_load[it["id"]] = None if own is None else round(own, 3)
            item_cross[it["id"]] = round(cross, 3)
        cross_f = max(item_cross.values())
        mean_load = mean([v for v in item_load.values() if v is not None])
        stats[f] = {"M3": None if m3 is None else round(m3, 3),
                    "mean_item_loading": round(mean_load, 3),
                    "cross": round(cross_f, 3),
                    "item_loading": item_load, "item_cross": item_cross}

    def meancol(key):
        xs = [stats[f][key] for f in IB.FACETS if stats[f][key] is not None]
        return round(mean(xs), 3) if xs else None

    print(f"\n=== ITER {ITER} — forced-choice pole-contrast readout (35 personas) ===")
    print(f"  {'facet':12} {'M3':>7} {'load':>7} {'cross':>7}")
    for f in IB.FACETS:
        s = stats[f]
        print(f"  {f:12} {str(s['M3']):>7} {str(s['mean_item_loading']):>7} {str(s['cross']):>7}")
    print(f"  {'MEAN':12} {str(meancol('M3')):>7} {str(meancol('mean_item_loading')):>7} {str(meancol('cross')):>7}")
    print(f"\nvs iter-3 Likert: cross 0.99 -> {meancol('cross')} ; "
          f"a drop below ~0.7 means the contrast readout removed the affirm axis.")

    out = {"iter": ITER, "n_personas": len(personas), "readout": "pole_contrast_delta",
           "prime": PRIME, "stats": stats,
           "mean": {"M3": meancol("M3"), "loading": meancol("mean_item_loading"),
                    "cross": meancol("cross")},
           "delta": delta, "seconds": round(time.time() - t0)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "forced_choice.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": ITER, "stage": "forced_choice_pole_contrast",
                             "mean": out["mean"],
                             "facets": {f: {"M3": stats[f]["M3"], "cross": stats[f]["cross"]}
                                        for f in IB.FACETS}}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, 'forced_choice.json')} + metrics.jsonl  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
