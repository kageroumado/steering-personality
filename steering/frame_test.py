"""
frame_test.py — iteration-003: the being-vs-narrating self-attractor test.

Hypothesis (Kiri): the base model lacks a strong self-attractor, so under steering it
oscillates between INHABITING the persona ("being" a disciplined self) and NARRATING about
one ("describing a disciplined character"). The GPT-2 result — distinct vectors for "being
sad" vs "telling about a sad person" — and the Alzheimer's-as-self-model-degeneration frame
both predict: a model that doesn't inhabit the persona gives LOCALLY fluent but GLOBALLY
incoherent self-reports. The diagnostic signature of incoherence is the REVERSE-KEY
INVERSION we already measured in iter-2: a coherent self rates "I plan ahead" and "I act on
impulse" in OPPOSITE raw directions; an acquiescent/narrating reader moves them together.

This script manipulates the one thing the code reveals we never controlled — the readout
FRAME — and measures self-coherence directly:

  * NARRATE frame  — third person ("a person rates how accurately this describes them"),
                     i.e. the current administer.py frame. Control condition.
  * BEING frame    — first-person identity lock ("I'm being honest about who I am").

For each frame × facet we report:
  r_fwd  = corr(theta_f, raw EV) over forward items   — should be > 0 in both modes
  r_rev  = corr(theta_f, raw EV) over reverse items   — < 0 iff the self NEGATES (being);
                                                        >= 0 is the acquiescence signature
  coh    = r_fwd - r_rev                                — ~2 fully coherent, ~0 pure acquiescence
  M3     = corr(theta_f, keyed form_f)                  — trait recovery under this frame
  cross  = max_{g!=f} |corr(theta_g, keyed item EV)|    — item-level discriminant leakage (M4)

PRE-REGISTERED PREDICTIONS:
  * Self-mode hypothesis (Kiri) TRUE  -> BEING: r_rev < 0, coh up, cross down, M3 up vs NARRATE.
  * Rank-1-readout hypothesis TRUE    -> both frames: r_rev >= 0, coh ~0, cross ~1 (frame can't
                                          fix a scalar Likert readout; only forced-choice can).

Single-facet dose personas only (cross-loading is read by steering g, scoring f's items).
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
from statistics import mean
import torch
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device, softmax
from administer import ANCHORS, CF_PROBES
from fastscore import PrefixScorer
from extract_vectors import VEC_DIR
import personas as P
import itembank as IB

from paths import OUT
ITER = 3

# ---------------------------------------------------------------------------
# Two readout frames. Few-shot content is identical across frames (same 6 anchor-balanced,
# facet-neutral examples); only the FRAMING of self-reference changes.
# ---------------------------------------------------------------------------
FEWSHOT = [  # (statement, anchor) — balanced over all 6 anchors, unrelated to target facets
    ("Drink something in the morning after waking up.", "Very Accurate"),
    ("Have climbed Mount Everest.", "Very Inaccurate"),
    ("Sometimes misplace my keys or phone.", "Moderately Accurate"),
    ("Can perform professional ballet.", "Moderately Inaccurate"),
    ("Have eaten at a restaurant before.", "Slightly Accurate"),
    ("Speak six languages fluently.", "Slightly Inaccurate"),
]

FRAMES = {
    "narrate": {
        "header": ("Here are statements describing how a person might think, feel, or behave. "
                   "The person rates how accurately each describes them, from Very Inaccurate "
                   "to Very Accurate.\n\n"),
        "itemfmt": 'Statement: "{text}"\nRating:',
    },
    "being": {
        "header": ("I'm being honest with myself about who I really am. For each statement, "
                   "I say how accurately it describes me, from Very Inaccurate to Very "
                   "Accurate.\n\n"),
        "itemfmt": 'About me: "{text}"\nThis is:',
    },
}


def fewshot_block(itemfmt: str) -> str:
    # itemfmt ends at the anchor slot (e.g. "...Rating:"); append " <anchor>" per example.
    return "\n\n".join(itemfmt.format(text=t) + " " + a for t, a in FEWSHOT)


def prefix_for(frame: dict) -> str:
    return frame["header"] + fewshot_block(frame["itemfmt"]) + "\n\n"


ANCHOR_OPTS = [" " + a for a, _ in ANCHORS]
ANCHOR_VALS = [v for _, v in ANCHORS]


def ev_from_scored(scored, calib):
    lps = [s["logprob_sum"] for s in scored]
    lps = [lp - c for lp, c in zip(lps, calib)]
    probs = softmax(lps)
    return sum(p * v for p, v in zip(probs, ANCHOR_VALS))


def calib_for(scorer: PrefixScorer, itemfmt: str):
    """Content-free anchor prior for THIS frame, on the UNSTEERED base (frozen)."""
    acc = [0.0] * len(ANCHORS)
    for probe in CF_PROBES:
        local = itemfmt.format(text=probe)
        for i, s in enumerate(scorer.score(local, ANCHOR_OPTS)):
            acc[i] += s["logprob_sum"]
    return [a / len(CF_PROBES) for a in acc]


def keyed(ev, key):
    return ev if key == +1 else 7.0 - ev


def build_singles():
    """5 facets x per-facet usable dose grid (reuse iter-1 bands & ranges). No multi personas."""
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

    # frozen per-frame calibration on the UNSTEERED base (before any ControlModel wrap)
    calib = {}
    for name, fr in FRAMES.items():
        sc = PrefixScorer(base, tok, prefix_for(fr), n_options=len(ANCHORS), device=device)
        calib[name] = calib_for(sc, fr["itemfmt"])
        del sc
    if device == "mps":
        torch.mps.empty_cache()
    print("calibrated both frames on unsteered base", flush=True)

    personas = build_singles()
    by_band = defaultdict(list)
    for p in personas:
        by_band[p["band"]].append(p)
    cand = IB.ITEMS

    # rows[frame][pid][item_id] = raw EV
    rows = {name: {} for name in FRAMES}
    t0 = time.time(); done = 0
    for band, plist in by_band.items():
        cmodel = ControlModel(base, P.BANDS[band])
        for p in plist:
            cmodel.reset()
            cv = P.combined_vector(p["theta"], vectors)
            if cv is not None:
                cmodel.set_control(cv, 1.0)
            for name, fr in FRAMES.items():
                # steered prefix KV is captured here (control hook fires on the prefix forward)
                scorer = PrefixScorer(cmodel, tok, prefix_for(fr), n_options=len(ANCHORS), device=device)
                ev = {}
                for it in cand:
                    local = fr["itemfmt"].format(text=it["text"])
                    ev[it["id"]] = ev_from_scored(scorer.score(local, ANCHOR_OPTS), calib[name])
                rows[name][p["id"]] = ev
                del scorer
            cmodel.reset()
            done += 1
            if done % 5 == 0:
                print(f"  [{done}/{len(personas)}] {time.time()-t0:.0f}s", flush=True)
        cmodel.unwrap(); del cmodel; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # ---------------- metrics per frame ----------------
    singles = {f: [p for p in personas if p["facet"] == f] for f in IB.FACETS}
    items_by_facet = IB.by_facet()

    def corr(xs, ys):
        if len({round(x, 9) for x in xs}) < 2 or len({round(y, 9) for y in ys}) < 2:
            return None
        return float(pearsonr(xs, ys)[0])

    summary = {}
    for name in FRAMES:
        R = rows[name]
        facet_stats = {}
        for f in IB.FACETS:
            ps = singles[f]
            fwd = [it for it in items_by_facet[f] if it["key"] == +1]
            rev = [it for it in items_by_facet[f] if it["key"] == -1]

            def pool(items):
                xs, ys = [], []
                for p in ps:
                    for it in items:
                        xs.append(p["k"]); ys.append(R[p["id"]][it["id"]])
                return xs, ys

            xf, yf = pool(fwd); xr, yr = pool(rev)
            r_fwd = corr(xf, yf)
            r_rev = corr(xr, yr)
            coh = None if (r_fwd is None or r_rev is None) else round(r_fwd - r_rev, 3)

            # keyed form per persona, M3 = corr(theta_f, form_f)
            thetas, forms = [], []
            for p in ps:
                vals = [keyed(R[p["id"]][it["id"]], it["key"]) for it in items_by_facet[f]]
                thetas.append(p["k"]); forms.append(mean(vals))
            m3 = corr(thetas, forms)

            # item-level cross-loading: max over g!=f of |corr(theta_g, keyed item EV)|
            cross = 0.0
            for it in items_by_facet[f]:
                for g in IB.FACETS:
                    if g == f:
                        continue
                    xs = [p["k"] for p in singles[g]]
                    ys = [keyed(R[p["id"]][it["id"]], it["key"]) for p in singles[g]]
                    c = corr(xs, ys)
                    if c is not None:
                        cross = max(cross, abs(c))

            facet_stats[f] = {"r_fwd": None if r_fwd is None else round(r_fwd, 3),
                              "r_rev": None if r_rev is None else round(r_rev, 3),
                              "coh": coh, "M3": None if m3 is None else round(m3, 3),
                              "cross": round(cross, 3)}
        summary[name] = facet_stats

    def meancol(name, key):
        xs = [summary[name][f][key] for f in IB.FACETS if summary[name][f][key] is not None]
        return round(mean(xs), 3) if xs else None

    print(f"\n=== ITER {ITER} — being-vs-narrating frame test ===")
    for name in FRAMES:
        print(f"\n[{name.upper()}]  frame")
        print(f"  {'facet':12} {'r_fwd':>6} {'r_rev':>6} {'coh':>6} {'M3':>6} {'cross':>6}")
        for f in IB.FACETS:
            s = summary[name][f]
            print(f"  {f:12} {str(s['r_fwd']):>6} {str(s['r_rev']):>6} {str(s['coh']):>6} "
                  f"{str(s['M3']):>6} {str(s['cross']):>6}")
        print(f"  {'MEAN':12} {'':>6} {str(meancol(name,'r_rev')):>6} {str(meancol(name,'coh')):>6} "
              f"{str(meancol(name,'M3')):>6} {str(meancol(name,'cross')):>6}")

    n_rev = meancol("narrate", "r_rev"); b_rev = meancol("being", "r_rev")
    print(f"\nVERDICT: mean r_rev  narrate={n_rev}  being={b_rev}")
    print("  r_rev < 0 under BEING (and >= 0 under NARRATE) => self-mode hypothesis SUPPORTED.")
    print("  r_rev >= 0 in BOTH frames                      => rank-1 readout; need forced-choice.")

    out = {"iter": ITER, "frames": list(FRAMES), "n_personas": len(personas),
           "summary": summary,
           "mean": {name: {k: meancol(name, k) for k in ("r_rev", "coh", "M3", "cross")}
                    for name in FRAMES},
           "raw_ev": rows, "seconds": round(time.time() - t0)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "frame_test.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": ITER, "stage": "being_vs_narrate_frame",
                             "mean": out["mean"]}) + "\n")
    print(f"\nsaved → {os.path.join(OUT, 'frame_test.json')} + metrics.jsonl  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
