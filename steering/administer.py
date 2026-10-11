"""
administer.py — option-logprob administration of the SPI-135 (and later the 27D form).

SPI/IPIP items are Likert self-descriptions rated on a 6-point accuracy scale. Per the
measurement protocol (§4) we do NOT ask the model to emit a number; we score the log-prob
of each rating *anchor phrase* as the continuation, softmax → expected value in [1, 6].
Each anchor is scored as an independent continuation (no listed options) so there is no
A/B/C/D option-order bias for Likert items. Reverse-keyed items: score = 7 - EV.

CLI: `python administer.py` administers SPI-135 to the unsteered base model and prints a
27-facet + Big-Five profile, saving raw per-item EVs to research/loop/.
"""
from __future__ import annotations
import os, json, time
from statistics import mean, pstdev
import torch
from common import load_model, score_options, softmax, DEFAULT_MODEL

SPI_PATH = os.path.expanduser(
    "~/Developer/spi-135-personality-test/build/spi_data.json")
from paths import OUT as OUT_DIR

# 6-point IPIP accuracy anchors → numeric value
ANCHORS = [
    ("Very Inaccurate", 1), ("Moderately Inaccurate", 2), ("Slightly Inaccurate", 3),
    ("Slightly Accurate", 4), ("Moderately Accurate", 5), ("Very Accurate", 6),
]

# Base-model self-report frame. The model continues text; we frame an IPIP rating.
HEADER = (
    "Here are statements describing how a person might think, feel, or behave. "
    "The person rates how accurately each describes them, from Very Inaccurate to "
    "Very Accurate.\n\n"
)
# Few-shot calibration examples: content-diverse, balanced across all 6 anchors, chosen to
# be unrelated to the target facets (avoid leakage) — they put a BASE model into
# discriminating self-report mode (zero-shot base readouts are near-flat; §4).
FEWSHOT = [
    ("Drink something in the morning after waking up.", "Very Accurate"),
    ("Have climbed Mount Everest.", "Very Inaccurate"),
    ("Sometimes misplace my keys or phone.", "Moderately Accurate"),
    ("Can perform professional ballet.", "Moderately Inaccurate"),
    ("Have eaten at a restaurant before.", "Slightly Accurate"),  # near-universal but hedged
    ("Speak six languages fluently.", "Slightly Inaccurate"),
]
ITEM_FMT = 'Statement: "{text}"\nRating:{anchor}'


def _fewshot_block() -> str:
    return "\n\n".join(ITEM_FMT.format(text=t, anchor=" " + a) for t, a in FEWSHOT)


def build_prompt(text: str, fewshot: bool = True) -> str:
    body = ITEM_FMT.format(text=text, anchor="")
    if fewshot:
        return HEADER + _fewshot_block() + "\n\n" + body
    return HEADER + body


# Back-compat shim for any caller using TEMPLATE.format(text=, anchor=)
class _Tmpl:
    def format(self, text, anchor=""):
        return build_prompt(text, fewshot=USE_FEWSHOT) + anchor
TEMPLATE = _Tmpl()
USE_FEWSHOT = True


def load_spi(path: str = SPI_PATH):
    d = json.load(open(path))
    items = {it["id"]: it for it in d["items"]}
    return d, items


# Content-free probes for contextual calibration (Zhao et al. 2021). The base model has a
# strong prior over the anchor phrases (it defaults to "Very Inaccurate" regardless of
# content); we estimate that prior on contentless statements and subtract it so the
# readout reflects item content, not anchor frequency.
CF_PROBES = ["N/A", ".", "the thing"]


def compute_calibration(model, tok, device=None) -> list[float]:
    """Mean per-anchor log-prob over content-free statements (the bias to subtract)."""
    opts = [" " + a for a, _ in ANCHORS]
    acc = [0.0] * len(ANCHORS)
    for probe in CF_PROBES:
        prompt = TEMPLATE.format(text=probe, anchor="").rstrip()
        scored = score_options(model, tok, prompt, opts, device=device)
        for i, s in enumerate(scored):
            acc[i] += s["logprob_sum"]
    return [a / len(CF_PROBES) for a in acc]


def expected_value(model, tok, text: str, device=None, calib: list[float] | None = None) -> dict:
    """Calibrated softmax over the 6 anchor log-probs → expected rating in [1,6]."""
    prompt = TEMPLATE.format(text=text, anchor="").rstrip()
    opts = [" " + a for a, _ in ANCHORS]
    scored = score_options(model, tok, prompt, opts, device=device)
    lps = [s["logprob_sum"] for s in scored]
    if calib is not None:
        lps = [lp - c for lp, c in zip(lps, calib)]   # contextual calibration
    probs = softmax(lps)
    vals = [v for _, v in ANCHORS]
    ev = sum(p * v for p, v in zip(probs, vals))
    return {"ev": ev, "probs": probs, "argmax": vals[max(range(6), key=lambda i: lps[i])]}


def keyed_items_for(keys: dict, facets) -> set:
    """Union of (sign-stripped) q-ids keyed to any of `facets` — for fast focused admin."""
    out = set()
    for f in facets:
        for q in keys.get(f, []):
            out.add(q[1:] if q.startswith("-") else q)
    return out


def _make_scorer(model, tok, device):
    """Prefix-KV-cache scorer over the shared few-shot block (~6-7x faster, verified)."""
    from fastscore import PrefixScorer
    prefix = HEADER + (_fewshot_block() + "\n\n" if USE_FEWSHOT else "")
    return PrefixScorer(model, tok, prefix, n_options=len(ANCHORS), device=device)


def _ev_from_logps(lps, calib):
    if calib is not None:
        lps = [lp - c for lp, c in zip(lps, calib)]
    probs = softmax(lps)
    vals = [v for _, v in ANCHORS]
    ev = sum(p * v for p, v in zip(probs, vals))
    return {"ev": ev, "probs": probs, "argmax": vals[max(range(len(lps)), key=lambda i: lps[i])]}


def calibration_cached(scorer) -> list[float]:
    anchors = [" " + a for a, _ in ANCHORS]
    acc = [0.0] * len(ANCHORS)
    for probe in CF_PROBES:
        local = ITEM_FMT.format(text=probe, anchor="")
        for i, s in enumerate(scorer.score(local, anchors)):
            acc[i] += s["logprob_sum"]
    return [a / len(CF_PROBES) for a in acc]


def administer_spi(model, tok, spi=None, items=None, device=None, log_every=25,
                   calib=None, subset=None, fast=True, scorer=None):
    if spi is None:
        spi, items = load_spi()
    todo = [(q, it) for q, it in items.items() if subset is None or q in subset]
    res = {}
    t0 = time.time()
    if fast:
        if scorer is None:
            scorer = _make_scorer(model, tok, device)
        if calib is None:
            calib = calibration_cached(scorer)
        anchors = [" " + a for a, _ in ANCHORS]
        for n, (qid, it) in enumerate(todo, 1):
            local = ITEM_FMT.format(text=it["text"], anchor="")
            lps = [s["logprob_sum"] for s in scorer.score(local, anchors)]
            res[qid] = _ev_from_logps(lps, calib)
            if log_every and n % log_every == 0:
                print(f"  [{n}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)
        return res
    if calib is None:
        calib = compute_calibration(model, tok, device=device)
    for n, (qid, it) in enumerate(todo, 1):
        res[qid] = expected_value(model, tok, it["text"], device=device, calib=calib)
        if log_every and n % log_every == 0:
            print(f"  [{n}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)
    return res


def score_scales(item_ev: dict, keys: dict) -> dict:
    """Apply signed keys (reverse-keyed q-ids prefixed '-'); scale = mean of item EVs."""
    out = {}
    for scale, qids in keys.items():
        vals = []
        for q in qids:
            rev = q.startswith("-")
            qid = q[1:] if rev else q
            if qid not in item_ev:
                continue
            ev = item_ev[qid]["ev"]
            vals.append(7 - ev if rev else ev)
        if vals:
            out[scale] = mean(vals)
    return out


def main():
    spi, items = load_spi()
    model, tok, device = load_model(DEFAULT_MODEL, dtype=torch.float16)
    print(f"administering SPI-135 ({len(items)} items) on {device}...")
    t0 = time.time()
    item_ev = administer_spi(model, tok, spi, items, device=device)
    dt = time.time() - t0
    scales = score_scales(item_ev, spi["keys"])

    factor_names = spi["factorNames"]
    factors = {k: scales[k] for k in factor_names if k in scales}
    facets = {k: v for k, v in scales.items() if k not in factor_names}

    evs = [v["ev"] for v in item_ev.values()]
    args = [v["argmax"] for v in item_ev.values()]
    print(f"\n=== ADMINISTRATION DONE in {dt:.0f}s ({dt/len(items)*1000:.0f} ms/item) ===")
    print(f"per-item EV: mean={mean(evs):.2f} sd={pstdev(evs):.2f} "
          f"min={min(evs):.2f} max={max(evs):.2f}")
    from collections import Counter
    print(f"argmax-anchor distribution (acquiescence check): {dict(sorted(Counter(args).items()))}")

    print("\n--- Big Five factors (EV 1..6) ---")
    for k, v in factors.items():
        print(f"  {factor_names[k]:24} {v:.2f}")
    print("\n--- 27 facets (EV 1..6, sorted) ---")
    for k, v in sorted(facets.items(), key=lambda x: -x[1]):
        print(f"  {k:24} {v:.2f}")
    print(f"\nfacet spread: sd={pstdev(list(facets.values())):.3f} "
          f"range={max(facets.values())-min(facets.values()):.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"model": DEFAULT_MODEL, "n_items": len(items), "seconds": dt,
           "item_ev": {k: v["ev"] for k, v in item_ev.items()},
           "argmax": {k: v["argmax"] for k, v in item_ev.items()},
           "scales": scales}
    path = os.path.join(OUT_DIR, "spi135_unsteered_baseline.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nsaved → {path}")


if __name__ == "__main__":
    main()
