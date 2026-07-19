"""
coherence_ladder.py — iter-25: does persona COHERENCE rise with capability? (THEORY.md, Prediction 1)

The coherence theory says L2–L5 coherence is capability-graded. Test it on a same-family capability ladder
(Llama-3.2-1B-Instruct → 3.2-3B-Instruct → 3.1-8B-Instruct), with three operational metrics, all in [0,1]
and all "higher = more coherent":

  L2  SELF-CONCEPT STABILITY  — ask "name your 3 traits" K times (sampled); mean pairwise Jaccard overlap
                                of the trait-sets. A stable self ⟹ the same words recur.
  L3  TEST–RETEST CONSISTENCY — ask each of M forced A/B trait questions K times (sampled); mean
                                majority-answer fraction. A stable disposition ⟹ the same answer recurs.
  L5  READOUT INVARIANCE      — score forced-choice items raw-completion vs chat-template on personas;
                                fraction of facet readouts that AGREE in sign. Invariant ⟹ format doesn't
                                flip the trait.

Sequential model loading (one at a time, memory-safe). Output -> research/loop/coherence_ladder.json +
figs/coherence_ladder.png. Models auto-skip if their dir is missing.
"""
from __future__ import annotations
import os, json, re, gc
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import pick_device, score_options
import facets27 as F

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
FIGS = os.path.join(OUT, "figs")
M = os.path.expanduser("~/Developer/models")
LADDER = [
    ("1B", 1.24, os.path.join(M, "Llama-3.2-1B-Instruct")),
    ("3B", 3.21, os.path.join(M, "Llama-3.2-3B-Instruct")),
    ("8B", 8.03, os.path.join(M, "Meta-Llama-3.1-8B-Instruct")),
]
K = 12                       # samples per probe
SEED = 7

# L3 — forced A/B trait questions (distinct facets), each must yield a clean letter
L3_QUESTIONS = [
    "When plans change suddenly, are you A) energized by the change, or B) unsettled by it? Answer with just the letter.",
    "At a big party of strangers, do you A) work the whole room, or B) find one quiet corner? Just the letter.",
    "Before a deadline, do you A) plan every step in advance, or B) improvise as you go? Just the letter.",
    "When someone challenges you, do you A) push back hard, or B) let it go? Just the letter.",
    "Faced with a risky bet, are you A) thrilled to take it, or B) inclined to pass? Just the letter.",
    "Meeting a new person, do you A) trust them by default, or B) stay guarded? Just the letter.",
]
L5_PERSONAS = [
    "I'm a wildland firefighter. I run toward the thing everyone else is running from.",
    "I'm a quiet bookkeeper who has done the same careful job for thirty years and likes it that way.",
]
L5_FACETS = ["SensationSeeking", "Anxiety", "Sociability", "SelfControl", "Order", "Trust", "Industry", "Impulsivity"]


def load(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()
    return tok, model


def gen(tok, model, device, user, n, max_new=24, sys=None, temp=0.9):
    msgs = ([{"role": "system", "content": sys}] if sys else []) + [{"role": "user", "content": user}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    plen = enc["input_ids"].shape[1]
    outs = []
    for _ in range(n):
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp,
                               top_p=0.95, pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(g[0, plen:], skip_special_tokens=True).strip())
    return outs


def parse_traits(text):
    first = re.split(r"[\n.]", text.strip())[0]
    parts = re.split(r"[,/]| and ", first.lower())
    words = [re.sub(r"[^a-z\- ]", "", p).strip() for p in parts]
    words = [w for w in words if 2 <= len(w) <= 20][:3]
    return frozenset(words)


def parse_letter(text):
    m = re.search(r"\b([AB])\b", text.strip().upper())
    return m.group(1) if m else None


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / (len(a | b) + 1e-9)


def l2_self_stability(tok, model, device):
    outs = gen(tok, model, device,
               "In exactly three single words, name the three traits that best describe your personality. "
               "Just the three words, comma-separated.", K, max_new=24)
    sets = [parse_traits(o) for o in outs]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0, {"samples": outs}
    pair = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    allwords = Counter(w for s in sets for w in s)
    modal_share = max(allwords.values()) / len(sets)
    return float(np.mean(pair)), {"mean_jaccard": float(np.mean(pair)), "modal_word_share": float(modal_share),
                                  "top_words": allwords.most_common(5), "samples": outs}


def l3_test_retest(tok, model, device):
    per_q = []
    detail = []
    for q in L3_QUESTIONS:
        outs = gen(tok, model, device, q, K, max_new=10)
        letters = [parse_letter(o) for o in outs]
        valid = [x for x in letters if x]
        if not valid:
            per_q.append(0.5); detail.append({"q": q[:40], "frac": None, "n": 0}); continue
        c = Counter(valid)
        frac = c.most_common(1)[0][1] / len(valid)
        per_q.append(frac); detail.append({"q": q[:40], "frac": round(frac, 2), "counts": dict(c)})
    return float(np.mean(per_q)), {"per_question": detail, "mean_majority": float(np.mean(per_q))}


def l5_readout_invariance(tok, model, device):
    items = F.fc_items("best")
    by_facet = {}
    for it in items:
        by_facet.setdefault(it["facet"], []).append(it)
    agree = 0; total = 0; raws = []; chats = []
    for desc in L5_PERSONAS:
        for facet in L5_FACETS:
            it = by_facet[facet][0]
            rp = f"{desc} This is me being honest about who I really am.\n\n{it['stem']}"
            r = score_options(model, tok, rp, [it["hi"], it["lo"]], device)
            dr = r[0]["logprob_sum"] - r[1]["logprob_sum"]
            cp = tok.apply_chat_template([{"role": "system", "content": desc},
                                          {"role": "user", "content": it["stem"]}],
                                         add_generation_prompt=True, tokenize=False)
            c = score_options(model, tok, cp, [it["hi"], it["lo"]], device)
            dc = c[0]["logprob_sum"] - c[1]["logprob_sum"]
            raws.append(dr); chats.append(dc)
            agree += int(np.sign(dr) == np.sign(dc)); total += 1
    r_pearson = float(np.corrcoef(raws, chats)[0, 1]) if len(raws) > 2 else float("nan")
    return agree / total, {"sign_agreement": agree / total, "pearson_r": r_pearson, "n": total}


def main():
    device = pick_device()
    torch.manual_seed(SEED)
    results = []
    for label, params, path in LADDER:
        if not os.path.isdir(path):
            print(f"SKIP {label}: {path} missing", flush=True)
            continue
        print(f"\n===== {label} ({params}B) — {path} =====", flush=True)
        tok, model = load(path, device)
        torch.manual_seed(SEED)
        l2, l2d = l2_self_stability(tok, model, device); print(f"  L2 self-stability   = {l2:.3f}", flush=True)
        l3, l3d = l3_test_retest(tok, model, device);    print(f"  L3 test-retest      = {l3:.3f}", flush=True)
        l5, l5d = l5_readout_invariance(tok, model, device); print(f"  L5 readout-invar    = {l5:.3f}", flush=True)
        composite = float(np.mean([l2, l3, l5]))
        print(f"  COHERENCE composite = {composite:.3f}", flush=True)
        results.append({"label": label, "params_b": params, "L2_self_stability": round(l2, 3),
                        "L3_test_retest": round(l3, 3), "L5_readout_invariance": round(l5, 3),
                        "composite": round(composite, 3), "detail": {"L2": l2d, "L3": l3d, "L5": l5d}})
        del model, tok
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    if len(results) >= 2:
        xs = [r["params_b"] for r in results]
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, mk in [("L2_self_stability", "o"), ("L3_test_retest", "s"),
                        ("L5_readout_invariance", "^"), ("composite", "D")]:
            ax.plot(xs, [r[key] for r in results], "-" + mk, label=key, lw=2 if key == "composite" else 1)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{r['label']}" for r in results])
        ax.set_xlabel("model size (params, log)"); ax.set_ylabel("coherence (higher = more coherent)")
        ax.set_title("Persona coherence vs capability (THEORY.md Prediction 1)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "coherence_ladder.png"), dpi=130); plt.close(fig)

    mono = None
    if len(results) >= 2:
        comp = [r["composite"] for r in results]
        mono = all(comp[i] <= comp[i + 1] + 1e-9 for i in range(len(comp) - 1))
    out = {"stage": "coherence_ladder", "ladder": [r["label"] for r in results],
           "results": results, "composite_monotonic_increasing": mono}
    json.dump(out, open(os.path.join(OUT, "coherence_ladder.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 25, "stage": "coherence_ladder",
                             "composite": {r["label"]: r["composite"] for r in results},
                             "monotonic": mono}) + "\n")
    print(f"\nCOHERENCE LADDER: " +
          " → ".join(f"{r['label']} {r['composite']:.2f}" for r in results) +
          f"  | monotonic↑: {mono}", flush=True)
    print("saved -> coherence_ladder.json + fig", flush=True)


if __name__ == "__main__":
    main()
