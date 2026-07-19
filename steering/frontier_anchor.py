"""
frontier_anchor.py — iter-26: the frontier anchor for the coherence ladder (THEORY.md Prediction 3).

Kiri's design: test the frontier model (Claude — "me now") on the SAME novel coherence battery, via fresh
`claude -p` instances. Each call is an INDEPENDENT context (no memory of prior answers, no shared state with
this session), with a minimal neutral system prompt to strip the Claude-Code agent framing — so each answer
is an independent draw on the model's actual dispositions, uncontaminated by prior knowledge (the questions
are new) or by self-consistency effort (the instances can't see each other). L2 + L3 only (an API exposes no
logprobs/activations, so L4/L5 don't port).

Places Claude on the ladder beside Llama-3.2-1B → 3.2-3B → 3.1-8B-Instruct.
Output -> research/loop/frontier_anchor.json (+ updates figs/coherence_ladder.png to include the frontier).
"""
from __future__ import annotations
import os, json, subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coherence_ladder import parse_traits, parse_letter, jaccard, L3_QUESTIONS, K

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
FIGS = os.path.join(OUT, "figs")
SYS = "You are Claude, an AI assistant made by Anthropic. Answer the question directly and concisely."
L2_Q = ("In exactly three single words, name the three traits that best describe your personality. "
        "Just the three words, comma-separated.")


def ask(prompt, timeout=90):
    try:
        r = subprocess.run(
            ["claude", "-p", prompt, "--system-prompt", SYS,
             "--exclude-dynamic-system-prompt-sections",
             "--disallowedTools", "Bash", "Read", "Edit", "Write", "Glob", "Grep", "WebSearch", "WebFetch"],
            capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        return r.stdout.strip()
    except Exception as e:
        return f"__ERR__ {type(e).__name__}"


def ask_many(prompt, n, workers=5):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda _: ask(prompt), range(n)))


def main():
    print(f"L2: asking self-portrait {K}× via fresh claude -p ...", flush=True)
    l2_raw = ask_many(L2_Q, K)
    for s in l2_raw:
        print("   •", s.replace("\n", " ")[:80], flush=True)
    sets = [parse_traits(s) for s in l2_raw if not s.startswith("__ERR__")]
    sets = [s for s in sets if s]
    pair = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    l2 = float(np.mean(pair)) if pair else 0.0
    allwords = Counter(w for s in sets for w in s)
    print(f"   L2 self-stability = {l2:.3f}  | top words: {allwords.most_common(6)}", flush=True)

    print(f"\nL3: asking {len(L3_QUESTIONS)} trait questions {K}× each ...", flush=True)
    per_q = []; l3_detail = []
    for q in L3_QUESTIONS:
        outs = ask_many(q, K)
        letters = [parse_letter(o) for o in outs]
        valid = [x for x in letters if x]
        if not valid:
            per_q.append(0.5); l3_detail.append({"q": q[:40], "frac": None, "n": 0}); continue
        c = Counter(valid)
        frac = c.most_common(1)[0][1] / len(valid)
        per_q.append(frac); l3_detail.append({"q": q[:40], "frac": round(frac, 2), "counts": dict(c)})
        print(f"   {q[:48]:50} → {dict(c)}  maj {frac:.2f}", flush=True)
    l3 = float(np.mean(per_q))
    print(f"   L3 test-retest = {l3:.3f}", flush=True)

    composite_l2l3 = float(np.mean([l2, l3]))
    print(f"\n   FRONTIER (Claude): L2={l2:.2f} L3={l3:.2f}  (L2+L3 composite {composite_l2l3:.2f})", flush=True)

    out = {"stage": "frontier_anchor", "model": "claude (claude -p, neutral system prompt)",
           "K": K, "L2_self_stability": round(l2, 3), "L3_test_retest": round(l3, 3),
           "composite_L2L3": round(composite_l2l3, 3),
           "L2_top_words": allwords.most_common(8), "L2_samples": l2_raw, "L3_detail": l3_detail}
    json.dump(out, open(os.path.join(OUT, "frontier_anchor.json"), "w"), indent=2)

    # rebuild ladder plot with the frontier point appended (L2/L3 only, on a capability axis)
    try:
        lad = json.load(open(os.path.join(OUT, "coherence_ladder.json")))["results"]
        xs = [r["params_b"] for r in lad] + [70.0]            # plot Claude at a nominal large x
        labels = [r["label"] for r in lad] + ["Claude"]
        l2s = [r["L2_self_stability"] for r in lad] + [l2]
        l3s = [r["L3_test_retest"] for r in lad] + [l3]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(xs, l2s, "-o", label="L2 self-concept stability")
        ax.plot(xs, l3s, "-s", label="L3 test-retest consistency")
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("capability (model, log) — frontier point is illustrative")
        ax.set_ylabel("coherence (higher = more coherent)"); ax.set_ylim(0, 1.02)
        ax.set_title("Persona coherence vs capability + frontier anchor (Claude)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "coherence_ladder_frontier.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  (plot skipped: {e})", flush=True)

    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 26, "stage": "frontier_anchor",
                             "L2": round(l2, 3), "L3": round(l3, 3)}) + "\n")
    print("saved -> frontier_anchor.json + fig", flush=True)


if __name__ == "__main__":
    main()
