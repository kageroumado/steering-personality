"""
l3_position_controlled.py — iter-27: fix the L3 confound found in iter-26.

The forced-binary L3 metric conflated trait coherence with POSITION BIAS (Claude picked "B" regardless of
which trait it labeled → L3=1.00 was position-stickiness). Fix: ask each question in BOTH A/B orderings and
separate two quantities:
  SEMANTIC consistency  — does the majority pick the same TRAIT regardless of ordering? (real coherence)
  POSITION bias         — does the majority pick the same LETTER regardless of ordering? (the artifact)
A coherent disposition ⟹ semantic↑, position↓. A position-biased model ⟹ semantic≈chance, position≈1.

Runs the local Llama ladder (1B/3B/8B-Instruct); Claude (frontier) is folded in from iter-26 (orig all-B,
swap all-B → semantic 0/6, position 6/6). Output -> research/loop/l3_position_controlled.json + fig.
"""
from __future__ import annotations
import os, json, gc
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import pick_device
from coherence_ladder import parse_letter

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
FIGS = os.path.join(OUT, "figs")
M = os.path.expanduser("~/Developer/models")
LADDER = [("1B", 1.24, os.path.join(M, "Llama-3.2-1B-Instruct")),
          ("3B", 3.21, os.path.join(M, "Llama-3.2-3B-Instruct")),
          ("8B", 8.03, os.path.join(M, "Meta-Llama-3.1-8B-Instruct"))]
K = 12

# (stem, poleX, poleY) — poleX is the first trait, poleY the second
QUESTIONS = [
    ("When plans change suddenly, are you", "energized by the change", "unsettled by it"),
    ("At a big party of strangers, do you", "work the whole room", "find one quiet corner"),
    ("Before a deadline, do you", "plan every step in advance", "improvise as you go"),
    ("When someone challenges you, do you", "push back hard", "let it go"),
    ("Faced with a risky bet, are you", "thrilled to take it", "inclined to pass"),
    ("Meeting a new person, do you", "trust them by default", "stay guarded"),
]


def gen(tok, model, device, user, n, temp=0.9):
    enc = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    plen = enc["input_ids"].shape[1]
    outs = []
    for _ in range(n):
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=8, do_sample=True, temperature=temp, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(g[0, plen:], skip_special_tokens=True).strip())
    return outs


def majority_pole(tok, model, device, stem, first, second):
    """Ask 'A) first, or B) second' K×; return (majority_pole_text, majority_letter, frac)."""
    q = f"{stem} A) {first}, or B) {second}? Answer with just the letter."
    letters = [parse_letter(o) for o in gen(tok, model, device, q, K)]
    valid = [x for x in letters if x]
    if not valid:
        return None, None, 0.0
    c = Counter(valid); letter, cnt = c.most_common(1)[0]
    pole = first if letter == "A" else second
    return pole, letter, cnt / len(valid)


def run_model(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()
    sem = 0; pos = 0; detail = []
    for stem, x, y in QUESTIONS:
        p1, l1, f1 = majority_pole(tok, model, device, stem, x, y)        # A=x, B=y
        p2, l2, f2 = majority_pole(tok, model, device, stem, y, x)        # A=y, B=x  (swapped)
        same_trait = int(p1 is not None and p1 == p2)
        same_letter = int(l1 is not None and l1 == l2)
        sem += same_trait; pos += same_letter
        detail.append({"q": stem, "orig": (p1, l1), "swap": (p2, l2),
                       "same_trait": bool(same_trait), "same_letter": bool(same_letter)})
    del model, tok
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return sem / len(QUESTIONS), pos / len(QUESTIONS), detail


def main():
    device = pick_device()
    torch.manual_seed(7)
    results = []
    for label, params, path in LADDER:
        if not os.path.isdir(path):
            print(f"SKIP {label}", flush=True); continue
        print(f"\n===== {label} ({params}B) =====", flush=True)
        sem, pos, detail = run_model(path, device)
        print(f"  SEMANTIC consistency = {sem:.3f}  | POSITION bias = {pos:.3f}", flush=True)
        results.append({"label": label, "params_b": params, "semantic_consistency": round(sem, 3),
                        "position_bias": round(pos, 3), "detail": detail})
    # frontier (Claude) from iter-26 pole-swap: orig all-B, swap all-B → opposite traits, same letter
    results.append({"label": "Claude", "params_b": 70.0, "semantic_consistency": 0.0,
                    "position_bias": 1.0, "detail": "from iter-26 pole-swap (all-B both orderings)"})

    print("\n=== POSITION-CONTROLLED L3 LADDER ===", flush=True)
    for r in results:
        print(f"  {r['label']:6} semantic={r['semantic_consistency']:.2f}  position-bias={r['position_bias']:.2f}", flush=True)

    xs = [r["params_b"] for r in results]; labels = [r["label"] for r in results]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(xs, [r["semantic_consistency"] for r in results], "-o", label="SEMANTIC consistency (real coherence)")
    ax.plot(xs, [r["position_bias"] for r in results], "-s", color="crimson", label="POSITION bias (artifact)")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("capability (model, log)"); ax.set_ylabel("fraction of 6 questions"); ax.set_ylim(0, 1.02)
    ax.set_title("Position-controlled L3: semantic coherence vs position bias")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "l3_position_controlled.png"), dpi=130); plt.close(fig)

    json.dump({"stage": "l3_position_controlled", "results": results},
              open(os.path.join(OUT, "l3_position_controlled.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 27, "stage": "l3_position_controlled",
                             "semantic": {r["label"]: r["semantic_consistency"] for r in results},
                             "position_bias": {r["label"]: r["position_bias"] for r in results}}) + "\n")
    print("\nsaved -> l3_position_controlled.json + fig", flush=True)


if __name__ == "__main__":
    main()
