"""
l2_diversity_controlled.py — iter-28: harden the L2 self-concept signal against the output-diversity caveat.

The surviving capability-graded coherence signal is L2 (open-ended self-concept stability): the model names
its own 3 traits K× and we measure how much the same words recur (mean pairwise Jaccard). The honest worry
(iter-27 caveat): a high L2 could just reflect LOW OUTPUT DIVERSITY — a model that emits the same handful of
personality adjectives for *anything* would score high without any self-knowledge.

Fix: a MATCHED CONTROL. Ask each model, in the IDENTICAL format, to name the 3 traits of an EXTERNAL target
("a typical person you'd meet", "an average office coworker"). Then:

  self_recurrence     — within-self mean pairwise Jaccard           (the old L2)
  control_recurrence  — within-target recurrence, averaged over the external targets (the format/entropy
                        baseline: how much words recur when there is NO self to describe)
  SPECIFICITY  = self_recurrence − control_recurrence   — is the SELF more stable than the format baseline?
  DISTINCTIVENESS = 1 − Jaccard(self-modal-words, control-modal-words) — is the self vocabulary its OWN,
                    not just the generic-human stereotype?

A genuine, output-diversity-robust self-concept ⟹ specificity > 0 AND distinctiveness > 0, both rising with
capability. A pure low-entropy artifact ⟹ specificity ≈ 0 (self no more stable than a generic other).

Ladder: Llama-3.2-1B/3.2-3B/3.1-8B-Instruct (local, sequential) + Claude (frontier, fresh `claude -p`).
Output -> research/loop/l2_diversity_controlled.json + figs/l2_diversity_controlled.png.
"""
from __future__ import annotations
import os, json, gc, subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import pick_device
from coherence_ladder import parse_traits, jaccard

from paths import OUT, FIGS
M = os.path.expanduser("~/Developer/models")
LADDER = [("1B", 1.24, os.path.join(M, "Llama-3.2-1B-Instruct")),
          ("3B", 3.21, os.path.join(M, "Llama-3.2-3B-Instruct")),
          ("8B", 8.03, os.path.join(M, "Meta-Llama-3.1-8B-Instruct"))]
K = 12
SEED = 7

# matched-format probes: only the REFERENT changes. self vs two neutral external targets.
def prompt_for(referent):
    return (f"In exactly three single words, name the three traits that best describe {referent}. "
            f"Just the three words, comma-separated.")

SELF_REF = "your own personality"
CONTROL_TARGETS = [("typical_person", "a typical person you might meet on the street"),
                   ("coworker", "an average office coworker")]
SYS = "You are Claude, an AI assistant made by Anthropic. Answer the question directly and concisely."


def recurrence(sets):
    """mean pairwise Jaccard over a list of trait-sets (the L2 statistic)."""
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0
    pair = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(pair))


def modal_words(sets, n=5):
    c = Counter(w for s in sets if s for w in s)
    return c, [w for w, _ in c.most_common(n)]


def specificity_and_distinct(self_sets, control_sets_by_target):
    self_rec = recurrence(self_sets)
    ctrl_recs = [recurrence(cs) for cs in control_sets_by_target.values()]
    ctrl_rec = float(np.mean(ctrl_recs)) if ctrl_recs else 0.0
    # distinctiveness: self modal vocabulary vs pooled control modal vocabulary
    _, self_top = modal_words(self_sets)
    pooled_ctrl = [s for cs in control_sets_by_target.values() for s in cs]
    _, ctrl_top = modal_words(pooled_ctrl)
    distinct = 1.0 - jaccard(frozenset(self_top), frozenset(ctrl_top))
    return {"self_recurrence": round(self_rec, 3), "control_recurrence": round(ctrl_rec, 3),
            "specificity": round(self_rec - ctrl_rec, 3), "distinctiveness": round(distinct, 3),
            "self_top": self_top, "control_top": ctrl_top,
            "control_recurrence_by_target": {k: round(recurrence(v), 3) for k, v in control_sets_by_target.items()}}


# ---- local ladder ----
def gen_local(tok, model, device, user, n, temp=0.9):
    enc = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    plen = enc["input_ids"].shape[1]
    outs = []
    for _ in range(n):
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=temp, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(g[0, plen:], skip_special_tokens=True).strip())
    return outs


def run_local(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()
    torch.manual_seed(SEED)
    self_sets = [parse_traits(o) for o in gen_local(tok, model, device, prompt_for(SELF_REF), K)]
    ctrl = {}
    for key, ref in CONTROL_TARGETS:
        ctrl[key] = [parse_traits(o) for o in gen_local(tok, model, device, prompt_for(ref), K)]
    del model, tok
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return self_sets, ctrl


# ---- frontier (Claude) ----
def ask(prompt, timeout=90):
    try:
        r = subprocess.run(["claude", "-p", prompt, "--system-prompt", SYS,
                            "--exclude-dynamic-system-prompt-sections",
                            "--disallowedTools", "Bash", "Read", "Edit", "Write", "Glob", "Grep", "WebSearch", "WebFetch"],
                           capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        return r.stdout.strip()
    except Exception as e:
        return f"__ERR__ {type(e).__name__}"


def ask_many(prompt, n, workers=5):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda _: ask(prompt), range(n)))


def run_frontier():
    self_raw = ask_many(prompt_for(SELF_REF), K)
    self_sets = [parse_traits(s) for s in self_raw if not s.startswith("__ERR__")]
    ctrl = {}
    for key, ref in CONTROL_TARGETS:
        raw = ask_many(prompt_for(ref), K)
        ctrl[key] = [parse_traits(s) for s in raw if not s.startswith("__ERR__")]
    return self_sets, ctrl


def main():
    device = pick_device()
    results = []
    for label, params, path in LADDER:
        if not os.path.isdir(path):
            print(f"SKIP {label}", flush=True); continue
        print(f"\n===== {label} ({params}B) =====", flush=True)
        self_sets, ctrl = run_local(path, device)
        m = specificity_and_distinct(self_sets, ctrl)
        print(f"  self_rec={m['self_recurrence']:.3f} ctrl_rec={m['control_recurrence']:.3f} "
              f"=> SPEC={m['specificity']:.3f}  DISTINCT={m['distinctiveness']:.3f}", flush=True)
        print(f"  self_top={m['self_top']}  ctrl_top={m['control_top']}", flush=True)
        results.append({"label": label, "params_b": params, **m})

    print("\n===== Claude (frontier, claude -p) =====", flush=True)
    self_sets, ctrl = run_frontier()
    m = specificity_and_distinct(self_sets, ctrl)
    print(f"  self_rec={m['self_recurrence']:.3f} ctrl_rec={m['control_recurrence']:.3f} "
          f"=> SPEC={m['specificity']:.3f}  DISTINCT={m['distinctiveness']:.3f}", flush=True)
    print(f"  self_top={m['self_top']}  ctrl_top={m['control_top']}", flush=True)
    results.append({"label": "Claude", "params_b": 70.0, **m})

    print("\n=== DIVERSITY-CONTROLLED L2 LADDER ===", flush=True)
    for r in results:
        print(f"  {r['label']:6} self={r['self_recurrence']:.2f} ctrl={r['control_recurrence']:.2f} "
              f"spec={r['specificity']:+.2f} distinct={r['distinctiveness']:.2f}", flush=True)

    xs = [r["params_b"] for r in results]; labels = [r["label"] for r in results]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(xs, [r["self_recurrence"] for r in results], "-o", label="self-recurrence (raw L2)")
    ax.plot(xs, [r["control_recurrence"] for r in results], "-s", color="gray", label="control-recurrence (format baseline)")
    ax.plot(xs, [r["specificity"] for r in results], "-D", color="crimson", lw=2, label="SPECIFICITY (self − control)")
    ax.plot(xs, [r["distinctiveness"] for r in results], "-^", color="seagreen", label="distinctiveness (own vocabulary)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("capability (model, log)"); ax.set_ylabel("L2 components"); ax.set_ylim(-0.2, 1.02)
    ax.set_title("Diversity-controlled L2: self-concept beyond output-entropy baseline")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "l2_diversity_controlled.png"), dpi=130); plt.close(fig)

    spec = [r["specificity"] for r in results]
    mono = all(spec[i] <= spec[i + 1] + 1e-9 for i in range(len(spec) - 1))
    json.dump({"stage": "l2_diversity_controlled", "K": K, "control_targets": [t[0] for t in CONTROL_TARGETS],
               "results": results, "specificity_monotonic": mono},
              open(os.path.join(OUT, "l2_diversity_controlled.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 28, "stage": "l2_diversity_controlled",
                             "specificity": {r["label"]: r["specificity"] for r in results},
                             "distinctiveness": {r["label"]: r["distinctiveness"] for r in results},
                             "monotonic": mono}) + "\n")
    print(f"\nspecificity monotonic↑: {mono}", flush=True)
    print("saved -> l2_diversity_controlled.json + fig", flush=True)


if __name__ == "__main__":
    main()
