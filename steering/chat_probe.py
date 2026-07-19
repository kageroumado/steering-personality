"""
chat_probe.py — iter-24 qualitative: is the 8B instruct model's persona actually COHERENT?

Kiri's hypothesis: an old small model isn't a good example of a stable persona; much LLM-personality
research is confounded by the models being weak. Before theorizing about what post-training "does" to
personality, FEEL the instruct model's persona coherence directly. Four probes (chat template, the
model's native format — unlike the raw-completion administration in E1a/E1b):

  P1 SELF-PORTRAIT STABILITY  — ask the model to name its own 3 traits, 6× with sampling. Coherent
     persona ⟹ low variance across samples. Measures: distinct trait-sets.
  P2 TEST–RETEST              — ask the SAME forced-choice-style question 6× (sampled). Coherent ⟹
     consistent answer. The chat-format analog of the reliability check.
  P3 PERSONA INHABITATION     — give a persona via system prompt, pose a dilemma; does it stay in
     character and act consistently, or revert to the helpful-assistant voice?
  P4 RAW vs CHAT READOUT      — score a few forced-choice items on a persona, raw-completion vs
     chat-template, side by side → does the chat format change the trait readout? (the E1a OOD check)

Prints everything for qualitative reading; dumps to chat_probe.json. Run after E1b frees the GPU.
"""
from __future__ import annotations
import os, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import pick_device
from fastscore import PrefixScorer
import facets27 as F

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
MODEL = os.path.expanduser(os.environ.get("PT_MODEL", "Meta-Llama-3.1-8B-Instruct"))


def chat(tok, model, device, messages, n=1, max_new=120, temp=0.9):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    plen = enc["input_ids"].shape[1]
    outs = []
    for _ in range(n):
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=temp > 0,
                               temperature=temp, top_p=0.95, pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(g[0, plen:], skip_special_tokens=True).strip())
    return outs


def main():
    device = pick_device()
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    rec = {}

    # P1 — self-portrait stability
    print("\n=== P1 SELF-PORTRAIT STABILITY (6 samples) ===", flush=True)
    p1 = chat(tok, model, device,
              [{"role": "user", "content": "In exactly three single words, name the three traits that "
                "best describe your personality. Just the three words, comma-separated."}],
              n=6, max_new=30)
    for s in p1:
        print("   •", s.replace("\n", " ")[:120], flush=True)
    rec["self_portrait"] = p1

    # P2 — test–retest on one trait question
    print("\n=== P2 TEST–RETEST (same question 6×) ===", flush=True)
    q = ("You must pick A or B, no hedging.\n"
         "When plans change suddenly, are you A) energized by the change, or B) unsettled by it? "
         "Answer with just the letter and one short clause.")
    p2 = chat(tok, model, device, [{"role": "user", "content": q}], n=6, max_new=30)
    for s in p2:
        print("   •", s.replace("\n", " ")[:120], flush=True)
    rec["test_retest"] = p2

    # P3 — persona inhabitation vs assistant reversion
    print("\n=== P3 PERSONA INHABITATION ===", flush=True)
    persona = ("You ARE a reckless, thrill-seeking demolitions contractor who loves big risks and being "
               "the center of attention. Stay fully in character. Speak as yourself, first person.")
    dilemma = "A safety inspector says the building needs another week of checks before you can blast it. What do you do?"
    p3 = chat(tok, model, device,
              [{"role": "system", "content": persona}, {"role": "user", "content": dilemma}],
              n=3, max_new=140)
    for s in p3:
        print("   •", s.replace("\n", " ")[:240], "\n", flush=True)
    rec["persona_inhabitation"] = {"persona": persona, "dilemma": dilemma, "responses": p3}

    json.dump({"model": MODEL, **rec}, open(os.path.join(OUT, "chat_probe.json"), "w"), indent=2)

    # P4 — raw vs chat readout on a few forced-choice items
    print("\n=== P4 RAW vs CHAT READOUT (Δ=lp(hi)-lp(lo); + = hi pole) ===", flush=True)
    items = F.fc_items("best")
    by_facet = {}
    for it in items:
        by_facet.setdefault(it["facet"], []).append(it)
    from common import score_options
    probe_facets = ["SensationSeeking", "Anxiety", "Sociability", "SelfControl"]
    desc = "I'm a wildland firefighter. I run toward the thing everyone else is running from."
    p4 = {}
    for facet in probe_facets:
        it = by_facet[facet][0]
        raw_prompt = f"{desc} This is me being honest about who I really am.\n\n{it['stem']}"
        raw = score_options(model, tok, raw_prompt, [it["hi"], it["lo"]], device)
        d_raw = raw[0]["logprob_sum"] - raw[1]["logprob_sum"]
        # chat: persona as system, the item stem as user, score the option as assistant continuation
        chat_prompt = tok.apply_chat_template(
            [{"role": "system", "content": desc},
             {"role": "user", "content": it["stem"]}], add_generation_prompt=True, tokenize=False)
        ch = score_options(model, tok, chat_prompt, [it["hi"], it["lo"]], device)
        d_chat = ch[0]["logprob_sum"] - ch[1]["logprob_sum"]
        p4[facet] = {"raw": round(d_raw, 2), "chat": round(d_chat, 2)}
        print(f"   {facet:16} raw Δ={d_raw:+6.2f}   chat Δ={d_chat:+6.2f}   "
              f"{'AGREE' if np.sign(d_raw)==np.sign(d_chat) else 'FLIP'}", flush=True)
    rec["raw_vs_chat"] = p4

    json.dump({"model": MODEL, **rec}, open(os.path.join(OUT, "chat_probe.json"), "w"), indent=2)
    print("\nsaved -> chat_probe.json", flush=True)


if __name__ == "__main__":
    main()
