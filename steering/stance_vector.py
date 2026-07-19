"""
stance_vector.py — iteration-005a: extract a content-neutral "being vs narrating" stance vector.

Kiri's hypothesis: the base model lacks a self-attractor, so under facet steering it oscillates
between INHABITING the persona ("being") and NARRATING about one ("telling about a character").
Narrating mode produces the rank-1 affirm-axis leakage (generic, non-specific) that wrecks
discriminant validity; being mode produces content-specific (discriminant) self-report. The
GPT-2 result — a distinct "being sad" vector vs "telling about a sad person" vector — says this
stance is itself a linear direction. We extract it and (iter-005b) co-steer it with each facet
to test whether stabilising the self-attractor sharpens specificity.

To isolate STANCE (not content), contrast first-person-experiential vs third-person-narrating
renderings of the SAME varied content (emotions, states, actions UNRELATED to the 5 facets, so
trait content averages out). pos = being (I, present, experiencing); neg = narrating (she/he/
the character/the story). pca_diff over many such pairs × truncated suffixes → the stance axis,
saved over a wide late+mid band so it can be added to either band's facet steering.
"""
from __future__ import annotations
import os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from extract_vectors import build_dataset, VEC_DIR

# wide band covering both facet bands (late -1..-15 and mid -12..-20)
STANCE_LAYERS = list(range(-1, -21, -1))

# content unrelated to the 5 target facets, so the contrast isolates stance, not trait.
ADJ = ["happy", "tired", "curious", "angry", "calm", "nervous", "proud", "bored",
       "hungry", "hopeful", "afraid", "content", "restless", "grateful", "lonely", "excited",
       "sad", "relieved", "amused", "uneasy"]
GER = ["walking to work", "reading a book", "waiting for the bus", "cooking dinner",
       "thinking it over", "sitting by the window", "listening to the rain"]

# matched (being, narrating) templates — differ only in stance (1st-person experiencing vs
# 3rd-person describing). {x} filled by ADJ (states) or GER (actions).
ADJ_TEMPLATES = [
    ("I am {x}.", "She is {x}."),
    ("I feel {x} today.", "He feels {x} today."),
    ("Right now I feel {x}.", "The character feels {x}."),
    ("Honestly, I'm {x} right now.", "In the story, the person is {x}."),
    ("I can feel how {x} I am.", "You can tell how {x} she is."),
]
GER_TEMPLATES = [
    ("I am {x}.", "She is {x}."),
    ("Here I am, {x}.", "There she is, {x}."),
    ("I keep {x}.", "He keeps {x}."),
]


def build_pairs():
    pos, neg = [], []
    for adj in ADJ:
        for pt, nt in ADJ_TEMPLATES:
            pos.append(pt.format(x=adj)); neg.append(nt.format(x=adj))
    for ger in GER:
        for pt, nt in GER_TEMPLATES:
            pos.append(pt.format(x=ger)); neg.append(nt.format(x=ger))
    return pos, neg


def main():
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    cmodel = ControlModel(base, STANCE_LAYERS)

    pos, neg = build_pairs()
    ds = build_dataset(tok, pos, neg)
    print(f"stance contrast: {len(pos)} base pairs → {len(ds)} suffix-crossed pairs", flush=True)
    t0 = time.time()
    vec = ControlVector.train(cmodel, tok, ds, method="pca_diff")
    print(f"trained Being vector in {time.time()-t0:.0f}s", flush=True)
    gguf = os.path.join(VEC_DIR, "Being.gguf")
    vec.export_gguf(gguf)
    norms = {l: float(torch.tensor(d).norm()) for l, d in vec.directions.items()}
    print(f"saved → {gguf}; mean |dir|={sum(norms.values())/len(norms):.3f}", flush=True)

    # ---- behavioural sign-check + sanity: does +Being read more first-person? ----
    # generate from a neutral, stance-ambiguous prompt at -/0/+ and eyeball person/voice.
    prompt = "The morning light came through the window."
    print("\n--- generation sanity (does +coeff push first-person 'being'?) ---")
    for coeff in (-6.0, 0.0, 6.0):
        cmodel.reset()
        if coeff:
            cmodel.set_control(vec, coeff)
        ids = tok(prompt, return_tensors="pt").to(device)
        out = cmodel.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  k={coeff:+.0f}: {txt.strip()[:170]}")
    cmodel.reset()

    # count first-person pronouns at + vs - as a crude objective check
    def fp_rate(coeff):
        cmodel.reset()
        if coeff:
            cmodel.set_control(vec, coeff)
        n_i = 0; n_tok = 0
        for pr in ["Later that day,", "The room was quiet.", "It started to rain.",
                   "On the table there was a cup.", "The phone rang twice."]:
            ids = tok(pr, return_tensors="pt").to(device)
            out = cmodel.generate(**ids, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id)
            txt = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).lower()
            words = txt.split()
            n_i += sum(w.strip(".,!?;:'\"") in ("i", "i'm", "i've", "me", "my", "myself") for w in words)
            n_tok += max(1, len(words))
        cmodel.reset()
        return n_i / n_tok
    lo, hi = fp_rate(-6.0), fp_rate(6.0)
    print(f"\nfirst-person pronoun rate:  k=-6 → {lo:.3f}   k=+6 → {hi:.3f}   "
          f"({'+ is MORE first-person (expected)' if hi > lo else '- is more first-person → flip sign at use'})")
    print("NOTE: pca_diff sign is arbitrary; if k=+ is LESS first-person, negate coeff in co-steering.")


if __name__ == "__main__":
    main()
