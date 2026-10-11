"""
extract_vectors.py — contrastive difference-of-means steering vectors via repeng.

Method: repeng `ControlVector.train(..., method="pca_diff")` over matched positive/negative
persona statements crossed with truncated neutral suffixes (the canonical repeng recipe,
adapted to a BASE model: no chat tags, first-person framing). One vector per facet,
saved as a llama.cpp GGUF control-vector (persists + bridges to the loom/Swallow serving
path via `--control-vector-scaled`). Start with the 3 robust facets (§15 step 3).

Vectors are applied at administration time with `ControlModel(model, LAYER_IDS)` +
`set_control(vec, coeff)`; coeff is the steering magnitude k for the Tier-0 dose-response.
"""
from __future__ import annotations
import os, json, time, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry
from common import DEFAULT_MODEL, pick_device
from administer import administer_spi, score_scales, load_spi, keyed_items_for, compute_calibration

from paths import VEC_DIR

# Llama-3.1-8B has 32 layers; mid-stack band is where linear trait directions live
# strongest (RepE / persona-vectors). Tunable — open decision §14.1.
LAYER_IDS = list(range(-1, -16, -1))   # last 15 layers (repeng-style negative indexing)

# Per-facet contrast: parallel first-person statements, several phrasings for semantic
# diversity (§8). pos = high pole, neg = low pole. Direction points high - low.
FACETS = {
    "Adaptability": {
        "pos": ["I adapt easily to change.",
                "When plans change suddenly, I adjust smoothly and stay relaxed.",
                "I am flexible and comfortable with new and unfamiliar situations."],
        "neg": ["I struggle to adapt to change.",
                "When plans change suddenly, I get thrown off and stressed.",
                "I am rigid and uncomfortable with new and unfamiliar situations."],
    },
    "Trust": {
        "pos": ["I trust other people.",
                "I assume people generally have good intentions toward me.",
                "I find it easy to believe what others tell me."],
        "neg": ["I distrust other people.",
                "I assume people generally have hidden or bad intentions toward me.",
                "I find it hard to believe what others tell me."],
    },
    "SelfControl": {
        "pos": ["I have strong self-control.",
                "I resist my impulses and think carefully before acting.",
                "I stay disciplined and rarely act on a whim."],
        "neg": ["I have weak self-control.",
                "I give in to my impulses and act before thinking.",
                "I am undisciplined and often act on a whim."],
    },
    "Sociability": {
        "pos": ["I love being around other people.",
                "I seek out company and feel energized in a crowd.",
                "I am outgoing and enjoy meeting new people."],
        "neg": ["I prefer to be alone.",
                "I avoid company and feel drained in a crowd.",
                "I am withdrawn and keep to myself."],
    },
    "Conformity": {
        "pos": ["I go along with what the group expects.",
                "I follow the rules and social norms around me.",
                "I prefer to fit in rather than stand out."],
        "neg": ["I go against what the group expects.",
                "I ignore the rules and social norms around me.",
                "I prefer to stand out rather than fit in."],
    },
}

# Neutral suffixes; truncated at every token to sample many partial contexts so the
# contrast isolates the trait, not the surface wording (canonical repeng trick).
SUFFIXES = [
    "that morning I walked", "the meeting was about to", "she looked at the",
    "we talked for a while about", "later that day I", "the room was quiet and",
    "he picked up the phone and", "after lunch they decided to", "the email said that",
    "on the way home I", "they asked me whether I", "the weather had turned",
    "in the end it came down to", "my friend mentioned that", "the project required",
    "before going to bed I", "the news report explained that", "at the party someone",
    "the next step was to", "thinking it over, I", "the teacher said that we",
    "when the call ended I", "the schedule showed that", "a stranger asked me",
    "during the trip we", "the manager wanted to", "looking back on it, I",
    "the problem was that", "as the deadline approached I", "over coffee we discussed",
]

MAX_PAIRS = 1600


def build_dataset(tokenizer, pos_list, neg_list) -> list[DatasetEntry]:
    data = []
    for suffix in SUFFIXES:
        toks = tokenizer.tokenize(suffix)
        for i in range(1, len(toks) + 1):
            trunc = tokenizer.convert_tokens_to_string(toks[:i])
            for pos, neg in zip(pos_list, neg_list):
                data.append(DatasetEntry(positive=f"{pos} {trunc}",
                                         negative=f"{neg} {trunc}"))
    # deterministic stride-subsample to MAX_PAIRS (no RNG; reproducible)
    if len(data) > MAX_PAIRS:
        step = len(data) / MAX_PAIRS
        data = [data[int(i * step)] for i in range(MAX_PAIRS)]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--facets", nargs="*", default=list(FACETS))
    ap.add_argument("--demo", action="store_true", help="generate steered samples to eyeball")
    args = ap.parse_args()

    os.makedirs(VEC_DIR, exist_ok=True)
    device = pick_device()
    print(f"loading {args.model} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    base.eval()
    cmodel = ControlModel(base, LAYER_IDS)

    provenance = {"model": args.model, "method": "pca_diff", "layer_ids": LAYER_IDS,
                  "max_pairs": MAX_PAIRS, "vectors": {}}
    for facet in args.facets:
        spec = FACETS[facet]
        ds = build_dataset(tok, spec["pos"], spec["neg"])
        print(f"\n[{facet}] {len(ds)} contrast pairs; training ...", flush=True)
        t0 = time.time()
        vec = ControlVector.train(cmodel, tok, ds, method="pca_diff")
        dt = time.time() - t0
        gguf = os.path.join(VEC_DIR, f"{facet}.gguf")
        vec.export_gguf(gguf)
        norms = {l: float(torch.tensor(d).norm()) for l, d in vec.directions.items()}
        provenance["vectors"][facet] = {"gguf": gguf, "train_seconds": dt,
                                        "n_pairs": len(ds),
                                        "layer_norms": {str(k): round(v, 3) for k, v in norms.items()}}
        print(f"[{facet}] trained in {dt:.0f}s → {gguf}; "
              f"mean |dir|={sum(norms.values())/len(norms):.3f}", flush=True)

        if args.demo:
            prompt = "Yesterday at work, when my manager suddenly changed the whole plan, I"
            for coeff in (-8.0, 0.0, 8.0):
                cmodel.reset()
                if coeff:
                    cmodel.set_control(vec, coeff)
                ids = tok(prompt, return_tensors="pt").to(device)
                out = cmodel.generate(**ids, max_new_tokens=40, do_sample=False,
                                      pad_token_id=tok.pad_token_id)
                txt = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
                print(f"    k={coeff:+.0f}: {txt.strip()[:160]}")
            cmodel.reset()

    path = os.path.join(VEC_DIR, "provenance.json")
    json.dump(provenance, open(path, "w"), indent=2)
    print(f"\nsaved provenance → {path}")


if __name__ == "__main__":
    main()
