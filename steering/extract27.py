"""
extract27.py — breadth 5→27 (D13): extract steering vectors for the remaining 22 SPI facets.

Same canonical repeng recipe as extract_vectors.py (pca_diff over high/low contrast
statements × truncated neutral suffixes; one GGUF control-vector per facet, directions stored
for all layers so any band can be probed). Content comes from facets27.FACETS. By default we
skip facets whose .gguf already exists (the 5 validated core are verbatim copies → identical
vectors); pass --force to re-extract all 27.
"""
from __future__ import annotations
import os, json, time, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
from common import DEFAULT_MODEL, pick_device
from extract_vectors import VEC_DIR, LAYER_IDS, SUFFIXES, MAX_PAIRS, build_dataset
import facets27 as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true", help="re-extract even if gguf exists")
    ap.add_argument("--facets", nargs="*", default=F.FACET_ORDER)
    args = ap.parse_args()

    os.makedirs(VEC_DIR, exist_ok=True)
    todo = []
    for f in args.facets:
        gguf = os.path.join(VEC_DIR, f"{f}.gguf")
        if os.path.exists(gguf) and not args.force:
            print(f"skip {f} (exists)", flush=True)
        else:
            todo.append(f)
    print(f"\nextracting {len(todo)} vectors: {todo}\n", flush=True)
    if not todo:
        return

    device = pick_device()
    print(f"loading {args.model} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    cmodel = ControlModel(base, LAYER_IDS)

    prov_path = os.path.join(VEC_DIR, "provenance27.json")
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else \
        {"model": args.model, "method": "pca_diff", "layer_ids": LAYER_IDS,
         "max_pairs": MAX_PAIRS, "source": "facets27.FACETS", "vectors": {}}

    t_all = time.time()
    for i, facet in enumerate(todo):
        spec = F.FACETS[facet]
        ds = build_dataset(tok, spec["pos"], spec["neg"])
        t0 = time.time()
        vec = ControlVector.train(cmodel, tok, ds, method="pca_diff")
        dt = time.time() - t0
        gguf = os.path.join(VEC_DIR, f"{facet}.gguf")
        vec.export_gguf(gguf)
        norms = {l: float(torch.tensor(d).norm()) for l, d in vec.directions.items()}
        prov["vectors"][facet] = {"gguf": gguf, "train_seconds": round(dt, 1),
                                  "n_pairs": len(ds), "n_layers": len(norms),
                                  "mean_norm": round(sum(norms.values()) / len(norms), 3)}
        print(f"[{i+1}/{len(todo)}] {facet:24} {dt:5.0f}s  "
              f"mean|dir|={prov['vectors'][facet]['mean_norm']:.3f}  "
              f"({time.time()-t_all:.0f}s total)", flush=True)
        json.dump(prov, open(prov_path, "w"), indent=2)

    print(f"\ndone — {len(todo)} vectors in {time.time()-t_all:.0f}s; provenance → {prov_path}",
          flush=True)


if __name__ == "__main__":
    main()
