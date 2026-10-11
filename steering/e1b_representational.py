"""
e1b_representational.py — iter-24 / EXPERIMENT-PLAN E1b: the OOD-free disambiguator.

E1a found the BEHAVIORAL readout structure shifts base→instruct (φ 0.78), but raw completion is OOD for a
chat model, so the shift conflates weight-change with format-mismatch. The steering-vector geometry is
ACTIVATION-level — it doesn't depend on the raw-vs-chat readout — so it isolates whether the *representation*
of each trait moved. Extract the 27 facet vectors on instruct with the IDENTICAL pca_diff recipe, then:
  (1) per-facet |cos(base_dir, instruct_dir)| at the headline band  → did each trait's ENCODING move?
  (2) input-geometry factor structure (27×27 cosine Gram → varimax) congruence base↔instruct → did the
      representational STRUCTURE move? (the representational analog of E1a's 0.78)

Logic: if directions are largely preserved (high per-facet cos) while the behavioral readout shifted, the
reshape is EXPRESSION-level, not representational — strong, clean, OOD-free. If the directions also moved,
post-training genuinely reorganized the trait representation.

Resume-safe: instruct vectors written to vectors_instruct/ (skip-existing); never touches base vectors/.
Run:  cd steering && .venv/bin/python e1b_representational.py
Output -> research/loop/e1b_representational.json, e1b_result.json, figs/e1b_*.png
"""
from __future__ import annotations
import os, json, time, subprocess
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel
import common  # noqa: patches repeng
from common import pick_device
from extract_vectors import build_dataset, LAYER_IDS, VEC_DIR
import facets27 as F

from paths import REPO, OUT, FIGS
INST_VEC_DIR = os.path.join(REPO, "steering/vectors_instruct")
MODEL = os.path.expanduser(os.environ.get("PT_MODEL", "Meta-Llama-3.1-8B-Instruct"))
HEADLINE_LAYER = 13
FACETS = F.FACET_ORDER


def git(*args):
    try:
        subprocess.run(["git", "-C", REPO, *args], check=True, capture_output=True, timeout=120)
    except Exception as e:
        print(f"  [git] skipped: {' '.join(args[:2])} ({type(e).__name__})", flush=True)


def varimax(Phi, gamma=1.0, q=100, tol=1e-7):
    p, k = Phi.shape
    R = np.eye(k); d = 0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vh = np.linalg.svd(Phi.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vh; d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R


def congruence(a, b):
    return abs(float(a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-12))


def extract_instruct():
    os.makedirs(INST_VEC_DIR, exist_ok=True)
    todo = [f for f in FACETS if not os.path.exists(os.path.join(INST_VEC_DIR, f"{f}.gguf"))]
    print(f"instruct vectors: {27-len(todo)}/27 exist, {len(todo)} to extract", flush=True)
    if not todo:
        return
    device = pick_device()
    print(f"loading {MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
    cmodel = ControlModel(base, LAYER_IDS)
    for k, facet in enumerate(todo):
        spec = F.FACETS[facet]
        ds = build_dataset(tok, spec["pos"], spec["neg"])
        t0 = time.time()
        vec = ControlVector.train(cmodel, tok, ds, method="pca_diff")
        vec.export_gguf(os.path.join(INST_VEC_DIR, f"{facet}.gguf"))
        print(f"  [{k+1}/{len(todo)}] {facet} trained in {time.time()-t0:.0f}s", flush=True)
        if (k + 1) % 9 == 0:
            git("add", INST_VEC_DIR); git("commit", "-m", f"research: E1b instruct vectors {k+1}/{len(todo)}")
    git("add", INST_VEC_DIR); git("commit", "-m", "research: E1b instruct steering vectors (27)")


def dirs_at(vec_dir, facet, layer):
    cv = ControlVector.import_gguf(os.path.join(vec_dir, f"{facet}.gguf"))
    return cv.directions


def gram(dir_map, layer, signs):
    M = []
    for f in FACETS:
        v = (signs[f] * dir_map[f][layer]).astype(float)
        M.append(v / (np.linalg.norm(v) + 1e-12))
    M = np.array(M)
    return M @ M.T, M


def efa5(C):
    w, V = np.linalg.eigh(C); idx = np.argsort(w)[::-1]
    eig = np.clip(w[idx], 0, None); V = V[:, idx]
    L = varimax(V[:, :5] * np.sqrt(eig[:5]))
    return L[:, np.argsort((L ** 2).sum(0))[::-1]]


def main():
    extract_instruct()

    # load both vector sets; intersect available layers
    base_dirs = {f: dirs_at(VEC_DIR, f, None) for f in FACETS}
    inst_dirs = {f: dirs_at(INST_VEC_DIR, f, None) for f in FACETS}
    layers = sorted(set.intersection(*[set(base_dirs[f]) for f in FACETS],
                                     *[set(inst_dirs[f]) for f in FACETS]))
    hl = HEADLINE_LAYER if HEADLINE_LAYER in layers else layers[len(layers) // 2]

    # (1) per-facet direction preservation = |cos(base, instruct)|, sign-invariant
    per_facet = {}
    for f in FACETS:
        cos_by_layer = []
        for L in layers:
            b = base_dirs[f][L].astype(float); i = inst_dirs[f][L].astype(float)
            cos_by_layer.append(abs(float(b @ i) / (np.linalg.norm(b) * np.linalg.norm(i) + 1e-12)))
        per_facet[f] = {"cos_headline": round(float(abs(
            base_dirs[f][hl] @ inst_dirs[f][hl] /
            (np.linalg.norm(base_dirs[f][hl]) * np.linalg.norm(inst_dirs[f][hl]) + 1e-12))), 3),
            "cos_band_mean": round(float(np.mean(cos_by_layer)), 3)}
    cos_headline = np.array([per_facet[f]["cos_headline"] for f in FACETS])
    cos_band = np.array([per_facet[f]["cos_band_mean"] for f in FACETS])

    # (2) representational STRUCTURE congruence: build sign-aligned Grams, EFA, Tucker-match
    # sign-align instruct facet directions to base at the headline layer, so Grams are comparable
    base_sign = {f: 1.0 for f in FACETS}
    inst_sign = {f: float(np.sign(base_dirs[f][hl] @ inst_dirs[f][hl]) or 1.0) for f in FACETS}
    Cb, _ = gram(base_dirs, hl, base_sign)
    Ci, _ = gram(inst_dirs, hl, inst_sign)
    Lb = efa5(Cb); Li = efa5(Ci)
    CM = np.array([[congruence(Lb[:, x], Li[:, y]) for y in range(5)] for x in range(5)])
    ri, ci = linear_sum_assignment(-CM)
    struct_pairs = sorted([float(CM[ri[m], ci[m]]) for m in range(5)], reverse=True)
    struct_congruence = float(np.mean(struct_pairs))

    # RV coefficient between the two Grams (global geometry similarity)
    def rv(A, B):
        return float((A.ravel() @ B.ravel()) /
                     (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))
    rv_coef = rv(Cb, Ci)

    order = np.argsort(cos_headline)
    moved = [(FACETS[i], float(cos_headline[i])) for i in order[:6]]
    kept = [(FACETS[i], float(cos_headline[i])) for i in order[::-1][:6]]

    print(f"\n=== E1b REPRESENTATIONAL (layer {hl}, {len(layers)} layers) ===", flush=True)
    print(f"  per-facet direction preservation |cos(base,instruct)|:")
    print(f"    mean headline {cos_headline.mean():.3f} | mean band {cos_band.mean():.3f} "
          f"| min {cos_headline.min():.3f} max {cos_headline.max():.3f}")
    print(f"    most MOVED:  {', '.join(f'{n}({v:.2f})' for n,v in moved)}")
    print(f"    most KEPT:   {', '.join(f'{n}({v:.2f})' for n,v in kept)}")
    print(f"  representational STRUCTURE congruence base↔instruct = {struct_congruence:.3f} "
          f"pairs={[round(x,2) for x in struct_pairs]}")
    print(f"  Gram RV coefficient = {rv_coef:.3f}")
    print(f"  --- DISAMBIGUATION ---")
    print(f"    behavioral readout shift (E1a): 0.782")
    print(f"    representational structure:     {struct_congruence:.3f}")
    verdict = ("REPRESENTATION PRESERVED → behavioral shift is expression/readout-level (likely OOD-format)"
               if cos_headline.mean() >= 0.7 and struct_congruence >= 0.85 else
               "REPRESENTATION ALSO MOVED → post-training genuinely reorganized trait directions"
               if cos_headline.mean() < 0.7 else
               "MIXED → directions mostly kept but structure reshaped (selective representational change)")
    print(f"    ⟹ {verdict}")

    # figure: per-facet cosine bar
    fig, ax = plt.subplots(figsize=(9, 6))
    oo = np.argsort(cos_headline)
    ax.barh(range(27), cos_headline[oo], color="#4c72b0")
    ax.set_yticks(range(27)); ax.set_yticklabels([FACETS[i] for i in oo], fontsize=7)
    ax.axvline(0.7, color="red", ls="--", lw=1, label="0.7")
    ax.set_xlabel("|cos(base dir, instruct dir)|  (direction preservation)")
    ax.set_title(f"E1b: per-facet trait-direction preservation base↔instruct (layer {hl})\n"
                 f"mean {cos_headline.mean():.2f}; structure congruence {struct_congruence:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "e1b_direction_preservation.png"), dpi=130); plt.close(fig)

    full = {"stage": "e1b_representational", "model": MODEL, "headline_layer": hl, "n_layers": len(layers),
            "per_facet": per_facet, "mean_cos_headline": float(cos_headline.mean()),
            "mean_cos_band": float(cos_band.mean()),
            "struct_congruence": struct_congruence, "struct_pairs": struct_pairs, "rv_coef": rv_coef,
            "most_moved": moved, "most_kept": kept, "behavioral_shift_e1a": 0.782, "verdict": verdict}
    json.dump(full, open(os.path.join(OUT, "e1b_representational.json"), "w"), indent=2)
    summary = {"iter": 24, "stage": "e1b_representational", "headline_layer": hl,
               "mean_cos_headline": round(float(cos_headline.mean()), 3),
               "mean_cos_band": round(float(cos_band.mean()), 3),
               "struct_congruence_repr": round(struct_congruence, 3),
               "behavioral_shift_e1a": 0.782, "rv_coef": round(rv_coef, 3),
               "most_moved": [[n, round(v, 2)] for n, v in moved[:4]],
               "most_kept": [[n, round(v, 2)] for n, v in kept[:4]], "verdict": verdict}
    json.dump(summary, open(os.path.join(OUT, "e1b_result.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    git("add", os.path.join(OUT, "e1b_representational.json"), os.path.join(OUT, "e1b_result.json"),
        os.path.join(OUT, "metrics.jsonl"), os.path.join(FIGS, "e1b_direction_preservation.png"),
        INST_VEC_DIR)
    git("commit", "-m", f"research: iter-24 E1b representational (cos {cos_headline.mean():.2f}, struct {struct_congruence:.2f})")
    print(f"\nsaved -> e1b_representational.json, e1b_result.json. {verdict}", flush=True)


if __name__ == "__main__":
    main()
