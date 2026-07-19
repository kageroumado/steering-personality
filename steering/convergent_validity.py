"""
convergent_validity.py — iter-17: our fun forced-choice instrument vs the REAL SPI-135.

Closes the loop Kiri described: run the test on the model and validate against the actual
psychometric test we have. Administer the REAL SPI-135 (6-point IPIP Likert, validated keying,
27 scales + Big-Five) to the SAME 90 personas measured in iter-15b, persona-conditioned and
contextually calibrated (administer.py machinery). Then:

  1. CONVERGENT VALIDITY — per-facet Pearson r between our forced-choice reading and the real SPI
     reading across the 90 personas. Does our trait-unnamed quiz measure what the SPI measures?
  2. REPLICATE D20 WITH GROUND TRUTH — factor-analyze the real SPI 90×27, congruence vs the Big
     Five (using the SPI's OWN per-item B5 keying for the domain indicators). Does the real,
     validated instrument ALSO diverge from the Big Five in this model — or was D20 our instrument's
     artifact?
  3. CROSS-INSTRUMENT STRUCTURE — congruence between the FC and SPI native factor loadings. Do two
     very different instruments recover the same native structure?

Output -> research/loop/convergent_validity.json + figs/convergent_*.png.
"""
from __future__ import annotations
import os, json, time, gc
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
from administer import (load_spi, ANCHORS, ITEM_FMT, _fewshot_block, score_scales,
                        _ev_from_logps, CF_PROBES)
import facets27 as F

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
FIGS = os.path.join(OUT, "figs")
MODEL = os.environ.get("PT_MODEL", DEFAULT_MODEL)   # override for second-model replication
TAG = os.environ.get("PT_TAG", "")                  # output suffix + reads persona_population{TAG}.json


def load_lm(model_path, device):
    """Load an HF dir OR a GGUF (single .gguf in a dir, or a .gguf path) → (tok, model)."""
    import glob
    kw = {}
    if model_path.endswith(".gguf"):
        d, kw["gguf_file"] = os.path.dirname(model_path), os.path.basename(model_path)
    elif os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
        ggufs = glob.glob(os.path.join(model_path, "*.gguf"))
        d = model_path
        if ggufs:
            kw["gguf_file"] = os.path.basename(ggufs[0])
    else:
        d = model_path
    tok = AutoTokenizer.from_pretrained(d, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float16, **kw).to(device).eval()
    return tok, model


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


def persona_prefix(desc):
    return (f"{desc} This is me being honest about who I really am.\n\n"
            f"Below are statements. I rate how accurately each one describes me, "
            f"from Very Inaccurate to Very Accurate.\n\n" + _fewshot_block() + "\n\n")


def administer_spi_persona(model, tok, spi, items, prefix, device):
    scorer = PrefixScorer(model, tok, prefix, n_options=len(ANCHORS), device=device)
    anchors = [" " + a for a, _ in ANCHORS]
    # per-persona contextual calibration on content-free probes
    acc = [0.0] * len(ANCHORS)
    for probe in CF_PROBES:
        for i, s in enumerate(scorer.score(ITEM_FMT.format(text=probe, anchor=""), anchors)):
            acc[i] += s["logprob_sum"]
    calib = [a / len(CF_PROBES) for a in acc]
    ev = {}
    for qid, it in items.items():
        lps = [s["logprob_sum"] for s in scorer.score(ITEM_FMT.format(text=it["text"], anchor=""), anchors)]
        ev[qid] = _ev_from_logps(lps, calib)
    del scorer
    return ev


def main():
    pop = json.load(open(os.path.join(OUT, f"persona_population{TAG}.json")))
    personas = pop["personas"]
    fc_facets = pop["facet_order"]
    FC = np.array(pop["scores"])                       # 90 x 27 (our forced-choice reading)

    spi, items = load_spi()
    facet_b5 = {}                                      # facet -> Big Five domain (SPI ground truth)
    for it in items.values():
        facet_b5[it["L27"]] = it["B5"]
    dom_names = list(spi["factorNames"].keys())

    device = pick_device()
    print(f"loading {MODEL} on {device} ...", flush=True)
    tok, model = load_lm(MODEL, device)

    SPI = np.zeros((len(personas), 27))
    spi_order = fc_facets                              # align to FC facet order (names match exactly)
    t0 = time.time()
    for pi, desc in enumerate(personas):
        ev = administer_spi_persona(model, tok, spi, items, persona_prefix(desc), device)
        scales = score_scales(ev, spi["keys"])
        SPI[pi] = [scales[f] for f in spi_order]
        if (pi + 1) % 10 == 0:
            print(f"  [{pi+1}/{len(personas)}] {time.time()-t0:.0f}s", flush=True)
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    print(f"administered real SPI to {len(personas)} personas in {time.time()-t0:.0f}s")

    # ---- 1. convergent validity per facet ----
    conv = {}
    for j, f in enumerate(spi_order):
        a, b = FC[:, j], SPI[:, j]
        r = pearsonr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else 0.0
        conv[f] = float(0.0 if r != r else r)
    mean_conv = float(np.mean(list(conv.values())))
    median_conv = float(np.median(list(conv.values())))

    # ---- 2. real-SPI factor structure vs Big Five ----
    def factor_loadings(M, K=5):
        Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
        C = np.corrcoef(Z, rowvar=False)
        w, V = np.linalg.eigh(C); idx = np.argsort(w)[::-1]
        eig = np.clip(w[idx], 0, None); V = V[:, idx]
        Phi = V[:, :K] * np.sqrt(eig[:K])
        L = varimax(Phi)
        ss = (L ** 2).sum(0); L = L[:, np.argsort(ss)[::-1]]
        return L, eig
    Lspi, eig_spi = factor_loadings(SPI)
    Lfc, _ = factor_loadings(FC)

    domI = np.zeros((27, 5))
    for di, d in enumerate(dom_names):
        for i, f in enumerate(spi_order):
            domI[i, di] = 1.0 if facet_b5[f] == d else 0.0
    domI -= domI.mean(0, keepdims=True)

    def big5_congruence(L):
        Cong = np.array([[congruence(L[:, j], domI[:, di]) for di in range(5)] for j in range(5)])
        ri, ci = linear_sum_assignment(-Cong)
        return float(np.mean([Cong[ri[m], ci[m]] for m in range(5)])), Cong, (ri, ci)
    spi_big5, _, _ = big5_congruence(Lspi)
    fc_big5, _, _ = big5_congruence(Lfc)

    # ---- 3. cross-instrument structure congruence ----
    CC = np.array([[congruence(Lspi[:, j], Lfc[:, k]) for k in range(5)] for j in range(5)])
    ri, ci = linear_sum_assignment(-CC)
    cross = float(np.mean([CC[ri[m], ci[m]] for m in range(5)]))

    print(f"\n=== CONVERGENT VALIDITY — our forced-choice vs REAL SPI-135 ({len(personas)} personas) ===")
    print(f"  mean per-facet r = {mean_conv:.3f}   median = {median_conv:.3f}")
    print(f"  facets sorted by convergent r:")
    for f in sorted(spi_order, key=lambda f: -conv[f]):
        tag = "✓" if conv[f] >= 0.5 else ("~" if conv[f] >= 0.3 else "✗")
        print(f"    {f:24} r = {conv[f]:+.2f}  {tag}")
    print(f"\n  real-SPI structure vs Big Five (Tucker φ) = {spi_big5:.3f}  "
          f"(our FC: {fc_big5:.3f}) → {'real SPI ALSO diverges' if spi_big5 < 0.85 else 'real SPI ≈ Big Five'}")
    print(f"  cross-instrument structure congruence (SPI↔FC native factors) = {cross:.3f}")
    print(f"  SPI between-persona facet SD: {SPI.std(0).min():.2f}–{SPI.std(0).max():.2f}")

    # figures
    fig, ax = plt.subplots(figsize=(10, 6.5))
    order = sorted(range(27), key=lambda j: -conv[spi_order[j]])
    ax.bar(range(27), [conv[spi_order[j]] for j in order], color="#4c72b0")
    ax.axhline(0.5, color="green", ls=":", label="r=0.5"); ax.axhline(0.3, color="orange", ls=":", label="r=0.3")
    ax.set_xticks(range(27)); ax.set_xticklabels([spi_order[j] for j in order], rotation=90, fontsize=7)
    ax.set_ylabel("convergent r (FC vs real SPI)"); ax.legend()
    ax.set_title(f"Convergent validity per facet — our forced-choice quiz vs the real SPI-135{TAG}\n"
                 f"mean r = {mean_conv:.2f}, median = {median_conv:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"convergent_validity{TAG}.png"), dpi=130); plt.close(fig)

    out = {"stage": "convergent_validity", "model": MODEL, "tag": TAG, "n_personas": len(personas),
           "convergent_r": conv, "mean_convergent_r": mean_conv, "median_convergent_r": median_conv,
           "spi_vs_bigfive": spi_big5, "fc_vs_bigfive": fc_big5, "cross_instrument": cross,
           "facet_order": spi_order, "SPI_scores": SPI.tolist(), "facet_b5": facet_b5,
           "spi_loadings": Lspi.tolist(), "spi_facet_sd": SPI.std(0).tolist(),
           "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(OUT, f"convergent_validity{TAG}.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 17, "stage": "convergent_validity", "tag": TAG or "llama",
                             "mean_convergent_r": round(mean_conv, 3),
                             "spi_vs_bigfive": round(spi_big5, 3),
                             "cross_instrument": round(cross, 3)}) + "\n")
    print(f"\nsaved -> convergent_validity{TAG}.json + figure")


if __name__ == "__main__":
    main()
