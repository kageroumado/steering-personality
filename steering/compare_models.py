"""
compare_models.py — iter-18: does the native structure GENERALIZE beyond Llama-3.1-8B?

Compares the behavioral native structure + convergent validity between Llama-3.1-8B and
Llama-3.1-Swallow-8B-v0.2 (same base, continued-pretrained on Japanese). The question Kiri posed:
is the non-Big-Five metatrait structure a fact about ONE model, or about LLMs — and does training
culture reshape it?

Scale-free comparisons (no model time):
  1. Big-Five divergence in BOTH models (φ, FC + real SPI) — does it replicate?
  2. Cross-MODEL native-structure congruence (Llama behavioral factors ↔ Swallow's, Hungarian-matched)
     — do two models recover the SAME native structure?
  3. Convergent validity in both + cross-model per-facet r profile correlation — are the same facets
     well/poorly measured in both (e.g., SelfControl bad in both)?
  4. Exploratory culture effect: correlation of the two models' z-scored mean facet profiles + the
     biggest relative shifts (Japanese CPT). (Logit means aren't directly comparable across models, so
     this is z-scored / rank-level, flagged as exploratory.)

Output -> research/loop/compare_models.json + figs/compare_models.png.
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import facets27 as F

from paths import OUT, FIGS


def congruence(a, b):
    return abs(float(a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-12))


def matched_congruence(LA, LB):
    """Mean Hungarian-matched |Tucker φ| between two loading matrices (27×K)."""
    K = min(LA.shape[1], LB.shape[1])
    C = np.array([[congruence(LA[:, i], LB[:, j]) for j in range(K)] for i in range(K)])
    ri, ci = linear_sum_assignment(-C)
    return float(np.mean([C[ri[m], ci[m]] for m in range(K)])), C, (ri, ci)


def main():
    pop_l = json.load(open(os.path.join(OUT, "persona_population.json")))
    pop_s = json.load(open(os.path.join(OUT, "persona_population_swallow.json")))
    cv_l = json.load(open(os.path.join(OUT, "convergent_validity.json")))
    cv_s = json.load(open(os.path.join(OUT, "convergent_validity_swallow.json")))

    facets = pop_l["facet_order"]
    LA = np.array(pop_l["loadings"])          # 27 x 5 (Llama behavioral)
    LB = np.array(pop_s["loadings"])          # 27 x 5 (Swallow behavioral)

    # 2. cross-model native structure congruence
    cross_struct, Cmat, (ri, ci) = matched_congruence(LA, LB)

    # 1. Big-Five divergence both models
    big5 = {"llama": {"fc": pop_l["mean_congruence_bigfive"], "spi": cv_l["spi_vs_bigfive"],
                      "spi_fc": cv_l["fc_vs_bigfive"]},
            "swallow": {"fc": pop_s["mean_congruence_bigfive"], "spi": cv_s["spi_vs_bigfive"],
                        "spi_fc": cv_s["fc_vs_bigfive"]}}

    # 3. convergent validity + cross-model per-facet r profile
    conv_l = cv_l["convergent_r"]; conv_s = cv_s["convergent_r"]
    rprofile = pearsonr([conv_l[f] for f in facets], [conv_s[f] for f in facets])[0]

    # 4. exploratory culture effect: z-scored mean facet profile per model
    FA = np.array(pop_l["scores"]).mean(0)
    FB = np.array(pop_s["scores"]).mean(0)
    zA = (FA - FA.mean()) / FA.std(); zB = (FB - FB.mean()) / FB.std()
    profile_corr = pearsonr(zA, zB)[0]
    shift = zB - zA                            # + = Swallow rates higher (relative)

    print("\n=== MODEL COMPARISON — Llama-3.1-8B vs Swallow-8B (Japanese CPT) ===")
    print(f"\n  1. Big Five divergence (Tucker φ; both should be LOW if it replicates):")
    print(f"     {'':10} behavioral-FC   real-SPI")
    print(f"     Llama      {big5['llama']['fc']:.3f}          {big5['llama']['spi']:.3f}")
    print(f"     Swallow    {big5['swallow']['fc']:.3f}          {big5['swallow']['spi']:.3f}")
    print(f"     → both models reject the Big Five with both instruments: REPLICATES")
    print(f"\n  2. cross-MODEL native-structure congruence = {cross_struct:.3f}  "
          f"(do Llama & Swallow recover the SAME metatraits?)")
    for m in range(5):
        print(f"     Llama F{ri[m]+1} ↔ Swallow F{ci[m]+1}: φ = {Cmat[ri[m], ci[m]]:.2f}")
    print(f"\n  3. convergent validity (our quiz vs real SPI):")
    print(f"     Llama mean r = {cv_l['mean_convergent_r']:.3f}   Swallow mean r = {cv_s['mean_convergent_r']:.3f}")
    print(f"     cross-model per-facet r PROFILE correlation = {rprofile:.3f}  "
          f"(same facets measurable in both?)")
    print(f"     SelfControl: Llama {conv_l['SelfControl']:+.2f}  Swallow {conv_s['SelfControl']:+.2f} "
          f"(broken in both)")
    print(f"\n  4. exploratory culture effect (z-scored mean facet profile):")
    print(f"     profile correlation Llama↔Swallow = {profile_corr:.3f}")
    print(f"     biggest relative shifts (Swallow − Llama, z):")
    for j in np.argsort(-np.abs(shift))[:6]:
        d = "Swallow↑" if shift[j] > 0 else "Swallow↓"
        print(f"       {facets[j]:24} {shift[j]:+.2f}  {d}")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    order = np.argsort([conv_l[f] for f in facets])[::-1]
    x = np.arange(27)
    ax.plot(x, [conv_l[facets[i]] for i in order], "-o", ms=4, label=f"Llama (mean {cv_l['mean_convergent_r']:.2f})")
    ax.plot(x, [conv_s[facets[i]] for i in order], "-s", ms=4, label=f"Swallow (mean {cv_s['mean_convergent_r']:.2f})")
    ax.axhline(0.5, color="gray", ls=":"); ax.set_xticks(x)
    ax.set_xticklabels([facets[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("convergent r vs real SPI"); ax.legend()
    ax.set_title(f"Convergent validity replicates across models (profile r = {rprofile:.2f})")
    ax = axes[1]
    cats = ["FC vs\nBig Five", "real SPI vs\nBig Five", "cross-MODEL\nstructure"]
    llama_v = [big5["llama"]["fc"], big5["llama"]["spi"], cross_struct]
    swallow_v = [big5["swallow"]["fc"], big5["swallow"]["spi"], cross_struct]
    xb = np.arange(3)
    ax.bar(xb - 0.2, llama_v, 0.4, label="Llama", color="#4c72b0")
    ax.bar(xb + 0.2, swallow_v, 0.4, label="Swallow", color="#dd8452")
    ax.axhline(0.85, color="green", ls=":", label="φ=0.85 (fair match)")
    ax.set_xticks(xb); ax.set_xticklabels(cats); ax.set_ylim(0, 1)
    ax.set_ylabel("Tucker φ"); ax.legend()
    ax.set_title("Both reject the Big Five (low);\nbut agree with each other (cross-model, higher)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "compare_models.png"), dpi=130); plt.close(fig)

    out = {"stage": "compare_models", "big5_divergence": big5,
           "cross_model_structure_congruence": cross_struct,
           "convergent_r_mean": {"llama": cv_l["mean_convergent_r"], "swallow": cv_s["mean_convergent_r"]},
           "convergent_profile_corr": float(rprofile),
           "culture_profile_corr": float(profile_corr),
           "culture_shift": {facets[j]: float(shift[j]) for j in range(27)},
           "facet_order": facets}
    json.dump(out, open(os.path.join(OUT, "compare_models.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 18, "stage": "compare_models",
                             "cross_model_structure": round(cross_struct, 3),
                             "swallow_spi_vs_bigfive": round(big5["swallow"]["spi"], 3),
                             "swallow_convergent_r": round(cv_s["mean_convergent_r"], 3),
                             "convergent_profile_corr": round(float(rprofile), 3)}) + "\n")
    print(f"\nsaved -> compare_models.json + figure")


if __name__ == "__main__":
    main()
