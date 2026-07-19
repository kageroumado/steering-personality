# Reading Personality Off the Steering Geometry of a Language Model

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21441021.svg)](https://doi.org/10.5281/zenodo.21441021)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research code and data for the preprint **_Reading Personality Off the Steering Geometry of a
Language Model_** (Mado Kagerou, 2026).

We recover a personality trait structure from *inside* Llama-3.1-8B — not by asking the model to rate
itself (a readout the skeptical literature shows is dominated by response bias) but from its **steering
geometry**: the contrastive activation directions that *induce* each trait. Bottom-up factor analysis
over 27 such directions across 409 personas yields five metatraits, three of them strong, that replicate
across a second instrument and a cross-language model swap, yet align to the Big Five no better than a
random relabeling — and survive a permutation null that rules out the construction-artifact reading.

- **Paper / DOI:** [10.5281/zenodo.21440937](https://doi.org/10.5281/zenodo.21440937)
- **Code archive / DOI:** [10.5281/zenodo.21441021](https://doi.org/10.5281/zenodo.21441021)
- **Human criterion-validation:** the question of whether this model-internal structure organizes
  *people* is a separate, pre-registered study. Human responses are collected anonymously and are **not**
  in this repository.

## Headline results (N = 409)

| Claim | Value |
|---|---|
| Recovered metatraits | 5 (Drive, Plasticity, Fluidity, Stability, Communion) — 3 strong |
| Internal reliability (split-half congruence) | **0.88** |
| Cross-instrument agreement (steering quiz vs SPI-135 Likert) | φ ≈ 0.75 |
| Cross-finetune / cross-language replication (base vs Swallow) | φ ≈ 0.90 |
| Congruence to the Big Five | φ ≈ 0.32 (at the random-label floor → orthogonal) |
| Permutation null (structure-destroying) | split-half 0.88 → 0.28, 14.6 SD below observed |

## Repository layout

```
steering/      the pipeline: direction extraction, the bias-cancelling readout, factor analysis, controls
instrument/    the frozen 27-facet (D27) instrument — items, named loadings, manifold seeds
results/       the canonical per-experiment result JSON the paper's figures and tables are built from
vectors/       the 27 extracted contrastive steering directions (base Llama-3.1-8B), one .gguf per facet
vectors_instruct/  the same directions extracted from Llama-3.1-8B-Instruct (used in the base-vs-instruct control)
figures/       the six figures in the paper
```

Key scripts, mapped to the paper:

| Script | What it does |
|---|---|
| `extract27.py`, `build_steering.py`, `extract_vectors.py` | extract the 27 contrastive facet directions (→ `vectors/`) |
| `persona_population.py`, `personas.py` | the N=409 naturalistic persona set |
| `forced_choice.py`, `administer.py`, `fastscore.py` | the bias-cancelling pole-contrast readout Δ = lp(hi) − lp(lo) |
| `native_factors.py`, `formalize_native.py`, `stabilize_n400.py` | the bottom-up factor analysis (→ `results/native_model_n400.json`, Figures 1–2) |
| `permutation_null.py` | the column-permutation null (→ `results/permutation_null.json`, Figure 3, §4.4) |
| `convergent_validity.py` | cross-instrument and cross-language (Swallow) convergence (Figures 4–5) |
| `compare_models.py` | cross-model comparison |
| `manifold_geometry.py` | the 27-direction cosine geometry (Figure 6) |
| `coherence_ladder.py`, `l2_diversity_controlled.py`, `l3_position_controlled.py` | capability-ladder analyses (companion work) |

## Reproducing

Requires Python 3.10+ and a GPU only if you re-extract directions; the extracted `vectors/` are included,
so the factor analysis and controls run on CPU.

```bash
pip install -r requirements.txt
export STEERING_MODEL="meta-llama/Meta-Llama-3.1-8B"   # or a local path to the weights
python steering/native_factors.py       # rebuild the native structure from the steering directions
python steering/permutation_null.py      # the structure-destroying null
python steering/convergent_validity.py   # cross-instrument + cross-language checks
```

Scripts read the base model from `$STEERING_MODEL` (default `meta-llama/Meta-Llama-3.1-8B`). The
cross-language control uses `tokyotech-llm/Llama-3.1-Swallow-8B-v0.2`.

## Data & scope

- The persona texts and all model-derived scores here are synthetic (model-generated / hand-authored);
  there is **no human data** in this repository.
- `instrument/d27_model.json` carries the frozen named loadings used to attach metatrait labels; the
  paper's headline numbers are the N=409 fit in `results/native_model_n400.json`.

## Citation

```bibtex
@misc{kagerou2026steering,
  title  = {Reading Personality Off the Steering Geometry of a Language Model},
  author = {Kagerou, Mado},
  year   = {2026},
  note   = {Preprint},
  doi    = {10.5281/zenodo.21440937},
  url    = {https://doi.org/10.5281/zenodo.21440937}
}
```

To cite the code archive specifically, use its DOI: [10.5281/zenodo.21441021](https://doi.org/10.5281/zenodo.21441021).

## License

Code and data released under the MIT License (see `LICENSE`). Built on
[`repeng`](https://github.com/vgel/repeng) for steering-vector extraction.
