# Reading Personality Off the Steering Geometry of a Language Model

Code, model-derived data and a CPU reproduction check for recovering personality structure from Llama-3.1-8B’s activation-steering directions. The analysis covers 27 facets across 409 synthetic personas; whether that structure describes people is a separate validation question.

[Preprint](https://doi.org/10.5281/zenodo.21440937) · [Code archive](https://doi.org/10.5281/zenodo.21441021)

```sh
pip install numpy scipy
python3 reproduce.py
```

## Highlights

- Recompute eleven reported values from committed scores on a CPU in about a minute.
- Inspect the frozen 27-facet instrument and 409 synthetic personas.
- Compare recovered factors against structure-destroying and random-label nulls.

<details>
<summary>Results and reproduction reference</summary>

## Headline results (N = 409)

| Claim | Value |
|---|---|
| Recovered metatraits | 5 (Drive, Plasticity, Fluidity, Stability, Communion) — 3 strong |
| Internal reliability (split-half congruence) | **0.88** |
| Cross-instrument agreement (steering quiz vs SPI-135 Likert) | φ ≈ 0.75 |
| Cross-finetune / cross-language replication (base vs Swallow) | φ ≈ 0.90 |
| Congruence to the Big Five | φ ≈ 0.32, inside the pre-specified random-label band 0.26–0.33 → orthogonal |
| Permutation null (structure-destroying) | split-half 0.88 → 0.28, 14.6 SD below observed |

## Reproducing

The committed-score checks run on a CPU in about a minute.

`python3 reproduce.py --quick` uses 200 null draws instead of 2,000.

`reproduce.py` recomputes eleven reported values from committed persona scores using
the repository’s functions. It exits non-zero on a mismatch.

| What it checks | Paper |
|---|---|
| N personas, facets | 409, 27 |
| Split-half congruence | 0.883 |
| Congruence to the Big Five | 0.322 |
| Parallel-analysis factors | 5 |
| Column-permutation null: mean, sd, z | 0.277, 0.041, 14.6 |
| P(parallel analysis recovers >= 5 factors under the null) | 0.03 |
| Big-Five label null: mean, p | 0.258, 0.923 |

Expected output ends with `11/11 checks passed`. Reports of a mismatch are
welcome as an issue: paste the table.

Re-extracting the 27 directions
needs the Llama-3.1-8B weights and a GPU. The cross-instrument control
additionally needs the SPI-135 item bank, which is not redistributable here.
The cross-language replication needs Llama-3.1-Swallow-8B-v0.2. `reproduce.py` lists these additional requirements.

### Re-running the pipeline itself

```bash
pip install -r requirements.txt
export STEERING_MODEL="meta-llama/Meta-Llama-3.1-8B"   # or a local path to the weights
python steering/native_factors.py        # rebuild the native structure from the directions
python steering/permutation_null.py      # the structure-destroying null
python steering/convergent_validity.py   # cross-instrument + cross-language checks
```

Scripts read the base model from `$STEERING_MODEL` (default
`meta-llama/Meta-Llama-3.1-8B`); the cross-language control uses
`tokyotech-llm/Llama-3.1-Swallow-8B-v0.2`. Inputs and outputs resolve through
`steering/paths.py`: an environment variable if set, otherwise this checkout's
`results/`, `vectors/` and `instrument/`. Override any of them with
`STEERING_OUT`, `STEERING_VECTORS` or `STEERING_INSTRUMENT`.

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

Steering-vector extraction uses [`repeng`](https://github.com/vgel/repeng).

</details>

## Notes

- CPU checks need NumPy and SciPy; re-extracting directions needs model weights and a GPU.
- Cross-instrument analysis also needs the non-redistributable SPI-135 item bank. The repository contains synthetic model-derived data, with human validity still unestablished.
