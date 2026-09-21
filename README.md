# Blind-Zone TVT Prediction in Horizontal Wells
### 🥈 Silver Medal — ROGII Wellbore Geology Prediction (Kaggle) — 241st of 6,125 teams (top 4%)

**Edidiong Anwanane** · [Kaggle competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) · Preprint: `papers/rogii_preprint_Anwanane.pdf` (arXiv link forthcoming)

A complete, CPU-only solution for predicting True Vertical Thickness (TVT) through the blind zone of horizontal wells — the core interpretive task of geosteering — built in **four weeks of participation**, finishing top 4% and **rising 3,039 places** at the private-leaderboard reveal as overfit public-tier solutions collapsed (a collapse this work's own analysis predicted in writing beforehand; see the preprint, Section 10).

![Campaign ladder](figures/F6_ladder.png)

## The method in one paragraph

The target is reparameterized into a structural coordinate **u = TVT + Z**, separating smooth geology from the exactly-known well path. A second-order **(u, dip) hidden-state tracker** (banded trellis, coarse-to-fine to 0.25 ft, GR-correlation emissions against a merged typewell reference) is fused with drift-cancelling, typewell-anchored, and spatial-field branches by **inverse-variance weighting of their live disagreement**. Deployment-side **self-limiting regularization** — damping toward the near-blind trend, high-frequency shrinkage, and a signed drift correction with a cubic along-well profile — was dosed directly against the evaluation set via a probe-submission methodology. ~4 seconds per well, one CPU core, no offset logs.

## What's actually interesting here

- **A validation methodology, not just a model.** Deliberate probe submissions fit a *local-to-leaderboard transfer law*; a synthetic-corruption suite earned veto power by predicting failures; seven pre-registered probes measured what the inputs can and cannot support.
- **Dose-response mapping on the evaluation set:** four axes (damping strength, shrinkage, drift dose, drift profile exponent) each solved to an interior optimum — optima that local validation priced as *harmful*.
- **A three-way split study** (local / public / private): the orderings disagree twice over. The portable law: *every finite evaluation set imposes its own optimum* — only structure-grounded mechanisms transferred across all three.

![Displacement law](figures/F5b_displacement.png)
![Public vs private](figures/F7_public_private.png)

## Repository layout

| Path | Contents |
|---|---|
| `model/` | The tracker notebooks (final base: `rogii_tvt_tracker_v22.ipynb`) + honest leave-one-well-out instrumentation (`model/oof/`) |
| `neural/` | The sequence-model line: 1D U-Net → BiGRU (five-fold OOF), and the tracker↔BiGRU join analysis |
| `postprocessing/` | The submission campaign v25–v58: every deployment-side stage, one mechanism per file |
| `diagnostics/` | Local instruments (L1–L8), drift/anchor/slope probes (D3/D4/D6), exploitability tests |
| `papers/` | Preprint PDF + manuscript sources (journal, URTeC, EAGE, IMAGE) |
| `figures/` | All figures, generated from the measured campaign record |
| `docs/` | Full campaign log (every submission, mechanism, public & private score) |

## Data

Competition data is **not redistributed** here, per the competition's data-use rules. To run the notebooks, join the [competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) and attach its dataset on Kaggle; the notebooks are written for that environment (CPU, internet off).

## Results

| | Public LB | Private LB |
|---|---|---|
| Constant-continuation baseline | 15.88 | — |
| Final base model (v22) | 8.913 | 8.416 |
| **Final submission (v55: full stack)** | **8.720** | **8.354** |

45% error reduction, every step attributed to a named mechanism — see `docs/campaign_log.md`.

## License

Code: MIT. Papers and figures: © 2026 Edidiong Anwanane (preprint distributed under CC BY 4.0 via arXiv).
