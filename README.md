<div align="center">

# Blind-Zone TVT Prediction in Horizontal Wells

**A structural-coordinate tracker with deployment-side dose calibration — and a three-way study of why leaderboards disagree**

[![Kaggle](https://img.shields.io/badge/Kaggle-Silver%20Medal-C0C0C0?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
![Rank](https://img.shields.io/badge/rank-241%20%2F%206125%20·%20top%204%25-2E86AB)
![Shakeup](https://img.shields.io/badge/private%20reveal-%2B3039%20places-E4572E)
![Compute](https://img.shields.io/badge/compute-CPU%20only%20·%204s%2Fwell-0F7B6C)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**Edidiong Anwanane** · [Kaggle competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) · [Preprint (PDF)](papers/rogii_preprint_Anwanane.pdf) · arXiv link forthcoming

</div>

---

Geosteering — keeping a horizontal well inside its target zone — depends on knowing the well's geological position where direct interpretation runs out: the **blind zone**, roughly 75% of every lateral. Horizontal wells produce ~94% of U.S. crude oil; this is the informational bottleneck under all of it.

This repository contains a complete solution built in **four weeks of participation**: top 4% of 6,125 teams, rising **3,039 places** at the private-leaderboard reveal — a collapse of the overfit public tier that this work's own analysis predicted in writing beforehand (preprint, §10).

<div align="center">

<img src="figures/F6_ladder.png" width="920" alt="Campaign ladder: 15.88 to 8.72 ft, every step a named mechanism"/>

</div>

## Results

| | Public LB | Private LB |
|---|:---:|:---:|
| Constant-continuation baseline | 15.88 | — |
| Base model (v22: tracker + fusion + field) | 8.913 | 8.416 |
| **Final (v55: + calibrated regularization stack)** | **8.720** | **8.354** |

A **45% error reduction** with every step attributed to a single named mechanism — the full ledger, including every falsified idea, is in [`docs/campaign_log.md`](docs/campaign_log.md).

## The method

Reparameterize the target into a structural coordinate **u = TVT + Z**: thickness is dominated by wellbore geometry, but *u* is smooth geology — a tracking problem, not a regression problem.

<div align="center">
<img src="figures/F2_lattice.png" width="880" alt="Second-order (u, dip) trellis with banded beam search"/>
</div>

A second-order **(u, dip) hidden-state tracker** (banded trellis, coarse-to-fine to 0.25 ft, GR-correlation emissions against a merged typewell reference) is fused per-station with drift-cancelling, typewell-anchored, and spatial-field branches by **inverse-variance weighting of their live disagreement**. One CPU core, ~4 seconds per well, no offset logs.

## Three findings worth stealing

**1 — Validation is an instrument you can point at deployment.** Deliberately spent probe submissions fit a *local-to-leaderboard transfer law*; departures from the law diagnose mechanisms. The correction that broke it — best-ever local score, CI-validated — was the worst deployment score of its era:

<div align="center">
<img src="figures/F3_transfer.png" width="760" alt="Transfer law and its inversion"/>
</div>

**2 — Deployment-side optima are displaced from local ones.** Four regularization axes (damping strength, high-frequency shrinkage, drift dose, drift profile) were dose-response mapped directly against the evaluation set. Each has a clean interior optimum — at doses local validation scores as *harmful*:

<div align="center">
<img src="figures/F5b_displacement.png" width="760" alt="The displacement law"/>
<img src="figures/F5d_drift.png" width="760" alt="Drift axis: four probes, one parabola"/>
</div>

**3 — Every finite evaluation set imposes its own optimum.** The same submissions were scored three ways — local CV, public LB, private LB — and the orderings disagreed *twice over*. Only structure-grounded mechanisms transferred across all three splits:

<div align="center">
<img src="figures/F7_public_private.png" width="760" alt="Public vs private verdicts disagree"/>
</div>

## Repository layout

| Path | Contents |
|---|---|
| [`model/`](model) | Tracker notebooks (final base: `rogii_tvt_tracker_v22.ipynb`) + leave-one-well-out instrumentation |
| [`neural/`](neural) | The sequence-model line: 1D U-Net → BiGRU (five-fold OOF) + tracker↔BiGRU join analysis |
| [`postprocessing/`](postprocessing) | The submission campaign v25–v58 — one mechanism per file |
| [`diagnostics/`](diagnostics) | Local instruments (L1–L8), drift/anchor/slope probes, exploitability tests |
| [`papers/`](papers) | Preprint PDF + manuscript sources (journal, URTeC, EAGE, IMAGE) |
| [`docs/`](docs) | Full campaign log — every submission, mechanism, public & private score |

## Data

Competition data is **not redistributed** here, per the competition's data-use rules. Join the [competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) and attach its dataset on Kaggle; notebooks are written for that environment (CPU, internet off).

## Citing

```bibtex
@misc{anwanane2026tvt,
  author = {Anwanane, Edidiong},
  title  = {Blind-Zone TVT Prediction in Horizontal Wells: An Information-Ceiling
            Study with Deployment-Side Dose Calibration},
  year   = {2026},
  note   = {Preprint. Code: https://github.com/EdidiongA/wellbore-tvt-prediction}
}
```

## License

Code: [MIT](LICENSE). Papers and figures © 2026 Edidiong Anwanane; preprint distributed under CC BY 4.0 via arXiv.
