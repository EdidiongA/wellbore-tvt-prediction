
## Figures

![**Figure 1.** Decomposition of the target. TVT is dominated by wellbore geometry; the structural coordinate u = TVT + Z is smooth and low-bandwidth (2,000-ft moving-average residual ≈ 3 ft), converting the problem to tracking a slowly varying state. *Synthetic example well shown for illustration; the decomposition statistics in Section 2 are measured on the full training corpus.*](figs/F1_decomposition.png)

![**Figure 2.** The tracking architecture: a second-order trellis over (structural position u, dip) in 30-ft blocks, banded beam search with coarse-to-fine refinement to 0.25 ft, emissions from windowed GR correlation against the merged reference. Max-product decoding with a forward-backward pass for confidence.](figs/F2_lattice.png)

![**Figure 3.** The local-to-hidden transfer law. Stable-cluster submissions fit hidden = 0.845·local + gap; the CI-validated typewell correction (v25, red) achieved the best local score of its era and the worst hidden score; the first measured inversion.](figs/F3_transfer.png)

![**Figure 4.** Left: out-of-fold error along the blind zone (measured endpoints 3.6 → 33 ft). Right: per-well error slope is large for an oracle but essentially unpredictable from any inference-time observable (0.801 vs 0.095); the central exploitability result.](figs/F4_ramp.png)

![**Figure 5a.** Hidden dose-response on the structural damping axis: three submissions on a parabola, interior optimum S* = 0.128.](figs/F5a_S_hidden.png)

![**Figure 5b.** The displacement law: the same damping operation dosed locally and on the hidden set. Optima at 0.05 vs 0.13; depths −0.018 vs −0.098 ft. Local validation prices the hidden optimum as strictly harmful.](figs/F5b_displacement.png)

![**Figure 5c.** Second axis: high-frequency shrinkage toward a 601-station running mean under the damper. Interior hidden optimum h* ≈ 0.46 where the local curve is already declining; the displacement law on an independent axis.](figs/F5c_h_axis.png)

![**Figure 5d.** Third axis: global signed-drift ramp. Four probes on a single parabola (residuals < 0.001), vertex δ* = −1.49 ft, against a training prior implying the opposite sign; the sharpest measured inversion.](figs/F5d_drift.png)

![**Figure 5e.** Fourth axis: drift profile exponent at projection-matched doses, with a transverse dose probe at the optimum. Interior optimum at the cubic (toe −2.4).](figs/F5e_exponent.png)

![**Figure 6.** The campaign ladder: 15.88 → 8.720 ft (45%), every step a single attributed mechanism, with the endgame closure cluster in which eight diversified arms confirm the completed surface.](figs/F6_ladder.png)

## Tables

**Table 1. Campaign ladder (principal steps).** Hidden-set (public leaderboard) MAE; one mechanism per step.

| Version | Mechanism added | Hidden MAE (ft) |
|---|---|---|
|; | constant continuation (distribution probe) | 15.88 |
| v2 | (u, dip) trellis tracker | 13.196 |
| v7 | merged reference + EM + drift-cancel branch | 11.630 |
| v8 | typewell-only branch | 11.197 |
| v14 | proportional spatial field blend | 9.624 |
| v15 | denser sampling + MD-shaped confidence | 9.382 |
| v18 | field length-scale retune | 10.200 |
| v19 | inverse-variance branch fusion | 8.926 |
| v22 | corrupted-suite weight retune (final base) | 8.913 |
| v28 | line-damping, S = 0.15 (Section 9) | 8.818 |
| v39 | + HF shrinkage h = 0.25, W601 | 8.785 |
| v40 | HF dose to h = 0.40 | 8.772 |
| v49b | + signed-drift ramp δ = −1.5 | 8.730 |
| v54 | quadratic drift profile, projection-matched | 8.722 |
| **v55** | **cubic drift profile, toe −2.4 (final)** | **8.720** |

**Table 2. Exploitability probes.** Oracle value vs observable capture; all pre-registered.

| Probe | Oracle / local signal | Observable capture (hidden where shipped) |
|---|---|---|
| Routing tracker↔BiGRU (J3) | 0.86 ft | 0.000 |
| Blending, dose-swept (J2) | +0.079 ft local | under 0.20 ft floor; not shipped |
| Per-station residual GBT (V2) | signals ρ ≤ 0.60 with \|error\| | 0.0005 ft |
| Prefix self-validation (C1) |; | selection −0.249 ft (harmful) |
| Slope correction (D3) | oracle ρ = 0.801 | best observable ρ = 0.095 |
| Anchor refinement (D4) |; | headroom 0.010 ft (none) |
| Typewell-scan slope (D6 → v25) | +0.082 ft local, CI-positive | **−0.389 ft shipped** |

**Table 3. Measured local↔hidden inversions.**

| Axis | Local signal | Hidden result |
|---|---|---|
| Typewell slope correction | best-ever local (CI-validated) | worst-ever hidden (9.302) |
| Damping dose S | optimum 0.05, depth −0.018 | optimum 0.13, depth −0.098 |
| HF shrinkage h | optimum ≈ 0.2, declining by 0.46 | optimum 0.46 |
| Drift sign δ | prior implies +δ (median resid. −0.335) | δ* = −1.49 |

**Table 4. Falsified-alternatives ledger (condensed).** Each entry names its falsifier.

| Mechanism | Falsifier |
|---|---|
| First-order smoothness priors | corrupted suite (shape collapse) |
| Learned emissions (fusion / matched filter / GBT) | OOF, all three variants |
| Discrete field prior | hidden set (v16/17: +1.3) |
| Cohort augmentation w/ unreachable guard | hidden set (v20: +1.7) |
| Drift-arrest mechanism | pad-holdout gate |
| Per-well adaptive damping (α; typewell rank) | L7 transfer ρ = −0.068; v42b +0.021 |
| Per-foot drift allocation | v52 +0.011 |
| Known-zone trend as damping target | v53 +1.577 |
| Open-loop steering term (unbounded; capped) | L6 local 4×; v58 +0.141 hidden |
| In-model retune extension (2.5×) | v43b +0.016 |
| Structural-only (u-space) damping | v31 +0.023 |
| Averaging (two forms) | ≤0.005, twice |
