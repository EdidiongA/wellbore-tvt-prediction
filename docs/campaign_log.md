# Campaign Log — every submission, one mechanism per step

Public LB unless noted. Private scores listed where evaluated at the reveal.

| Ver | Mechanism | Public | Private |
|---|---|---|---|
| — | constant continuation (distribution probe) | 15.88 | — |
| v2 | (u, dip) trellis tracker | 13.196 | — |
| v7 | merged reference + EM + drift-cancel branch | 11.630 | — |
| v8 | typewell-only branch | 11.197 | — |
| v9–v13 | known-dip handling, weight tuning, spatial prior trials | 11.044–10.577 | — |
| v14 | proportional spatial field blend | 9.624 | — |
| v15 | denser sampling + MD-shaped confidence | 9.382 | — |
| v16/17 | discrete field prior (falsified) | 10.200 | — |
| v18 | field length-scale retune | 10.200 | 9.070 |
| v19 | inverse-variance branch fusion | 8.926 | — |
| v20 | cohort augmentation w/ unreachable guard (falsified) | 10.625 | — |
| v21 | fine grid 0.25 ft | 9.019 | — |
| v22 | corrupted-suite retune — **sealed base** | 8.913 | 8.416 |
| v23 | retune variant | 8.934 | 8.320 |
| v24 | ivar boost | 8.914 | — |
| v25 | CI-validated typewell slope correction | 9.302 | 8.291 |
| v26 | toe damping S=0.35 | 9.113 | — |
| v27 | average with v19 | 8.912 | — |
| v28 | line-damping S=0.15 | 8.818 | — |
| v29 | v27 + damper (additivity control) | 8.823 | — |
| v30 | toe-concentrated damping | 8.889 | 8.345 |
| v31 | structural-only (u-space) damping | 8.936 | — |
| v33–v35 | flat HF shrinkage (dose/window scan) | 8.893/8.883/8.871 | — |
| v39 | damper + HF h=0.25 W601 | 8.785 | — |
| v40 | HF dose h=0.40 | 8.772 | — |
| v41 | h=0.60 W901 | 8.777 | — |
| v49 | drift ramp δ=+1.5 (triangulation probe) | 8.899 | — |
| v49b | drift ramp δ=−1.5 | 8.730 | — |
| v51 | δ=−3.0 (deep-tail probe) | 8.773 | — |
| v52 | per-foot drift allocation | 8.741 | — |
| v53 | known-zone trend as damping target | 10.297 | — |
| v48b | max width W1201 on full stack | 8.739 | — |
| v43b | in-model retune ×2.5 on full stack | 8.746 | — |
| v47b | dose extrapolation on full stack | 8.747 | 8.350 |
| v42b | typewell-adaptive dosing on full stack | 8.751 | — |
| v46b | S=0.25 under full stack (interaction probe) | 8.835 | — |
| v54 | quadratic drift profile (projection-matched) | 8.722 | 8.360 |
| **v55** | **cubic drift profile, toe −2.4 — FINAL** | **8.720** | **8.354** |
| v56 | quartic profile | 8.722 | — |
| v57 | cubic transverse dose −2.7 | 8.721 | 8.363 |
| v58 | capped open-loop steering (last gamble) | 8.861 | — |

Final standing: **241 / 6,125 (top 4%, Silver)**, +3,039 places at the reveal.
