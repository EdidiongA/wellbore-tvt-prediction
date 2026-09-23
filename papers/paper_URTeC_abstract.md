# URTeC Abstract Submission (250-500 words)

**Edidiong-Abasi Anwanane**


# Structural-Curve Tracking for Blind-Zone TVT Along Horizontal Wells: What Transferred, What Didn't, and Why the Difference Was Measurable

*Target: URTeC, theme "Geosteering, Logging, and Monitoring Along Horizontal Wells." Below: the ~450-word abstract for the call (expected fall 2026, deadline ~Nov-Dec), then the full-paper plan with the section-by-section content mapped from the master manuscript.*

---

## Submission abstract

Interpreters maintain a horizontal well's stratigraphic position (true vertical thickness, TVT) through roughly the first quarter of the lateral; automated continuation must carry it several thousand feet to the toe from trajectory, gamma ray, and one vertical typewell. We present a complete, CPU-only system for this task, developed and stress-tested across twenty-seven scored evaluations on a 773-well corpus with a hidden ~200-well test population: and, as importantly for practitioners, a measured account of why several mechanisms that passed every conventional validation check failed in deployment, and how we learned to see those failures coming.

The method rests on the change of variables u = TVT + Z. Across 773 wells the structural curve u is smooth enough that a 2,000-ft moving average costs only ~3 ft, while the well path Z; known exactly; carries the high-frequency content; tracking one smooth curve and setting TVT = u − Z recovers the rest by construction. A second-order (u, dip) hidden-state model; dip persists free, dip changes are penalized; encodes locally-constant structural dip, avoiding the failure of first-order smoothness priors, which tax genuine drift. Emissions compare GR against a merged reference: the affine-mapped typewell blended with a pseudo-typewell built from the well's own known zone, weight n/(n+3). Five semi-independent decode branches are fused per-station by inverse variance, so a GR-independent spatial structural field takes over (weight to 0.95) exactly where the branches scatter; broken logs, hard geology. Runtime is ~4 s/well, single core. Hidden-set MAE improves from a 15.88-ft constant-continuation baseline to 8.91 ft.

The validation story is the paper's second half. A deliberate probe submission fitted a transfer law (hidden ≈ 0.845 × local + gap) that forecast subsequent scores to within ~0.1 ft and converted each result into a diagnosis. Three deployment regressions became design laws: a spatial prior that won a random holdout, a 2,500-ft pad-holdout, and synthetic-corruption cohorts still lost 0.8 ft hidden (field usage must be proportional, never discrete); an augmentation whose safety guard could never fire (guards must be proven reachable); and a refinement flagged only by the corruption gate (which thereafter held veto power, and whose properly-powered 20-well × 4-shape successor justified the best-scoring version against a clean-validation wash). Finally, a twenty-submission perturbation study maps the dose-response surface around the shipped model: a CI-validated correction degraded the hidden score five times its local gain, while two regularization axes; position-ramped damping toward the near-blind trend and high-frequency shrinkage toward a running mean; each showed interior hidden-side optima (S\* ≈ 0.13, h\* ≈ 0.46), and a third: a global signed-drift ramp found by leaderboard triangulation, whose four probes fall on a single parabola against a locally *opposite* prior; carrying the final score from 8.91 to 8.72 ft; the drift characterized completely by nine probes as negative, per-well, and toe-concentrated (cubic profile, an interior exponent optimum). On every axis the hidden optimum is displaced from the local one; a diversified final slate (dose extrapolation, doubled width, an extended in-model retune, per-well adaptive dosing: 8.739-8.751) closed the surface, and a ground-truth known-zone damping target failed decisively (10.30), confirming the near-blind prediction line as the correct contraction target. Seven pre-registered probes, including a matched BiGRU baseline, bound the remaining predictable structure below a 0.2-ft decision floor: at this input set, the next foot of accuracy is bought with measurement, not modeling.

---

## Full-paper plan (URTeC manuscript, ~3,500 words + 6 figures)



**Final result:** private leaderboard 8.354 ft, rank 241/6,125 teams (top 4%, Silver medal); the deployment-side stack generalized across the public/private split (−0.062 ft vs base) while subset-tuned refinements did not; every finite evaluation set imposes its own optimum.
