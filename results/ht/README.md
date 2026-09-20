# Genuine HT tree selection: dense eight-dimensional verification

14/14 non-tie cases certified among all **315 balanced binary trees**; rank ≤ 2 on every nonroot edge, including singleton leaves. This is a finite balanced-tree family, not all possible binary topologies.

**The strengthened exhaustive-rounding heuristic is faster on every non-tie case in this run.** The demonstrated benefit of this general-HT extension is a global tree-family certificate with few feasible HT fits, not a wall-time improvement over that heuristic. Comparing only against naive repeated rounding would hide this result.

Joint-dual schedule: `early`. In 14/14 non-tie cases the selector took zero Richardson steps. An early schedule tries joint cut bounds immediately when a few competitors remain, before fitting those competitors or refining a solution proxy.

The source is constructed from random nested HT cores; no target solution is manufactured. The band cases mix different low- and high-frequency structures. Seeds 101, 103 and 107 are development cases; 131 and 137 were first used with the late schedule and were subsequently observed during early-schedule development. Seeds 151, 157 and 163, when present, were fixed before testing the early schedule and use the same axis-scrambling rule. All algorithms receive the same tensor and diagonal operator factors.

The new method computes feasible energy gains, shared cut bounds, and optionally a small number of preconditioned Richardson steps. The **proxy steps** column makes those partial iterations explicit. It does not form the full solution. The baseline forms the exact spectral solution, performs nested HT rounding for all 315 trees, ranks them by optimally rescaled actual energy gain, then applies two energy-ALS sweeps only to the best candidate. The strengthened version shares identical leaf/subtree SVDs across all trees, with fresh caches for each call; it is the primary timing baseline whenever present. This baseline is a heuristic, not a global energy-optimal tree certificate.

| Case | Certifies | HT fits | Proxy steps | New, s | Strong baseline, s | Baseline/new | Same tree |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_trees_tie | no | 1 | 0 | 0.0635 | 0.0290 | 0.46× | yes |
| band_seed101 | yes | 1 | 0 | 0.8370 | 0.5398 | 0.64× | yes |
| band_seed103 | yes | 1 | 0 | 0.7299 | 0.5698 | 0.78× | yes |
| band_seed107 | yes | 1 | 0 | 0.5856 | 0.5253 | 0.90× | yes |
| band_seed131 | yes | 1 | 0 | 0.5938 | 0.5382 | 0.91× | yes |
| band_seed137 | yes | 1 | 0 | 0.9759 | 0.5445 | 0.56× | yes |
| band_seed151 | yes | 1 | 0 | 0.6202 | 0.5534 | 0.89× | yes |
| band_seed157 | yes | 1 | 0 | 0.7100 | 0.6564 | 0.92× | yes |
| band_seed163 | yes | 1 | 0 | 0.7894 | 0.6092 | 0.77× | yes |
| n2_seed31 | yes | 1 | 0 | 0.0744 | 0.0399 | 0.54× | yes |
| n2_seed47 | yes | 1 | 0 | 0.0734 | 0.0390 | 0.53× | yes |
| n2_seed59 | yes | 1 | 0 | 0.0740 | 0.0397 | 0.54× | yes |
| n3_seed31 | yes | 1 | 0 | 0.3052 | 0.0990 | 0.32× | yes |
| n3_seed47 | yes | 1 | 0 | 0.2733 | 0.1015 | 0.37× | yes |
| n3_seed59 | yes | 1 | 0 | 0.2749 | 0.1018 | 0.37× | yes |

Times are medians of the number of repeats recorded per case in `summary.json`. Method order is shuffled deterministically within each repeat of the original comparison. For the original late-schedule data, the shared-subtree baseline was added in a separate pass. For the early-schedule data, both reported methods are shuffled within each repeat. Input generation is excluded; operator setup, cut SVDs, all tree fits and decision checks are included. Candidate-tree enumeration is shared and warmed before timings. Post-timing independent edge-rank diagnostics on the baseline output are excluded. NumPy runs with one BLAS thread. Raw per-repeat bounds, gains, decisions, source hashes and environment are in `all_results.json.gz`.

The naive repeated-rounding validation is also retained in raw data (outside reported timing when `--baseline shared` is used); its times should not be used to claim an advantage over the stronger shared-subtree baseline. All 315 individual rounded gains from the two implementations are cross-checked for agreement.

The exact spectral solve itself is inexpensive for this separable model. Any speedup here comes from reducing tree search and approximation work; it does not show an advantage over the spectral PDE solver alone. Dense storage is used and no continuum-discretization or large-dimension scalability guarantee is claimed.

A certificate compares a feasible lower gain for the winner with valid upper gains for every competitor. Local energy ALS need not find the exact optimum for certification to be valid. Conversely, failure to certify is retained as an abstention. Bounds use floating-point arithmetic with tolerances, not interval arithmetic. The equal-gain control must abstain.

Reproduce from the project root:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_ht_v3.py --repeats 3 --output results/generated/ht --joint-schedule early --groups small band band_fresh control --baseline shared
```

To regenerate the report and plots from raw files: `PYTHONPATH=src python scripts/benchmark_ht_v3.py --report-only --output results/ht`.
