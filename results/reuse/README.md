# Fixed-operator affine-forcing certificate reuse

Nine predeclared sequences, 480 parameter queries, six methods, three complete
timing repetitions. All setup, fitting, failed warm fits, and structural refreshes
are included. Input construction and post-timing validation are excluded. Each
entry is the median total seconds for that complete sequence, not per-query time.

| Grid/seed/path | Queries | D | D-R | S | S-R | H-all | H-warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8/41/local | 64 | 3.186 | 0.449 | 1.950 | 0.433 | 0.762 | 0.434 |
| 8/41/stable | 64 | 3.463 | 0.884 | 1.972 | 0.538 | 0.906 | 0.498 |
| 8/41/transition | 64 | 3.760 | 2.115 | 2.066 | 1.155 | 0.746 | 0.550 |
| 8/67/local | 64 | 3.145 | 0.442 | 1.953 | 0.435 | 0.733 | 0.426 |
| 8/67/stable | 64 | 3.390 | 1.760 | 2.091 | 0.547 | 0.780 | 0.607 |
| 8/67/transition | 64 | 3.892 | 1.883 | 2.073 | 1.208 | 0.865 | 0.587 |
| 12/89/local | 32 | 15.608 | 2.130 | 6.981 | 1.890 | 2.232 | 1.672 |
| 12/89/stable | 32 | 15.267 | 7.305 | 6.583 | 2.344 | 2.167 | 1.822 |
| 12/89/transition | 32 | 17.442 | 11.465 | 6.751 | 4.634 | 2.156 | 2.263 |

D: fresh operator/forcing certificate at every query with cached operator setup.
D-R: D with homogeneous transport and retained feasible-direction bounds.
S: three offline basis PDE solves, fresh solution-based family certificate.
S-R: S with exactly the same certificate-reuse policy as D-R.
H-all: offline basis solves, round all cuts, refine selected candidate.
H-warm: offline basis solves, reuse the previous tree and candidate; refresh every
16 queries or when the monitored squared-energy error increases by more than 10%.

D-R certifies 480/480 queries with 347 reuse hits;
S-R certifies 474/480 with 393 hits. D-R improves
on D by 1.52–7.33 times in total time. The individual case tables, including
slower comparisons and abstentions, are the primary evidence. These numbers do
not establish superiority over S-R or a general end-to-end PDE speedup.

Quality uses the actual returned feasible tensor and a separate residual-controlled
post-timing solution oracle. The saved field `relative_energy_error` is **squared**
relative energy error, ||u-x||_A^2/||u||_A^2. Equal iteration limits alone do not
establish equal quality; per-method maximum and mean errors are in summary.json.
Heuristic disagreement is counted only where a strict certificate is available.
All certified winners agree; every reported upper bound contains every independent
feasible gain recorded for that family. All three repetitions have consistent
pointwise decisions, reuse events, gains, and intervals within the recorded tolerance.

## Scope

Four-dimensional reaction-diffusion on a box with smooth, analytically generated
correlated source channels, three balanced matrix cuts, rank one and unrestricted
leaf ranks. These are not general all-edge HT reuse tests and not engineering data.
The operator remains fixed within each sequence. The theory applies to fixed
general HT cones, but the repeated-PDE experiment does not establish that scaling.
Ordinary floating-point guards are not interval arithmetic.

## Reproduction

From the project root (write new timing runs to a new output directory):

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_certificate_reuse.py --n 8 --seeds 41 67 --queries 64 --repeats 3 --groups local stable transition --output results/generated/reuse_n8
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/benchmark_certificate_reuse.py --n 12 --seeds 89 --queries 32 --repeats 3 --groups local stable transition --output results/generated/reuse_n12
python scripts/report_certificate_reuse.py --sources results/generated/reuse_n8/all_results.json results/generated/reuse_n12/all_results.json --output results/generated/reuse_report
```

With no arguments, the reporting command reads the bundled compressed runs.
Source hashes in each original config describe the measured revision; the summary
records hashes of the uncompressed data. Packaging changed paths and reporting,
not the numerical library or the recorded measurements. Seed zero was used for a small pilot;
the final policies were fixed before seeds 41, 67 and 89 were evaluated.
