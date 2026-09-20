# Contributing

Install with `python -m pip install -e .`, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
python examples/quickstart.py
bash scripts/reproduce.sh smoke
```

GitHub Actions runs the unit suite and quickstart on Python 3.10 and 3.12.
Long benchmarks are separate. Reproduction commands and the three retained
studies are described in the [README](README.md).

For numerical changes, explain the assumptions and check small cases against
independent oracles. Preserve tie abstentions, feasible returned tensors, and
comparisons with strong baselines. Keep new measurements in `results/generated/`
until reviewed; do not silently overwrite the recorded data.

Report total setup and fitting costs, quality, failures, and abstentions alongside
speed. A tighter bound alone is not a runtime improvement. Do not conflate the
4D cut experiment with all-edge HT constraints or claim established novelty from
an internal check. See the [theory](docs/THEORY.md), [prior-art audit](docs/RELATED_WORK.md),
and [draft](paper/draft.pdf).

Bug reports should include the command, minimal input, expected and observed
behavior, and Python/NumPy/SciPy versions.
