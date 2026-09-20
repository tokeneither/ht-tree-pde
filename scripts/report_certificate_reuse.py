#!/usr/bin/env python3
"""Build an auditable report from the two recorded reuse datasets."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/reuse'
METHODS = ('direct_fresh', 'direct_reuse', 'offline_certificate_fresh',
           'offline_certificate_reuse', 'offline_round_all', 'offline_warm_tree')
LABELS = ('D', 'D-R', 'S', 'S-R', 'H-all', 'H-warm')
SOURCES = (ROOT / 'results/reuse/n8.json.gz', ROOT / 'results/reuse/n12.json.gz')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs=2, type=Path, default=SOURCES,
                        help='Recorded n8/n12 .json.gz or reproduced all_results.json files')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    hashes = {}
    for path in args.sources:
        relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        packed = path.read_bytes()
        raw = gzip.decompress(packed) if path.suffix == '.gz' else packed
        rows = json.loads(raw)
        config_path = (path.with_name(path.name.removesuffix('.json.gz') + '_config.json')
                       if path.suffix == '.gz' else path.parent / 'config.json')
        config = json.loads(config_path.read_text())
        assert len(rows) == len(config['seeds']) * len(config['groups']), relative
        assert config['repeats'] == 3 and config['homogeneous_scaling']
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        for row in rows:
            assert not row['validation']['violations']
            for method in METHODS:
                runs = row['runs'][method]
                assert len(runs) == 3
                assert abs(median(x['total_seconds'] for x in runs)
                           - row['summary'][method]['total_seconds']) < 1e-12
                assert all(x['offline_basis_solves'] == (0 if method.startswith('direct') else 3)
                           for x in runs)
                records = runs[0]['records']
                assert sum(r['certified'] for r in records) == row['summary'][method]['certified']
                assert sum(r['reuse_hit'] for r in records) == row['summary'][method]['reuse_hits']
                disagreements = sum(
                    q['certified_winner'] is not None
                    and r['selected_index'] != q['certified_winner']
                    for r,q in zip(records,row['validation']['queries']))
                assert disagreements == row['summary'][method]['certified_disagreements']
        cases.extend(rows)
    assert len(cases) == 9
    count = sum(len(c['parameters']) for c in cases)
    facts = dict(case_count=len(cases), distinct_parameter_queries=count, repetitions=3,
                 methods=len(METHODS), source_sha256=hashes, totals={}, cases=[])
    for method in METHODS:
        summaries = [c['summary'][method] for c in cases]
        facts['totals'][method] = dict(
            total_of_median_seconds=sum(s['total_seconds'] for s in summaries),
            certified=sum(s['certified'] for s in summaries),
            reuse_hits=sum(s['reuse_hits'] for s in summaries),
            disagreements_with_certified_winner=sum(s['certified_disagreements'] for s in summaries),
            max_relative_squared_energy_error=max(s['max_relative_energy_error'] for s in summaries),
            max_excess_over_best_observed_relative=max(s['max_excess_over_best_observed_relative']
                                                      for s in summaries))
    timing_rows, markdown_rows = [], []
    for c in cases:
        n, seed, group = c['metadata']['n'], c['metadata']['seed'], c['group']
        name = f'{n}/{seed}/{group}'
        s, queries = c['summary'], len(c['parameters'])
        timing_rows.append([name, queries] + [f"{s[m]['total_seconds']:.3f}" for m in METHODS])
        markdown_rows.append('| ' + ' | '.join(map(str,timing_rows[-1])) + ' |')
        facts['cases'].append(dict(id=c['id'], queries=queries, summary=s,
            direct_reuse_speedup=s['direct_fresh']['total_seconds']/s['direct_reuse']['total_seconds'],
            solve_reuse_over_direct_reuse=s['offline_certificate_reuse']['total_seconds']/s['direct_reuse']['total_seconds']))
    (OUT/'summary.json').write_text(json.dumps(facts,indent=2)+'\n')
    low = min(c['direct_reuse_speedup'] for c in facts['cases'])
    high = max(c['direct_reuse_speedup'] for c in facts['cases'])
    dr, sr = facts['totals']['direct_reuse'],facts['totals']['offline_certificate_reuse']
    report = f'''# Fixed-operator affine-forcing certificate reuse

Nine predeclared sequences, {count} parameter queries, six methods, three complete
timing repetitions. All setup, fitting, failed warm fits, and structural refreshes
are included. Input construction and post-timing validation are excluded. Each
entry is the median total seconds for that complete sequence, not per-query time.

| Grid/seed/path | Queries | D | D-R | S | S-R | H-all | H-warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
''' + '\n'.join(markdown_rows) + f'''

D: fresh operator/forcing certificate at every query with cached operator setup.
D-R: D with homogeneous transport and retained feasible-direction bounds.
S: three offline basis PDE solves, fresh solution-based family certificate.
S-R: S with exactly the same certificate-reuse policy as D-R.
H-all: offline basis solves, round all cuts, refine selected candidate.
H-warm: offline basis solves, reuse the previous tree and candidate; refresh every
16 queries or when the monitored squared-energy error increases by more than 10%.

D-R certifies {dr['certified']}/{count} queries with {dr['reuse_hits']} reuse hits;
S-R certifies {sr['certified']}/{count} with {sr['reuse_hits']} hits. D-R improves
on D by {low:.2f}–{high:.2f} times in total time. The individual case tables, including
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
'''
    (OUT/'README.md').write_text(report)

    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,1,figsize=(7,6.5),layout='constrained')
    xs=np.arange(len(cases))
    for j,(method,label) in enumerate(zip(METHODS,LABELS)):
        axes[0].plot(xs,[c['summary'][method]['total_seconds'] for c in cases],
                     marker='o',markersize=3,label=label,linewidth=1.3)
    names=[f"n{c['metadata']['n']}/s{c['metadata']['seed']}\n{c['group']}" for c in cases]
    axes[0].set_xticks(xs,names,rotation=45,ha='right',fontsize=8)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Total sequence time (s, median of 3)')
    axes[0].set_title('All setup and returned-candidate fits included')
    axes[0].grid(True,axis='y',alpha=.2); axes[0].legend(ncol=3,fontsize=8)
    chosen=next(c for c in cases if c['id']=='n8_s67_transition')
    energy=np.array([.5*(q['solution_energy_lower']+q['solution_energy_upper'])
                     for q in chosen['validation']['queries']])
    for method,label in [('direct_reuse','Direct reuse'),('offline_certificate_reuse','Solve-first reuse'),
                         ('offline_warm_tree','Warm tree heuristic')]:
        gains=np.array([q['candidate_gain'] for q in chosen['runs'][method][0]['records']])
        axes[1].plot(np.linspace(0,1,len(gains)),100*(1-gains/energy),label=label,linewidth=1.4)
    axes[1].set_xlabel('Transition parameter');axes[1].set_ylabel('Relative squared energy error (%)')
    axes[1].set_title('Transition quality: n=8, seed 67')
    axes[1].legend(fontsize=8);axes[1].grid(True,alpha=.2)
    fig.savefig(OUT/'figure.png',dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in facts.items() if k not in ('cases','source_sha256')},indent=2))


if __name__=='__main__':
    main()
