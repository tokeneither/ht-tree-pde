#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$project_root/.cache/matplotlib}"
python_bin="${PYTHON:-python}"
mode="${1:-smoke}"
result_root="results/generated"

case "$mode" in
  smoke)
    "$python_bin" -B scripts/benchmark_compression_gap.py --sizes 32 128 256 \
      --exhaustive-max 6 --output "$result_root/gap"
    "$python_bin" -B scripts/audit_compression_extension.py --output "$result_root/gap/dirichlet.json"
    "$python_bin" -B scripts/benchmark_ht_v3.py --groups small control \
      --joint-schedule early --baseline shared --repeats 1 --output "$result_root/ht"
    "$python_bin" -B scripts/benchmark_certificate_reuse.py --n 6 --seeds 0 \
      --queries 12 --repeats 1 --groups local transition --output "$result_root/reuse_demo"
    ;;
  full)
    "$python_bin" -B scripts/benchmark_compression_gap.py --output "$result_root/gap"
    "$python_bin" -B scripts/audit_compression_extension.py --output "$result_root/gap/dirichlet.json"
    "$python_bin" -B scripts/benchmark_ht_v3.py --groups small band band_fresh control \
      --joint-schedule early --baseline shared --repeats 3 --output "$result_root/ht"
    "$python_bin" -B scripts/benchmark_certificate_reuse.py --n 8 --seeds 41 67 \
      --queries 64 --repeats 3 --groups local stable transition --output "$result_root/reuse_n8"
    "$python_bin" -B scripts/benchmark_certificate_reuse.py --n 12 --seeds 89 \
      --queries 32 --repeats 3 --groups local stable transition --output "$result_root/reuse_n12"
    "$python_bin" -B scripts/report_certificate_reuse.py \
      --sources "$result_root/reuse_n8/all_results.json" "$result_root/reuse_n12/all_results.json" \
      --output "$result_root/reuse_report"
    ;;
  *)
    printf 'Usage: bash scripts/reproduce.sh [smoke|full]\n' >&2
    exit 2
    ;;
esac
