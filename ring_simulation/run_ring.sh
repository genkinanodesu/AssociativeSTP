#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N ring_cfg
#$ -o logs/ring_cfg.o$JOB_ID
#$ -e logs/ring_cfg.e$JOB_ID
#$ -pe OpenMP 48

set -euo pipefail

resolve_run_dir() {
  local here script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local -a candidates=()

  if [[ -n "${RUN_RING_DIR:-}" ]]; then
    candidates+=("$RUN_RING_DIR")
  fi
  candidates+=(
    "$PWD"
    "$PWD/ring_simulation"
  )
  if [[ -n "${SGE_O_WORKDIR:-}" ]]; then
    candidates+=(
      "$SGE_O_WORKDIR"
      "$SGE_O_WORKDIR/ring_simulation"
    )
  fi
  candidates+=("$script_dir")

  for here in "${candidates[@]}"; do
    if [[ -f "$here/run_ring.sh" && -f "$here/run_from_cfg.py" ]]; then
      printf '%s\n' "$here"
      return 0
    fi
  done
  return 1
}

resolve_activate() {
  local cand
  local -a candidates=(
    "$RUN_DIR/../.venv/bin/activate"
    "$RUN_DIR/../../.venv/bin/activate"
    "$RUN_DIR/.venv/bin/activate"
    "$PWD/.venv/bin/activate"
    "$PWD/../.venv/bin/activate"
    "$PWD/../../.venv/bin/activate"
  )
  if [[ -n "${SGE_O_WORKDIR:-}" ]]; then
    candidates+=(
      "$SGE_O_WORKDIR/.venv/bin/activate"
      "$SGE_O_WORKDIR/../../.venv/bin/activate"
    )
  fi
  for cand in "${candidates[@]}"; do
    if [[ -f "$cand" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

RUN_DIR="$(resolve_run_dir)" || {
  echo "Could not resolve run directory for run_ring.sh." >&2
  exit 1
}
export RUN_RING_DIR="$RUN_DIR"

ACTIVATE_PATH="$(resolve_activate)" || {
  echo "Could not find virtualenv activate script (.venv/bin/activate)." >&2
  exit 1
}
source "$ACTIVATE_PATH"
mkdir -p "$RUN_DIR/logs"

CFG="${1:?usage: $0 cfg.json [run_from_cfg args...] [--compare-only] [--stp-type TYPE ...]}"
shift

stp_types=()
pass_args=()
compare_only=0
skip_compare=0
paper_mode=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stp-type)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Missing value for --stp-type" >&2
        exit 2
      fi
      stp_types+=("$1")
      ;;
    --compare-only|--comparison-only)
      compare_only=1
      ;;
    --skip-compare)
      skip_compare=1
      ;;
    --paper)
      paper_mode=1
      pass_args+=("$1")
      ;;
    *)
      pass_args+=("$1")
      ;;
  esac
  shift
done

if [[ ${#stp_types[@]} -eq 0 ]]; then
  stp_types=(with_stp pre_stp no_stp)
fi

run_compare() {
  local -a cmd
  cmd=(python -u "$RUN_DIR/compare_condition_runs.py" "$CFG" --reuse-only)
  if [[ "$paper_mode" -eq 1 ]]; then
    cmd+=(--paper)
  fi
  for stp in "${stp_types[@]}"; do
    cmd+=(--stp-type "$stp")
  done
  "${cmd[@]}"
}

run_training_once() {
  local stp="$1"
  python -u "$RUN_DIR/run_from_cfg.py" "$CFG" "${pass_args[@]}" --stp-type "$stp"
}

if [[ -n "${JOB_ID:-}" ]]; then
  if [[ "$compare_only" -eq 1 ]]; then
    run_compare
  else
    for stp in "${stp_types[@]}"; do
      run_training_once "$stp"
    done
  fi
  exit 0
fi

if [[ "$compare_only" -eq 1 ]]; then
  run_compare
  exit 0
fi

if command -v qsub >/dev/null 2>&1; then
  job_ids=()
  for stp in "${stp_types[@]}"; do
    submit_out="$(qsub -v "RUN_RING_DIR=$RUN_DIR" -N "ring_${stp}" "$RUN_DIR/run_ring.sh" "$CFG" "${pass_args[@]}" --skip-compare --stp-type "$stp")"
    echo "$submit_out"
    job_id="$(echo "$submit_out" | grep -Eo '[0-9]+' | head -n 1 || true)"
    if [[ -n "$job_id" ]]; then
      job_ids+=("$job_id")
    fi
  done

  if [[ "$skip_compare" -eq 0 ]]; then
    if [[ ${#job_ids[@]} -eq ${#stp_types[@]} ]]; then
      hold_jid="$(IFS=,; echo "${job_ids[*]}")"
      compare_submit_args=("$CFG" --compare-only)
      if [[ "$paper_mode" -eq 1 ]]; then
        compare_submit_args+=(--paper)
      fi
      for stp in "${stp_types[@]}"; do
        compare_submit_args+=(--stp-type "$stp")
      done
      qsub -v "RUN_RING_DIR=$RUN_DIR" -N "ring_compare" -hold_jid "$hold_jid" "$RUN_DIR/run_ring.sh" "${compare_submit_args[@]}"
    else
      echo "Warning: failed to parse all qsub job IDs; skipping automatic comparison submission." >&2
    fi
  fi
else
  echo "qsub not found; running sequentially on this node."
  for stp in "${stp_types[@]}"; do
    run_training_once "$stp"
  done
  if [[ "$skip_compare" -eq 0 ]]; then
    run_compare
  fi
fi
