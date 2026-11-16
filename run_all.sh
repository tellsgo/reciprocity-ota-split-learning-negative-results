#!/bin/sh
set -eu
cd "$(dirname "$0")"
case "${1:-quick}" in
  quick) "${PYTHON:-python3}" check_table_c.py; "${PYTHON:-python3}" check_logged.py ;;
  c) "${PYTHON:-python3}" check_table_c.py ;;
  logged) "${PYTHON:-python3}" check_logged.py ;;
  figure-b) "${PYTHON:-python3}" plot_test_b.py ;;
  a|b|f|g) "${PYTHON:-python3}" check_logged.py ;;
  c1|c2|c3|c4|c5|c6|c7) "${PYTHON:-python3}" check_table_c.py ;;
  c8|c9|c10) echo "$1: archived source only; no logged-output check is packaged" ;;
  d1|d2|d3|e1|e2)
    name=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')
    mkdir -p "experiments/$name/results" "experiments/$name/tmp" "experiments/$name/figures/cache"
    "${PYTHON:-python3}" "experiments/$name/src/run.py" ;;
  *) echo 'Available targets: quick, a, b, figure-b, c, c1-c10, d1-d3, e1-e2, f, g, logged' >&2; exit 2 ;;
esac
