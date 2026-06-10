#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/download_sabdab_nano_temporal.sh" >&2
  exit 2
fi

OUT_ROOT="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT_ROOT/scripts"

for helper in \
  sabdab_nano_temporal_common.py \
  download_sabdab_nano_temporal.py \
  build_sabdab_nano_temporal_splits.py \
  verify_sabdab_nano_temporal.py
do
  src="$SCRIPT_DIR/$helper"
  dst="$OUT_ROOT/scripts/$helper"
  if [[ "$src" != "$dst" && ! -e "$dst" ]]; then
    ln -s "$src" "$dst" 2>/dev/null || cp "$src" "$dst"
  fi
done

python scripts/download_sabdab_nano_temporal.py --output-root "$OUT_ROOT"
python scripts/build_sabdab_nano_temporal_splits.py --output-root "$OUT_ROOT"
python scripts/verify_sabdab_nano_temporal.py --output-root "$OUT_ROOT"

echo "Split manifests:"
echo "$OUT_ROOT/data/sabdab_nano_temporal/manifests/single_nano_train_pre2022_manifest.csv"
echo "$OUT_ROOT/data/sabdab_nano_temporal/manifests/single_nano_val_2022h1_manifest.csv"
echo "$OUT_ROOT/data/sabdab_nano_temporal/manifests/single_nano_test_2022h2_manifest.csv"
echo "$OUT_ROOT/data/sabdab_nano_temporal/manifests/complex_nanoag_train_pre2022_manifest.csv"
echo "$OUT_ROOT/data/sabdab_nano_temporal/manifests/complex_nanoag_val_2022h1_manifest.csv"
echo "$OUT_ROOT/data/sabdab_nano_temporal/manifests/complex_nanoag_test_2022h2_manifest.csv"
echo "Verification report:"
echo "$OUT_ROOT/data/sabdab_nano_temporal/manifests/verify_report.json"
