#!/usr/bin/env bash
# Download one train shard + features.json + dataset_info.json per OXE dataset.
# Targets only the 13 Franka subsets analyzed in OXE_Franka抓取数据集分析.xlsx.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
GCS_BASE="https://storage.googleapis.com/gresearch/robotics"

# dataset:version pairs (all current OXE Franka subsets are v0.1.0)
DATASETS=(
  "viola"
  "stanford_hydra_dataset_converted_externally_to_rlds"
  "austin_buds_dataset_converted_externally_to_rlds"
  "austin_sailor_dataset_converted_externally_to_rlds"
  "austin_sirius_dataset_converted_externally_to_rlds"
  "utaustin_mutex"
  "furniture_bench_dataset_converted_externally_to_rlds"
  "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
  "taco_play"
  "nyu_franka_play_dataset_converted_externally_to_rlds"
  "ucsd_kitchen_dataset_converted_externally_to_rlds"
  "ucsd_pick_and_place_dataset_converted_externally_to_rlds"
  "cmu_franka_exploration_dataset_converted_externally_to_rlds"
)
VERSION="0.1.0"

# Read shard counts from oxe_manifest.json so the filename suffix matches GCS exactly
get_total_shards() {
  python3 - <<PY "$1"
import json, sys
ds = sys.argv[1]
m = json.load(open("$ROOT/scripts/oxe_manifest.json"))
name = m[ds]["train_shard0"][0]   # e.g. ".../viola-train.tfrecord-00000-of-00081"
print(name.rsplit("-of-", 1)[1])
PY
}

for ds in "${DATASETS[@]}"; do
  total=$(get_total_shards "$ds")
  shard_name="${ds}-train.tfrecord-00000-of-${total}"
  out_dir="$RAW/$ds/$VERSION"
  mkdir -p "$out_dir"
  echo "== $ds (1/${total} shards) =="
  for f in "features.json" "dataset_info.json" "$shard_name"; do
    if [[ -s "$out_dir/$f" ]]; then
      echo "  [skip] $f"
    else
      echo "  [get ] $f"
      curl -fL --retry 3 --retry-delay 2 -o "$out_dir/$f" \
        "$GCS_BASE/$ds/$VERSION/$f"
    fi
  done
done
echo "Done. Raw data under: $RAW"
