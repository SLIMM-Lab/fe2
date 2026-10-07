#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [ "$#" -ne 1 ]; then
  echo "Usage: bash go.sh /absolute/path/to/exported-weights.pt" >&2
  exit 2
fi
weights_file=$(realpath -- "$1")
test -f "$weights_file"
weights_property=${weights_file//\\/\\\\}
weights_property=${weights_property//\"/\\\"}
python ../fe2_j2/create_macro_mesh.py -nopopup
# Prevent nested BLAS worker pools inside SolidModel's element workers.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
../../../src/fe2 macro.pro -p "model.model.bulk.material.weights = \"$weights_property\";"
