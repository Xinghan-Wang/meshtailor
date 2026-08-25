#!/bin/bash
# Smoke test for abf_unwrap on a single garment.
set -e
export LD_LIBRARY_PATH=/root/abf_toolkit/geogram/build/Release/lib:${LD_LIBRARY_PATH}
BIN=/root/abf_toolkit/tool/build/abf_unwrap
G="$1"
mkdir -p "$(dirname "$G")"
"$BIN" "$G/mesh.obj" "$G/seam.json" "$G/uv_abf.obj"
echo "exit=$?"
echo "vt=$(grep -c '^vt ' "$G/uv_abf.obj") v=$(grep -c '^v ' "$G/uv_abf.obj") f=$(grep -c '^f ' "$G/uv_abf.obj")"
echo "--- first 3 lines ---"
head -3 "$G/uv_abf.obj"
