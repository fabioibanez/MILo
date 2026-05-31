#!/usr/bin/env bash
# Wrapper for the common Modal workflows for MILo.
#
# Usage:
#   ./run_modal.sh setup                       # install modal + auth (one-time)
#   ./run_modal.sh upload <local-scene-dir>    # push a COLMAP scene to the data volume
#   ./run_modal.sh train  <scene> [extra...]   # train + extract mesh
#   ./run_modal.sh sweep  <scene> [extra...]   # hyperparameter sweep (stage 1)
#   ./run_modal.sh eval   <scene> [extra...]   # DTU Chamfer eval on sweep runs (stage 2)
#   ./run_modal.sh fetch  <scene>              # pull trained outputs back locally
#   ./run_modal.sh shell                       # drop into an interactive container
#
# Examples:
#   ./run_modal.sh upload ./milo/data/Ignatius
#   ./run_modal.sh train Ignatius --imp-metric outdoor --rasterizer radegs
#   ./run_modal.sh train Ignatius --extra-args "--dense_gaussians --decoupled_appearance"
#   ./run_modal.sh fetch Ignatius

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP="modal_app.py"
OUTPUT_DIR="./milo/output"

cmd="${1:-}"
shift || true

case "$cmd" in
  setup)
    if ! command -v modal >/dev/null 2>&1; then
      echo "[run_modal] installing modal via uv tool..."
      uv tool install modal
    fi
    modal setup
    ;;

  upload)
    local_path="${1:?usage: ./run_modal.sh upload <local-scene-dir>}"
    modal run "$APP::upload_scene" --local-path "$local_path"
    ;;

  train)
    scene="${1:?usage: ./run_modal.sh train <scene> [--imp-metric ...] [--rasterizer ...] [--extra-args ...]}"
    shift
    modal run "$APP::main" --scene "$scene" "$@"
    ;;

  sweep)
    scene="${1:?usage: ./run_modal.sh sweep <scene> [--topo-weight ...] [--no-parallel]}"
    shift
    modal run "$APP::sweep" --scene "$scene" "$@"
    ;;

  eval)
    scene="${1:?usage: ./run_modal.sh eval <scene> [--run-names a,b] [--no-parallel]}"
    shift
    modal run "$APP::eval_sweep" --scene "$scene" "$@"
    ;;

  fetch)
    scene="${1:?usage: ./run_modal.sh fetch <scene>}"
    mkdir -p "$OUTPUT_DIR"
    modal volume get milo-outputs "$scene" "$OUTPUT_DIR/"
    ;;

  shell)
    modal shell "$APP::train"
    ;;

  ""|-h|--help|help)
    sed -n '2,16p' "$0"
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    sed -n '2,16p' "$0"
    exit 1
    ;;
esac
