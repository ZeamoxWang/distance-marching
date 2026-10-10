#!/usr/bin/env bash
# Copy this repository (a clean checkout at the current commit, including .git)
# to /pv/distance-marching on the Nautilus PVC through the running CPU pod.
# Jobs copy /pv/distance-marching to node-local scratch before running, so the
# PV checkout is never modified by a job.
#
# Usage: execute/sync_to_pv.sh [pod-name]
set -euo pipefail
NTX="${NTX:-nautilus}"
NNS="${NNS:-ucsd-ravigroup}"
POD="${1:-zimo-deployment-h3-cpu-5dfd8fdbd5-2r2d8}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
  echo "Working tree is dirty; commit first." >&2
  exit 1
fi
echo "Syncing $REPO @ $COMMIT -> $POD:/pv/distance-marching"
tar -C "$REPO" -czf - --exclude='__pycache__' --exclude='.DS_Store' --exclude='*/data' --exclude='*/output_*' . \
  | kubectl --context "$NTX" -n "$NNS" exec -i "$POD" --request-timeout=300s -- \
      bash -c 'rm -rf /pv/distance-marching.new && mkdir -p /pv/distance-marching.new && tar -xzf - -C /pv/distance-marching.new && rm -rf /pv/distance-marching.prev && { [ -d /pv/distance-marching ] && mv /pv/distance-marching /pv/distance-marching.prev || true; } && mv /pv/distance-marching.new /pv/distance-marching && cat /pv/distance-marching/.git/HEAD'
echo "done: $COMMIT"
