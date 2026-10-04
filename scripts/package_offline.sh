#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-offline_bundle}"
mkdir -p "$OUT/wheels"
python -m pip download -r requirements/train.txt --dest "$OUT/wheels"
tar --exclude='.git' --exclude='runs' --exclude='data/raw' -czf "$OUT/project.tar.gz" .
cat > "$OUT/INSTALL.txt" <<'EOF'
python -m venv .venv
. .venv/bin/activate
python -m pip install --no-index --find-links wheels -r requirements/train.txt
python -m pip install --no-index --find-links wheels -e .
EOF
sha256sum "$OUT"/project.tar.gz "$OUT"/wheels/* > "$OUT/SHA256SUMS"
