#!/usr/bin/env bash
# Clona o projeto-base no commit fixado e o instala em modo editável.
# O benchmark roda a implementação ORIGINAL, sem modificações.
set -euo pipefail
BASE_REPO="https://github.com/RitikPatill/ai-image-authenticity-scanner.git"
BASE_COMMIT="4ec7edc"   # único commit auditado na Fase 1
DEST="vendor/base"

mkdir -p vendor
[ -d "$DEST/.git" ] || git clone "$BASE_REPO" "$DEST"
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$BASE_COMMIT"
test -z "$(git -C "$DEST" status --porcelain)" || { echo "base com alterações locais"; exit 1; }
pip install -r "$DEST/requirements.txt"
pip install -e "$DEST"
echo "OK: base em $(git -C "$DEST" rev-parse --short HEAD)"
