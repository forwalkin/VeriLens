#!/usr/bin/env python3
"""Auditoria de vieses do dataset ANTES de avaliar detectores.

Um detector pode 'acertar' por motivos triviais (formato, resolução, EXIF) em vez de
conteúdo gerado. Aqui medimos o quanto metadados simples já separam real de IA (AUC
univariado). AUC >= 0.8 ou <= 0.2 em um atributo = confundidor forte a reportar.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalkit.manifest import read_manifest  # noqa: E402
from evalkit.metrics import auc  # noqa: E402


def features(path: pathlib.Path) -> dict[str, float]:
    with Image.open(path) as im:
        w, h = im.size
        fmt = im.format
        exif = bool(im.getexif())
    size = path.stat().st_size
    return {
        "log_megapixels": float(np.log10(w * h / 1e6)),
        "bytes_per_pixel": size / (w * h),
        "is_jpeg": float(fmt == "JPEG"),
        "has_exif": float(exif),
        "is_square": float(w == h),
        "aspect_ratio": max(w, h) / min(w, h),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("dataset"))
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    rows = read_manifest(args.dataset / "manifest.csv")
    feats = {r.id: features(args.dataset / r.path) for r in rows}
    lines = ["# Auditoria do dataset", ""]

    cnt = collections.Counter((r.label, r.generator or "-") for r in rows)
    lines += ["## Contagem por classe / gerador-origem", "", "| classe | gerador/origem | n |", "|---|---|---|"]
    lines += [f"| {k[0]} | {k[1]} | {v} |" for k, v in sorted(cnt.items())]

    sub = [r for r in rows if r.label in ("real", "ai")]
    y = np.array([r.is_ai for r in sub])
    lines += ["", f"## Confundidores (real n={int((~y).sum())} vs IA n={int(y.sum())})", "",
              "| atributo | média real | média IA | AUC univariado (IA=positivo) | alerta |", "|---|---|---|---|---|"]
    for name in next(iter(feats.values())):
        v = np.array([feats[r.id][name] for r in sub])
        a = auc(y, v)
        warn = "**CONFUNDIDOR**" if a == a and (a >= 0.8 or a <= 0.2) else ""
        lines.append(f"| {name} | {v[~y].mean():.3g} | {v[y].mean():.3g} | {a:.3f} | {warn} |")
    lines += ["", "Atributos marcados permitem separar as classes sem olhar o conteúdo: "
              "interprete os resultados dos detectores com cautela ou equalize essa propriedade."]
    text = "\n".join(lines)
    print(text)
    out = args.out or args.dataset / "AUDIT.md"
    out.write_text(text + "\n")


if __name__ == "__main__":
    main()
