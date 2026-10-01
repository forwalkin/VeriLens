#!/usr/bin/env python3
"""Gera dataset/edited: fotos REAIS editadas de forma determinística.

Para cada edição escolhe --per-edit pais distintos entre as imagens reais do manifesto.
Pais podem se repetir entre edições diferentes (documentado; o relatório não usa as
edições para o IC do AUC principal, justamente para evitar pseudo-replicação).
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
import zlib

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalkit.edits import EDITS, EXT, apply_edit  # noqa: E402
from evalkit.manifest import Row, make_id, read_manifest, write_manifest  # noqa: E402

MIN_SIDE = 256


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("dataset"))
    ap.add_argument("--per-edit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    mpath = args.dataset / "manifest.csv"
    rows = {r.id: r for r in read_manifest(mpath)}
    parents = sorted((r for r in rows.values() if r.label == "real"), key=lambda r: r.id)
    if not parents:
        sys.exit("nenhuma imagem real no manifesto; rode build_dataset.py primeiro")
    # descarta edições antigas para a geração ser idempotente
    rows = {k: v for k, v in rows.items() if v.label != "edited"}

    for edit in EDITS:
        rng = random.Random(args.seed * 1_000_003 + zlib.crc32(edit.encode()))
        pool = [p for p in parents if min(Image.open(args.dataset / p.path).size) >= MIN_SIDE]
        for parent in rng.sample(pool, min(args.per_edit, len(pool))):
            im = Image.open(args.dataset / parent.path)
            im.load()
            out, fmt, kw = apply_edit(edit, im, seed=zlib.crc32((parent.id + edit).encode()))
            d = args.dataset / "edited" / edit
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "_tmp"
            out.save(tmp, format=fmt, **kw)
            rid = make_id(tmp)
            dest = d / f"{parent.id}{EXT[fmt]}"
            tmp.replace(dest)
            rows[rid] = Row(
                id=rid, path=str(dest.relative_to(args.dataset)), label="edited",
                source="make_edited", generator=parent.generator, license=parent.license,
                parent_id=parent.id, edit=edit,
            )
        print(f"{edit}: ok")
    write_manifest(mpath, list(rows.values()))
    print(f"manifesto atualizado: {sum(r.label == 'edited' for r in rows.values())} editadas")


if __name__ == "__main__":
    main()
