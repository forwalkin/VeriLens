#!/usr/bin/env python3
"""Gera dataset/edited (fotos REAIS editadas) e, por padrão, IA editada (label ai_edited).

Para cada edição escolhe --per-edit pais distintos por classe. Pais podem se repetir entre
edições diferentes; por isso o relatório usa bootstrap por cluster (imagem-pai) nos grupos de edição.
A IA editada permite medir robustez do RECALL/AUC (só com real editada mediríamos apenas falsos positivos).
Os pais de IA são escolhidos em rodízio entre os geradores, para equilibrar a mistura.
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
    ap.add_argument("--per-edit", type=int, default=30, help="pais por edição e por classe")
    ap.add_argument("--no-ai-edited", action="store_true", help="não gerar IA editada")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    mpath = args.dataset / "manifest.csv"
    rows = {r.id: r for r in read_manifest(mpath)}
    all_rows = list(rows.values())
    pools = {"edited": sorted((r for r in all_rows if r.label == "real"), key=lambda r: r.id)}
    if not args.no_ai_edited:
        pools["ai_edited"] = sorted((r for r in all_rows if r.label == "ai"), key=lambda r: r.id)
    if not pools["edited"]:
        sys.exit("nenhuma imagem real no manifesto; rode build_dataset.py primeiro")
    # descarta edições antigas para a geração ser idempotente
    rows = {k: v for k, v in rows.items() if v.label not in ("edited", "ai_edited")}
    sizes: dict[str, bool] = {}

    def big_enough(r: Row) -> bool:
        if r.id not in sizes:
            with Image.open(args.dataset / r.path) as im:
                sizes[r.id] = min(im.size) >= MIN_SIDE
        return sizes[r.id]

    def pick(label: str, pool: list[Row], n: int, rng: random.Random) -> list[Row]:
        pool = [p for p in pool if big_enough(p)]
        if label != "ai_edited":
            return rng.sample(pool, min(n, len(pool)))
        by_gen: dict[str, list[Row]] = {}
        for p in pool:
            by_gen.setdefault(p.generator, []).append(p)
        for g in by_gen.values():
            rng.shuffle(g)
        out: list[Row] = []
        while len(out) < n and any(by_gen.values()):
            for g in sorted(by_gen):
                if by_gen[g] and len(out) < n:
                    out.append(by_gen[g].pop())
        return out

    for edit in EDITS:
        for label, pool in pools.items():
            rng = random.Random(args.seed * 1_000_003 + zlib.crc32((edit + label).encode()))
            for parent in pick(label, pool, args.per_edit, rng):
                im = Image.open(args.dataset / parent.path)
                im.load()
                out, fmt, kw = apply_edit(edit, im, seed=zlib.crc32((parent.id + edit).encode()))
                d = args.dataset / label / edit
                d.mkdir(parents=True, exist_ok=True)
                tmp = d / "_tmp"
                out.save(tmp, format=fmt, **kw)
                rid = make_id(tmp)
                dest = d / f"{parent.id}{EXT[fmt]}"
                tmp.replace(dest)
                rows[rid] = Row(
                    id=rid, path=str(dest.relative_to(args.dataset)), label=label,
                    source="make_edited", generator=parent.generator, license=parent.license,
                    parent_id=parent.id, edit=edit,
                )
        print(f"{edit}: ok")
    write_manifest(mpath, list(rows.values()))
    n_e = sum(r.label == "edited" for r in rows.values())
    n_a = sum(r.label == "ai_edited" for r in rows.values())
    print(f"manifesto atualizado: {n_e} reais editadas, {n_a} IA editadas")


if __name__ == "__main__":
    main()
