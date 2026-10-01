#!/usr/bin/env python3
"""Monta dataset/real e dataset/ai a partir de sources.toml e escreve dataset/manifest.csv.

Reprodutível: amostragem com seed fixa; bytes originais preservados; manifesto com sha256-id.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import random
import shutil
import sys
import tomllib
import urllib.request
import zipfile

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalkit.manifest import Row, make_id, read_manifest, write_manifest  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _sample(items: list, n: int, rng: random.Random) -> list:
    items = sorted(items, key=str)
    return items if n >= len(items) else rng.sample(items, n)


def _group(items: list[tuple[str, object]], src: dict) -> dict[str, list]:
    """Agrupa (nome_relativo, ref) por gerador: subpasta (parent_dir) ou valor fixo."""
    if src.get("generator_from") == "parent_dir":
        groups: dict[str, list] = {}
        for rel, ref in items:
            groups.setdefault(pathlib.PurePosixPath(rel).parent.name or "root", []).append((rel, ref))
        return groups
    return {src.get("generator", ""): items}


def _local(src: dict):
    root = pathlib.Path(src["path"]).expanduser()
    items = [(str(p.relative_to(root)).replace("\\", "/"), p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    return items, lambda ref: pathlib.Path(ref).read_bytes()


def _zip(src: dict, cache: pathlib.Path):
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / pathlib.Path(src["url"]).name
    if not zpath.exists():
        print(f"  baixando {src['url']} ...")
        urllib.request.urlretrieve(src["url"], zpath)
    zf = zipfile.ZipFile(zpath)
    items = [(n, n) for n in zf.namelist() if pathlib.PurePosixPath(n).suffix.lower() in IMG_EXT]
    return items, lambda ref: zf.read(ref)


def _hf(src: dict, n: int, seed: int):
    from datasets import Image as HFImage, load_dataset  # import tardio: dependência opcional

    ds = load_dataset(src["repo"], src.get("config"), split=src.get("split", "train"), streaming=True)
    ds = ds.cast_column(src.get("image_column", "image"), HFImage(decode=False))
    ds = ds.shuffle(seed=seed, buffer_size=2000).take(n)
    col = src.get("image_column", "image")
    items = [(f"{src['name']}/{i}", rec[col]["bytes"]) for i, rec in enumerate(ds)]
    return items, lambda ref: ref


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=pathlib.Path, default=pathlib.Path("sources.toml"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("dataset"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = tomllib.loads(args.sources.read_text())
    manifest_path = args.out / "manifest.csv"
    rows = {r.id: r for r in (read_manifest(manifest_path) if manifest_path.exists() else [])}
    rng = random.Random(args.seed)

    for src in cfg["source"]:
        label, name = src["label"], src["name"]
        assert label in ("real", "ai"), "edited é gerado por make_edited.py"
        print(f"[{name}] kind={src['kind']}")
        kind = src["kind"]
        if kind == "local_dir":
            items, read = _local(src)
        elif kind == "url_zip":
            items, read = _zip(src, pathlib.Path("cache"))
        elif kind == "hf_dataset":
            items, read = _hf(src, src["n"], args.seed)
        else:
            raise ValueError(f"kind desconhecido: {kind}")

        for gen, group in _group(items, src).items():
            chosen = group if kind == "hf_dataset" else _sample(group, src["n"], rng)
            for rel, ref in chosen:
                data = read(ref)
                try:
                    fmt = Image.open(io.BytesIO(data)).format or "UNK"
                except Exception as exc:  # arquivo corrompido: registrar e seguir
                    print(f"  ignorando {rel}: {exc}")
                    continue
                ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif", "BMP": ".bmp"}.get(fmt, ".img")
                dest_dir = args.out / label / name / (gen or "_")
                dest_dir.mkdir(parents=True, exist_ok=True)
                tmp = dest_dir / "_tmp"
                tmp.write_bytes(data)
                rid = make_id(tmp)
                dest = dest_dir / f"{rid}{ext}"
                shutil.move(tmp, dest)
                rows[rid] = Row(
                    id=rid, path=str(dest.relative_to(args.out)), label=label, source=name,
                    generator=gen, license=src.get("license", ""),
                )
            print(f"  {gen or '-'}: {len(chosen)} imagens")

    write_manifest(manifest_path, list(rows.values()))
    print(f"manifesto: {manifest_path} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
