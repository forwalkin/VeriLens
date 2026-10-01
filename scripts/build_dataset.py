#!/usr/bin/env python3
"""Monta dataset/real e dataset/ai a partir de sources.toml e escreve dataset/manifest.csv.

Reprodutível: amostragem com seed fixa; bytes originais preservados; manifesto com sha256-id.
"""

from __future__ import annotations

import argparse
import csv
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


def _remote_zip(src: dict):
    """ZIP remoto lido por HTTP Range: só o índice e os membros escolhidos são transferidos (sem baixar o ZIP todo)."""
    from evalkit.remotezip import open_remote_zip  # import tardio

    zf, _ = open_remote_zip(src["url"])
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


def _stem(rel: str) -> str:
    return pathlib.PurePosixPath(rel).stem


def _load_pair_map(path: pathlib.Path | None) -> dict[str, str]:
    """CSV opcional com colunas ai_key,real_key (quando os nomes de arquivo não coincidem)."""
    if not path:
        return {}
    with open(path, newline="") as f:
        return {r["ai_key"]: r["real_key"] for r in csv.DictReader(f)}


def _plan_pairs(cfg: dict, listed: dict, rng: random.Random) -> dict:
    """Escolhe um conjunto único de chaves presentes no real E em todos os geradores de IA.

    Retorna {nome_da_fonte: {gerador: [(rel, ref, pair_key)]}} já restrito às chaves escolhidas.
    Observação: o Synthbuster foi gerado a partir de DESCRIÇÕES das imagens do RAISE-1k, então o
    par compartilha conteúdo/tema, não é a mesma imagem. A estrutura de nomes NÃO foi verificada.
    """
    pc = cfg["pairing"]
    pair_map = _load_pair_map(pathlib.Path(pc["map_csv"]).expanduser() if pc.get("map_csv") else None)
    real_items = listed[pc["real_source"]][0]  # grupos -> um único grupo
    real_by_key = {_stem(rel): (rel, ref) for g in real_items.values() for rel, ref in g}
    per_gen: dict[tuple[str, str], dict[str, tuple]] = {}
    for name in pc["ai_sources"]:
        for gen, items in listed[name][0].items():
            per_gen[(name, gen)] = {pair_map.get(_stem(rel), _stem(rel)): (rel, ref) for rel, ref in items}
    keys = set(real_by_key)
    for m in per_gen.values():
        keys &= set(m)
    if not keys:
        ex_r = sorted(real_by_key)[:3]
        ex_a = [sorted(m)[:3] for m in per_gen.values()][:1]
        sys.exit(f"pareamento vazio. Exemplos de chave real: {ex_r}; IA: {ex_a}. "
                 "Forneça [pairing].map_csv com colunas ai_key,real_key.")
    n = min(pc["n_pairs"], len(keys))
    if n < pc["n_pairs"]:
        print(f"  AVISO: só {len(keys)} chaves em comum (pedido: {pc['n_pairs']})")
    chosen = sorted(rng.sample(sorted(keys), n))
    plan = {pc["real_source"]: {"": [(*real_by_key[k], k) for k in chosen]}}
    for (name, gen), m in per_gen.items():
        plan.setdefault(name, {})[gen] = [(*m[k], k) for k in chosen]
    return plan


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

    srcs = {src["name"]: src for src in cfg["source"]}
    for src in srcs.values():
        assert src["label"] in ("real", "ai"), "edited é gerado por make_edited.py"

    # 1) listar candidatos de cada fonte
    listed: dict[str, tuple[dict, object]] = {}
    for name, src in srcs.items():
        kind = src["kind"]
        print(f"[{name}] kind={kind}")
        if kind == "local_dir":
            items, read = _local(src)
        elif kind == "url_zip":
            items, read = _zip(src, pathlib.Path("cache"))
        elif kind == "remote_zip":
            items, read = _remote_zip(src)
        elif kind == "hf_dataset":
            items, read = _hf(src, src["n"], args.seed)
        else:
            raise ValueError(f"kind desconhecido: {kind}")
        listed[name] = (_group(items, src), read)

    # 2) selecionar: pareado (se [pairing]) ou amostragem simples por fonte/gerador
    plan: dict[str, dict[str, list]] = {}
    paired = set()
    if "pairing" in cfg:
        pc = cfg["pairing"]
        for nm in [pc["real_source"], *pc["ai_sources"]]:
            if srcs[nm]["kind"] == "hf_dataset":
                sys.exit("pareamento não suporta kind=hf_dataset")
        plan.update(_plan_pairs(cfg, listed, rng))
        paired = set(plan)
    for name, src in srcs.items():
        if name in paired:
            continue
        groups, _ = listed[name]
        plan[name] = {gen: [(rel, ref, "") for rel, ref in (g if src["kind"] == "hf_dataset" else _sample(g, src["n"], rng))]
                      for gen, g in groups.items()}

    # 3) copiar bytes originais e registrar no manifesto
    for name, groups in plan.items():
        src, read = srcs[name], listed[name][1]
        for gen, chosen in groups.items():
            gen_name = gen or src.get("generator", "")
            for rel, ref, key in chosen:
                data = read(ref)
                try:
                    fmt = Image.open(io.BytesIO(data)).format or "UNK"
                except Exception as exc:  # arquivo corrompido: registrar e seguir
                    print(f"  ignorando {rel}: {exc}")
                    continue
                ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif", "BMP": ".bmp"}.get(fmt, ".img")
                dest_dir = args.out / src["label"] / name / (gen_name or "_")
                dest_dir.mkdir(parents=True, exist_ok=True)
                tmp = dest_dir / "_tmp"
                tmp.write_bytes(data)
                rid = make_id(tmp)
                dest = dest_dir / f"{rid}{ext}"
                shutil.move(tmp, dest)
                rows[rid] = Row(
                    id=rid, path=str(dest.relative_to(args.out)), label=src["label"], source=name,
                    generator=gen_name, license=src.get("license", ""), pair_key=key,
                )
            print(f"  {name}/{gen_name or '-'}: {len(chosen)} imagens" + (" (pareadas)" if name in paired else ""))

    write_manifest(manifest_path, list(rows.values()))
    print(f"manifesto: {manifest_path} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
