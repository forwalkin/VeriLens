#!/usr/bin/env python3
"""Inspeciona o synthbuster.zip SEM extraí-lo e SEM baixar os 12,4 GB.

Fonte: Zenodo, https://zenodo.org/records/10066460 (DOI 10.5281/zenodo.10066460).
Lê apenas o diretório central do ZIP (HTTP Range) + prompts.csv, e imprime/salva:
  1. nomes das pastas; 2. nomes de arquivos (padrão); 3. identificadores;
  4. como prompts.csv se relaciona com as imagens; 5. se é possível selecionar só os pares necessários.
Também lê o cabeçalho (apenas) de poucas imagens por pasta para dimensões/formato/tamanho comprimido.
Saída: dataset/inspection/synthbuster_inspection.json e .md. NADA é assumido: tudo vem do ZIP real.

Uso:
  python scripts/inspect_synthbuster.py                       # remoto (Range)
  python scripts/inspect_synthbuster.py --zip ~/synthbuster.zip   # ZIP local já baixado
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import pathlib
import re
import sys
import zipfile

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalkit.remotezip import open_remote_zip  # noqa: E402

URL = "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1"
EXPECTED_MD5 = "0695bd328e16ea21c5c9cc2ae1d994ff"  # informado na página do registro Zenodo 10066460
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


def shape_of(name: str) -> str:
    """Padrão do nome: dígitos viram 'N', hex longo vira 'H' (para agrupar convenções de nomenclatura)."""
    stem = pathlib.PurePosixPath(name).stem
    return re.sub(r"\d+", "N", re.sub(r"[0-9a-f]{16,}", "H", stem))


def analyze(zf: zipfile.ZipFile, sample_per_dir: int = 3) -> dict:
    infos = [i for i in zf.infolist() if not i.is_dir()]
    imgs = [i for i in infos if pathlib.PurePosixPath(i.filename).suffix.lower() in IMG_EXT]
    others = [i for i in infos if i not in imgs]
    by_dir: dict[str, list[zipfile.ZipInfo]] = collections.defaultdict(list)
    for i in imgs:
        by_dir[str(pathlib.PurePosixPath(i.filename).parent)].append(i)

    dirs = {}
    for d, items in sorted(by_dir.items()):
        names = sorted(pathlib.PurePosixPath(i.filename).name for i in items)
        sizes = [i.file_size for i in items]
        comp = collections.Counter({0: "stored", 8: "deflate"}.get(i.compress_type, str(i.compress_type)) for i in items)
        dims = []
        for i in items[:sample_per_dir]:
            with zf.open(i) as fh:  # só os bytes necessários para o cabeçalho
                im = Image.open(io.BufferedReader(fh, 1 << 16))
                dims.append({"file": pathlib.PurePosixPath(i.filename).name, "size": im.size, "format": im.format, "mode": im.mode})
        dirs[d] = {
            "n_images": len(items),
            "extensions": dict(collections.Counter(pathlib.PurePosixPath(n).suffix.lower() for n in names)),
            "first_names": names[:5], "last_names": names[-3:],
            "name_patterns": dict(collections.Counter(shape_of(n) for n in names).most_common(5)),
            "uncompressed_bytes_total": sum(sizes), "uncompressed_bytes_mean": int(sum(sizes) / len(sizes)),
            "zip_compression": dict(comp), "sampled_headers": dims,
        }
    return {
        "n_entries": len(infos), "n_images": len(imgs),
        "other_files": [{"name": i.filename, "bytes": i.file_size} for i in others],
        "directories": dirs,
    }


def analyze_prompts(zf: zipfile.ZipFile, name: str, dirs: dict) -> dict:
    with zf.open(name) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
        rows = list(csv.reader(text))
    header, body = rows[0], rows[1:]
    out = {"file": name, "n_rows": len(body), "columns": header, "first_rows": body[:3]}
    # Quais colunas parecem identificar imagens? Compara valores de cada coluna com nomes de arquivo/stems/pastas.
    stems = {d: {pathlib.PurePosixPath(n).stem for n in info["first_names"] + info["last_names"]} for d, info in dirs.items()}
    all_stems = set().union(*stems.values()) if stems else set()
    cand = {}
    for ci, col in enumerate(header):
        vals = [r[ci] for r in body if ci < len(r)]
        hit_stem = sum(v in all_stems or pathlib.PurePosixPath(v).stem in all_stems for v in vals)
        cand[col] = {"distinct": len(set(vals)), "example": vals[:3], "matches_known_stems_(amostra)": hit_stem}
    out["column_profile"] = cand
    return out


def verdict(report: dict) -> list[str]:
    """Respostas objetivas às 5 perguntas, derivadas do que foi encontrado (e do que NÃO foi)."""
    d = report["directories"]
    lines = [f"1. Pastas com imagens ({len(d)}): " + ", ".join(sorted(d)),
             "2. Padrões de nome por pasta: " + "; ".join(f"{k}: {list(v['name_patterns'])}" for k, v in sorted(d.items()))]
    counts = {v["n_images"] for v in d.values()}
    lines.append(f"3. Imagens por pasta: {sorted(counts)} (esperado 1000 por gerador). "
                 "Compare os padrões acima entre pastas: se forem iguais, o identificador é o nome do arquivo.")
    pr = report.get("prompts")
    if pr:
        lines.append(f"4. {pr['file']}: {pr['n_rows']} linhas; colunas = {pr['columns']}. Veja column_profile para quais colunas casam com nomes de imagem.")
    else:
        lines.append("4. prompts.csv NÃO encontrado no ZIP (ver other_files).")
    lines.append("5. Seleção parcial: o ZIP tem diretório central; membros individuais podem ser lidos por Range. "
                 "Se a compressão for 'stored'/'deflate' por arquivo (campo zip_compression), basta ler só os escolhidos.")
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=pathlib.Path, help="ZIP local (em vez de ler remoto por Range)")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("dataset/inspection"))
    ap.add_argument("--sample-per-dir", type=int, default=3)
    ap.add_argument("--verify-md5", action="store_true", help="com --zip: confere o md5 do arquivo local (lê 12,4 GB do disco)")
    args = ap.parse_args(argv)
    if args.verify_md5:
        import hashlib
        if not args.zip:
            sys.exit("--verify-md5 exige --zip")
        h = hashlib.md5()
        with open(args.zip, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        print(f"md5 {h.hexdigest()} ({'OK' if h.hexdigest() == EXPECTED_MD5 else 'DIFERENTE do esperado ' + EXPECTED_MD5})")

    if args.zip:
        zf, remote = zipfile.ZipFile(args.zip), None
    else:
        zf, remote = open_remote_zip(args.url)
    report = analyze(zf, args.sample_per_dir)
    pname = next((i.filename for i in zf.infolist() if pathlib.PurePosixPath(i.filename).name.lower() == "prompts.csv"), None)
    if pname:
        report["prompts"] = analyze_prompts(zf, pname, report["directories"])
    if remote:
        report["transfer"] = {"zip_total_bytes": remote.size, "bytes_fetched": remote.bytes_fetched, "http_requests": remote.requests}
    report["answers"] = verdict(report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "synthbuster_inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = ["# Inspeção do synthbuster.zip", "", *[f"- {a}" for a in report["answers"]], ""]
    if "transfer" in report:
        t = report["transfer"]
        md += [f"Transferido: {t['bytes_fetched'] / 1e6:.1f} MB em {t['http_requests']} requisições (ZIP total {t['zip_total_bytes'] / 1e9:.2f} GB).", ""]
    md += ["## Pastas", "", "| pasta | n | extensões | MB médios | compressão no ZIP | 1ª imagem (dim/formato) |", "|---|---|---|---|---|---|"]
    for d, v in sorted(report["directories"].items()):
        h = v["sampled_headers"][0] if v["sampled_headers"] else {}
        md.append(f"| {d} | {v['n_images']} | {v['extensions']} | {v['uncompressed_bytes_mean'] / 1e6:.2f} | {v['zip_compression']} | {h.get('size')} {h.get('format')} |")
    if "prompts" in report:
        p = report["prompts"]
        md += ["", f"## {p['file']}", "", f"colunas: `{p['columns']}`", "", "primeiras linhas:", "```", *[repr(r) for r in p["first_rows"]], "```"]
    (args.out / "synthbuster_inspection.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
