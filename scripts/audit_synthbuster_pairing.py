#!/usr/bin/env python3
"""Auditoria COMPLETA do pareamento entre os nomes das imagens do Synthbuster e as chaves do prompts.csv.

Etapa 2 do fluxo (TASK-0002). Lê só a listagem do ZIP (diretório central, via HTTP Range) e o prompts.csv.
NÃO baixa nem abre imagens, NÃO monta dataset, NÃO altera nada. Compara TODOS os nomes de cada pasta com
TODAS as chaves da coluna candidata, por gerador, sem normalização destrutiva:

  * exata (nome com extensão): chave == nome do arquivo
  * por stem (nome sem extensão): chave == stem do arquivo
  * diferenças só de caixa/espaços: listadas à parte e NUNCA contadas como correspondência

Saída (locais, fora do git): <out>/synthbuster_pairing_audit.json e .md. Os relatórios só contêm nomes de
arquivos e chaves; nenhum texto de prompt é copiado.

Uso:
  python scripts/audit_synthbuster_pairing.py                  # remoto (Range)
  python scripts/audit_synthbuster_pairing.py --zip synthbuster.zip   # ZIP local
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalkit.remotezip import open_remote_zip  # noqa: E402

URL = "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1"
KEY_COLUMN = "image name (matching Raise-1k)"
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}  # mesmo conjunto do inspect_synthbuster
EXAMPLES = 10  # exemplos no Markdown; o JSON traz as listas completas


def _norm(s: str) -> str:
    return s.strip().casefold()


def audit_generator(names: list[str], keys: list[str]) -> dict:
    """Compara os nomes de UMA pasta com as chaves do CSV (lista completa, com repetições)."""
    name_set = set(names)
    stem_map: dict[str, list[str]] = collections.defaultdict(list)
    for n in names:
        stem_map[pathlib.PurePosixPath(n).stem].append(n)
    key_counts = collections.Counter(keys)
    distinct = sorted(key_counts)

    exact = sorted(k for k in distinct if k in name_set)  # chave == nome com extensão
    by_stem = sorted(k for k in distinct if k in stem_map)  # chave == nome sem extensão
    candidates = {k: set(stem_map.get(k, [])) | ({k} if k in name_set else set()) for k in distinct}
    unique_match = [k for k in distinct if len(candidates[k]) == 1]
    ambiguous = {k: sorted(c) for k, c in candidates.items() if len(c) > 1}
    keys_without_image = [k for k in distinct if not candidates[k]]
    used = set().union(*candidates.values()) if candidates else set()
    files_without_key = sorted(n for n in names if n not in used)

    # diferenças só de caixa/espaços entre chave sem imagem e arquivo sem chave (NÃO contam como correspondência)
    loose: dict[str, list[str]] = collections.defaultdict(list)
    for n in files_without_key:
        loose[_norm(pathlib.PurePosixPath(n).stem)].append(n)
    case_or_space_only = [
        {"key": k, "files": sorted(loose[_norm(k)])} for k in keys_without_image if _norm(k) in loose
    ]

    dup_names = sorted(n for n, c in collections.Counter(names).items() if c > 1)
    dup_stems = sorted(s for s, v in stem_map.items() if len(v) > 1)
    dup_keys = sorted(k for k, c in key_counts.items() if c > 1)
    perfect = (
        not keys_without_image and not files_without_key and not ambiguous
        and not dup_names and not dup_stems and not dup_keys
        and len(unique_match) == len(distinct) == len(names)
    )
    return {
        "n_files": len(names), "n_unique_names": len(name_set), "n_unique_stems": len(stem_map),
        "n_csv_rows": len(keys), "n_unique_keys": len(distinct),
        "exact_name_matches": len(exact), "stem_matches": len(by_stem), "keys_matched_exactly_once": len(unique_match),
        "keys_without_image": keys_without_image, "files_without_key": files_without_key,
        "keys_with_multiple_candidate_images": ambiguous, "duplicate_file_names": dup_names,
        "duplicate_stems": dup_stems, "duplicate_csv_keys": dup_keys,
        "case_or_space_only_differences": case_or_space_only, "perfect": perfect,
    }


def audit(files_by_gen: dict[str, list[str]], keys: list[str], expected_keys: int = 1000) -> dict:
    gens = {g: audit_generator(sorted(names), keys) for g, names in sorted(files_by_gen.items())}
    distinct = set(keys)
    by_key_stem: dict[str, set[str]] = collections.defaultdict(set)
    for k in distinct:
        by_key_stem[pathlib.PurePosixPath(k).stem].add(k)
    collisions = {s: sorted(v) for s, v in by_key_stem.items() if len(v) > 1}
    stem_sets = {frozenset(pathlib.PurePosixPath(n).stem for n in names) for names in files_by_gen.values()}
    whitespace_keys = sorted(k for k in distinct if k != k.strip())

    reasons = []
    if len(distinct) != expected_keys:
        reasons.append(f"chaves distintas = {len(distinct)} (esperado {expected_keys})")
    if not gens:
        reasons.append("nenhuma pasta de imagens encontrada")
    reasons += [f"{g}: pareamento não é 1 para 1" for g, r in gens.items() if not r["perfect"]]
    if collisions:
        reasons.append(f"{len(collisions)} colisões de chaves diferentes com o mesmo stem")
    if whitespace_keys:
        reasons.append(f"{len(whitespace_keys)} chaves com espaços nas pontas")
    summary = {
        "n_generators": len(gens), "n_image_files": sum(r["n_files"] for r in gens.values()),
        "n_csv_rows": len(keys), "n_distinct_keys": len(distinct), "expected_keys": expected_keys,
        "same_key_set_in_every_generator": bool(gens) and all(not r["keys_without_image"] for r in gens.values()),
        "same_stems_in_every_generator": len(stem_sets) == 1,
        "every_key_has_exactly_one_image_in_every_generator": bool(gens) and all(
            r["keys_matched_exactly_once"] == len(distinct) and not r["keys_with_multiple_candidate_images"]
            for r in gens.values()),
        "key_stem_collisions": collisions, "keys_with_edge_whitespace": whitespace_keys,
        "pairing_proven": not reasons, "reasons_not_proven": reasons,
    }
    return {"summary": summary, "generators": gens}


def to_markdown(rep: dict) -> str:
    s = rep["summary"]
    verdict = "CONFIRMADO" if s["pairing_proven"] else "NÃO CONFIRMADO"
    md = ["# Auditoria de pareamento do Synthbuster", "",
          f"**Pareamento completo: {verdict}**", ""]
    if not s["pairing_proven"]:
        md += ["Motivos:", *[f"- {r}" for r in s["reasons_not_proven"]], ""]
    md += [f"- Pastas: {s['n_generators']} | imagens: {s['n_image_files']} | linhas do CSV: {s['n_csv_rows']} | chaves distintas: {s['n_distinct_keys']} (esperado {s['expected_keys']})",
           f"- Mesmo conjunto de chaves em todos os geradores: {s['same_key_set_in_every_generator']}",
           f"- Mesmos nomes (stems) em todas as pastas: {s['same_stems_in_every_generator']}",
           f"- Cada chave tem exatamente 1 imagem em cada pasta: {s['every_key_has_exactly_one_image_in_every_generator']}", "",
           "| pasta | arquivos | nomes únicos | chaves casadas 1x | exatas (c/ ext.) | por stem | chave sem imagem | arquivo sem chave | ambíguas | dup. nomes | dup. stems | só caixa/espaço |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, r in rep["generators"].items():
        md.append(f"| {g} | {r['n_files']} | {r['n_unique_names']} | {r['keys_matched_exactly_once']} | {r['exact_name_matches']} | {r['stem_matches']} | "
                  f"{len(r['keys_without_image'])} | {len(r['files_without_key'])} | {len(r['keys_with_multiple_candidate_images'])} | "
                  f"{len(r['duplicate_file_names'])} | {len(r['duplicate_stems'])} | {len(r['case_or_space_only_differences'])} |")
    problems = []
    for g, r in rep["generators"].items():
        for label, val in (("chaves sem imagem", r["keys_without_image"]), ("arquivos sem chave", r["files_without_key"]),
                           ("chaves ambíguas", list(r["keys_with_multiple_candidate_images"])), ("nomes duplicados", r["duplicate_file_names"]),
                           ("stems duplicados", r["duplicate_stems"]), ("chaves duplicadas no CSV", r["duplicate_csv_keys"]),
                           ("só caixa/espaço", [d["key"] for d in r["case_or_space_only_differences"]])):
            if val:
                problems.append(f"- {g}, {label} ({len(val)}): {val[:EXAMPLES]}")
    if s["key_stem_collisions"]:
        problems.append(f"- colisões de chaves com o mesmo stem: {dict(list(s['key_stem_collisions'].items())[:EXAMPLES])}")
    if s["keys_with_edge_whitespace"]:
        problems.append(f"- chaves com espaços nas pontas ({len(s['keys_with_edge_whitespace'])}): {s['keys_with_edge_whitespace'][:EXAMPLES]}")
    md += ["", "## Exemplos de divergências", "", *(problems or ["Nenhuma divergência."])]
    if "transfer" in rep:
        t = rep["transfer"]
        md += ["", f"Transferido: {t['bytes_fetched'] / 1e6:.1f} MB em {t['http_requests']} requisições, {t['retried']} retentativa(s) (ZIP total {t['zip_total_bytes'] / 1e9:.2f} GB; valor mínimo)."]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=pathlib.Path, help="ZIP local (em vez de ler remoto por Range)")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("dataset/inspection"))
    ap.add_argument("--key-column", default=KEY_COLUMN)
    ap.add_argument("--expected-keys", type=int, default=1000)
    args = ap.parse_args(argv)

    zf, remote = (zipfile.ZipFile(args.zip), None) if args.zip else open_remote_zip(args.url)
    infos = [i for i in zf.infolist() if not i.is_dir()]
    files_by_gen: dict[str, list[str]] = collections.defaultdict(list)
    for i in infos:
        p = pathlib.PurePosixPath(i.filename)
        if p.suffix.lower() in IMG_EXT:
            files_by_gen[str(p.parent)].append(p.name)
    csvs = sorted((i.filename for i in infos if pathlib.PurePosixPath(i.filename).name.lower() == "prompts.csv"), key=len)
    if not csvs:
        sys.exit("prompts.csv não encontrado no ZIP")
    with zf.open(csvs[0]) as fh:
        import io
        rows = list(csv.reader(io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")))
    header, body = rows[0], rows[1:]
    if args.key_column not in header:
        sys.exit(f"coluna {args.key_column!r} não existe em {csvs[0]}; colunas disponíveis: {header}")
    ci = header.index(args.key_column)
    keys = [r[ci] for r in body if ci < len(r)]  # só a coluna-chave é lida; nenhuma outra coluna entra no relatório

    rep = audit(files_by_gen, keys, args.expected_keys)
    rep["summary"]["n_csv_short_rows"] = len(body) - len(keys)
    rep["source"] = {"zip": str(args.zip) if args.zip else args.url, "csv": csvs[0], "key_column": args.key_column,
                     "run_at": datetime.datetime.now().astimezone().isoformat()}
    if remote:
        rep["transfer"] = {"zip_total_bytes": remote.size, "bytes_fetched": remote.bytes_fetched,
                           "http_requests": remote.requests, "retried": remote.retried}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "synthbuster_pairing_audit.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    md = to_markdown(rep)
    (args.out / "synthbuster_pairing_audit.md").write_text(md)
    print(md)
    return rep


if __name__ == "__main__":
    main()
