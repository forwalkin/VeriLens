"""Manifesto do dataset: uma linha por imagem, fonte única da verdade para a avaliação."""

from __future__ import annotations

import csv
import hashlib
import pathlib
from dataclasses import asdict, dataclass, fields

LABELS = ("real", "ai", "edited")
FIELDS = [
    "id", "path", "label", "source", "generator", "license",
    "parent_id", "edit", "notes",
]


@dataclass
class Row:
    id: str
    path: str  # relativo à raiz do dataset
    label: str  # real | ai | edited
    source: str  # nome da fonte em sources.toml
    generator: str = ""  # IA: modelo gerador; real: origem/câmera quando conhecida
    license: str = ""
    parent_id: str = ""  # edited: id da imagem real de origem
    edit: str = ""  # edited: nome da edição
    notes: str = ""

    @property
    def is_ai(self) -> bool:
        """Verdade-terreno binária. 'edited' = real editada, portanto negativa."""
        return self.label == "ai"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_id(path: pathlib.Path) -> str:
    return sha256_file(path)[:16]


def read_manifest(path: pathlib.Path) -> list[Row]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [Row(**{k: r.get(k, "") for k in FIELDS}) for r in csv.DictReader(f)]
    for r in rows:
        if r.label not in LABELS:
            raise ValueError(f"label inválido {r.label!r} em {r.path}")
    return rows


def write_manifest(path: pathlib.Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r.label, r.id)):
            w.writerow(asdict(r))


def manifest_digest(path: pathlib.Path) -> str:
    return sha256_file(path)[:12]


assert [f.name for f in fields(Row)] == FIELDS
