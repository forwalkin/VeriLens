"""Pareamento e geração de editadas com imagens sintéticas (valida a lógica, não os detectores)."""

import csv

import numpy as np
from PIL import Image

from evalkit.manifest import read_manifest
from scripts import build_dataset, make_edited


def _png(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(path)


def _setup(tmp_path, map_names=False):
    keys = [f"k{i:03d}" for i in range(12)]
    for i, k in enumerate(keys):
        _png(tmp_path / "raise" / f"{k}.png", i)
    for g, off in (("sd14", 100), ("dalle3", 200)):
        for i, k in enumerate(keys[:10]):  # só 10 dos 12 existem nos geradores
            name = f"ai_{k}" if map_names else k
            _png(tmp_path / "synth" / g / f"{name}.png", off + i)
    toml = f"""
[pairing]
real_source = "raise"
ai_sources = ["synth"]
n_pairs = 6
{'map_csv = "' + str(tmp_path / 'map.csv') + '"' if map_names else ''}

[[source]]
name = "raise"
label = "real"
kind = "local_dir"
path = "{tmp_path / 'raise'}"
n = 0
generator = "camera"

[[source]]
name = "synth"
label = "ai"
kind = "local_dir"
path = "{tmp_path / 'synth'}"
generator_from = "parent_dir"
n = 0
"""
    if map_names:
        with open(tmp_path / "map.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ai_key", "real_key"])
            w.writerows([(f"ai_{k}", k) for k in keys])
    (tmp_path / "sources.toml").write_text(toml)


def _run(tmp_path):
    build_dataset.main(["--sources", str(tmp_path / "sources.toml"), "--out", str(tmp_path / "ds")])
    return read_manifest(tmp_path / "ds" / "manifest.csv")


def test_pairing_uses_same_keys_across_generators(tmp_path):
    _setup(tmp_path)
    rows = _run(tmp_path)
    real = {r.pair_key for r in rows if r.label == "real"}
    assert len(real) == 6
    for gen in ("sd14", "dalle3"):
        assert {r.pair_key for r in rows if r.label == "ai" and r.generator == gen} == real
    assert all(k in {f"k{i:03d}" for i in range(10)} for k in real)  # só chaves comuns a todos


def test_pairing_with_map_csv(tmp_path):
    _setup(tmp_path, map_names=True)
    rows = _run(tmp_path)
    real = {r.pair_key for r in rows if r.label == "real"}
    assert real and {r.pair_key for r in rows if r.label == "ai" and r.generator == "sd14"} == real


def test_make_edited_creates_real_and_ai_edited(tmp_path):
    _setup(tmp_path)
    _run(tmp_path)
    make_edited.main(["--dataset", str(tmp_path / "ds"), "--per-edit", "4"])
    rows = read_manifest(tmp_path / "ds" / "manifest.csv")
    ed = [r for r in rows if r.label == "edited"]
    ae = [r for r in rows if r.label == "ai_edited"]
    assert ed and ae and all(r.parent_id for r in ed + ae)
    assert all(r.is_ai for r in ae) and not any(r.is_ai for r in ed)
    ai_parents = {r.id: r for r in rows if r.label == "ai"}
    assert {ai_parents[r.parent_id].generator for r in ae} == {"sd14", "dalle3"}  # rodízio entre geradores
    # idempotente: rodar de novo não duplica
    make_edited.main(["--dataset", str(tmp_path / "ds"), "--per-edit", "4"])
    assert len(read_manifest(tmp_path / "ds" / "manifest.csv")) == len(rows)
