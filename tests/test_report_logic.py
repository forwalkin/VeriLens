"""Verifica a lógica do relatório com scores FABRICADOS (só teste; não são resultados de detector).

Cenário: detector bom no gerador A, no acaso no gerador B, e dataset com confundidor de formato.
O relatório deve (1) destacar B como próximo do acaso, (2) NÃO aprovar o detector só pelo AUC global,
(3) sinalizar o confundidor e (4) incluir a seção de pareamento e a de decisão.
"""

import csv
import json

import numpy as np

from scripts import report

COLS = ["id", "label", "generator", "edit", "source", "parent_id", "pair_key", "width", "height", "format",
        "file_bytes", "has_exif", "error", "frequency_score", "frequency_ms"]


def _row(i, label, gen, score, fmt, pair="", edit="", parent=""):
    return {"id": f"i{i}", "label": label, "generator": gen, "edit": edit, "source": "t", "parent_id": parent,
            "pair_key": pair, "width": 1024, "height": 1024, "format": fmt, "file_bytes": 300000,
            "has_exif": 1 if fmt == "JPEG" else 0, "error": "", "frequency_score": score, "frequency_ms": 5.0}


def _make_run(tmp_path):
    rng = np.random.default_rng(0)
    rows, i = [], 0
    for k in range(80):
        rows.append(_row(i, "real", "cam", float(np.clip(rng.normal(0.3, 0.1), 0, 1)), "JPEG", pair=f"k{k}")); i += 1
    for gen, mu in (("genA", 0.75), ("genB", 0.3)):
        for k in range(40):
            rows.append(_row(i, "ai", gen, float(np.clip(rng.normal(mu, 0.1), 0, 1)), "PNG", pair=f"k{k}")); i += 1
    for k in range(25):  # edições: comprimir quebra o detector (AUC ~ acaso)
        rows.append(_row(i, "edited", "cam", float(rng.uniform()), "JPEG", edit="jpeg_q30", parent=f"r{k}")); i += 1
        rows.append(_row(i, "ai_edited", "genA", float(rng.uniform()), "JPEG", edit="jpeg_q30", parent=f"a{k}")); i += 1
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    (run / "meta.json").write_text(json.dumps({"run_id": "t", "kind": "PARTIAL-SMOKE", "base_commit": "4ec7edc",
                                               "base_dirty": False, "manifest_digest": "x", "env": {}, "load_seconds": {},
                                               "pairing": {"requested_pairs": 300, "common_keys_available": 80, "pairs_used": 80,
                                                           "generators": ["genA", "genB"],
                                                           "keys_available_per_generator": {"s/genA": 80, "s/genB": 80}}}))
    return run


def test_report_flags_chance_generator_confounder_and_does_not_approve(tmp_path):
    run = _make_run(tmp_path)
    text = report.main([str(run), "--n-boot", "200"]).read_text()
    genb = next(l for l in text.splitlines() if l.startswith("| genB |") and "n REAL" not in l)
    assert "PRÓXIMO DO ACASO" in genb
    gena = next(l for l in text.splitlines() if l.startswith("| genA |") and "ok" in l)
    assert gena
    assert "**CONFUNDIDOR**" in text                      # formato separa real (JPEG) de IA (PNG)
    assert "## 3. Pareamento por conteúdo/descrição" in text and "taxa de vitória" in text
    assert "não é a mesma fotografia" in text
    assert "pares solicitados (máximo desejado): 300" in text and "pares usados: 80" in text
    assert "ATENÇÃO" in text.split("## 3.")[1].split("## 4.")[0]  # usados < solicitados é informado
    assert "## 8. Decisão de port para Android" in text
    dec_line = next(l for l in text.splitlines() if l.startswith("| Frequency") and "preliminar" not in l and "**" in l)
    assert "**manter para investigação**" not in dec_line  # AUC global alto não basta
    assert "genB" in text.split("## 8.")[1]               # a lacuna é citada na decisão
    assert "compressao" in text                           # robustez reportada por grupo
