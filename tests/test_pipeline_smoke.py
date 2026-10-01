"""Smoke test do encanamento: manifesto -> run_benchmark (só Frequency) -> report.

Usa imagens SINTÉTICAS. Valida apenas que o pipeline roda e produz as saídas;
NÃO diz nada sobre a qualidade dos detectores.
"""

import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("ai_image_scanner")

from evalkit.manifest import Row, make_id, write_manifest  # noqa: E402


def _noise(seed, size=300):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def test_pipeline_runs_end_to_end(tmp_path):
    from scripts import make_edited, report, run_benchmark  # noqa: F401

    ds = tmp_path / "dataset"
    rows = []
    for label, gen in (("real", "cam"), ("ai", "genA")):
        (ds / label).mkdir(parents=True)
        for i in range(6):
            p = ds / label / f"{label}{i}.png"
            _noise(hash((label, i)) % 1000).save(p)
            rid = make_id(p)
            q = ds / label / f"{rid}.png"
            p.rename(q)
            rows.append(Row(id=rid, path=str(q.relative_to(ds)), label=label, source="smoke", generator=gen))
    write_manifest(ds / "manifest.csv", rows)

    make_edited.main(["--dataset", str(ds), "--per-edit", "2"])
    out = run_benchmark.main(["--dataset", str(ds), "--out", str(tmp_path / "res"),
                              "--detectors", "frequency", "--run-id", "smoke", "--allow-dirty"])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["kind"] == "PARTIAL-SMOKE"
    md = report.main([str(out)]).read_text()
    assert "EXECUÇÃO PARCIAL" in md and "Frequency" in md
    assert (out / "score_distributions.png").exists()
