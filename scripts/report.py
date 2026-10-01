#!/usr/bin/env python3
"""Gera o relatório do benchmark (results/<run>/report.md + figuras) a partir de raw.csv.

Limiar de decisão = 0.5 (o do projeto original). O AUC (independente de limiar) responde
se o detector separa real de IA; o IC do AUC é bootstrap estratificado.
Edições não entram no IC do AUC principal (pais repetidos => pseudo-replicação).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalkit.metrics import auc, bootstrap_auc_ci, chance_verdict, confusion, prf, wilson_ci  # noqa: E402

THRESHOLD = 0.5
NAMES = {"frequency": "Frequency", "clip_drift": "CLIP", "hf_classifier": "SDXL Detector", "ensemble": "Ensemble"}
GROUPS = ("real", "edited", "ai")


def load(run: pathlib.Path):
    with open(run / "raw.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if not r["error"]]
    return rows, ok


def col(rows, key):
    return np.array([float(r[key]) for r in rows])


def fmt(x: float, d: int = 3) -> str:
    return "n/d" if x != x else f"{x:.{d}f}"


def metrics_table(rows, dets, title, labels) -> list[str]:
    sub = [r for r in rows if r["label"] in labels]
    y = np.array([r["label"] == "ai" for r in sub])
    out = [f"### {title}", "", f"n negativos = {int((~y).sum())}, n IA = {int(y.sum())}", "", "```",
           f"{'Detector':<16}{'Accuracy':>9}{'Precision':>10}{'Recall':>8}{'F1':>7}{'FPR':>7}{'AUC [IC95%]':>24}  Veredito (AUC)",
           "-" * 112]
    cms = []
    for d in dets:
        s = col(sub, f"{d}_score")
        c = confusion(y, s >= THRESHOLD)
        m = prf(c)
        a = auc(y, s)
        ci = bootstrap_auc_ci(y, s) if title.startswith("A") else (float("nan"),) * 2
        verdict = chance_verdict(a, ci) if ci[0] == ci[0] else "(IC não calculado)"
        out.append(f"{NAMES[d]:<16}{fmt(m['accuracy']):>9}{fmt(m['precision']):>10}{fmt(m['recall']):>8}"
                   f"{fmt(m['f1']):>7}{fmt(m['fpr']):>7}{fmt(a) + ' [' + fmt(ci[0], 2) + ',' + fmt(ci[1], 2) + ']':>24}  {verdict}")
        cms.append((d, c))
    out += ["```", "", "Matrizes de confusão (linhas = verdade; colunas = previsto):", ""]
    for d, c in cms:
        out += [f"- **{NAMES[d]}** — Real/Edit→Real: {c.tn}, Real/Edit→IA: {c.fp} | IA→Real: {c.fn}, IA→IA: {c.tp}"]
    return out + [""]


def group_tables(rows, dets) -> list[str]:
    out = ["## Resultados por grupo (limiar 0.5)", ""]
    for key, label, rate_name in (("real", "REAL", "falso positivo"), ("edited", "EDITADA", "falso positivo"), ("ai", "IA", "detecção (recall)")):
        sub = [r for r in rows if r["label"] == key]
        if not sub:
            continue
        out += [f"### {label} (n={len(sub)}) — taxa de {rate_name} e distribuição dos scores", "",
                "| detector | taxa ≥0.5 [IC95%] | média | p10 | mediana | p90 |", "|---|---|---|---|---|---|"]
        for d in dets:
            s = col(sub, f"{d}_score")
            k = int((s >= THRESHOLD).sum())
            lo, hi = wilson_ci(k, len(s))
            q = np.quantile(s, [0.1, 0.5, 0.9])
            out.append(f"| {NAMES[d]} | {k / len(s):.2f} [{lo:.2f},{hi:.2f}] | {s.mean():.3f} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} |")
        out.append("")
    # por gerador (IA) e por edição
    for key, field, title in (("ai", "generator", "IA por gerador (recall)"), ("edited", "edit", "Editadas por edição (taxa de falso positivo)")):
        sub = [r for r in rows if r["label"] == key]
        if not sub:
            continue
        names = sorted({r[field] for r in sub})
        out += [f"### {title}", "", "| " + field + " | n | " + " | ".join(NAMES[d] for d in dets) + " |",
                "|---|---|" + "---|" * len(dets)]
        for g in names:
            gs = [r for r in sub if r[field] == g]
            cells = []
            for d in dets:
                k = int((col(gs, f"{d}_score") >= THRESHOLD).sum())
                lo, hi = wilson_ci(k, len(gs))
                cells.append(f"{k / len(gs):.2f} [{lo:.2f},{hi:.2f}]")
            out.append(f"| {g or '-'} | {len(gs)} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def histograms(rows, dets, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(len(dets), 3, figsize=(10, 2.6 * len(dets)), squeeze=False)
    colors = {"real": "#2a9d8f", "edited": "#e9c46a", "ai": "#e76f51"}
    for i, d in enumerate(dets):
        for j, g in enumerate(GROUPS):
            ax = axes[i][j]
            s = [float(r[f"{d}_score"]) for r in rows if r["label"] == g]
            ax.hist(s, bins=np.linspace(0, 1, 21), color=colors[g])
            ax.axvline(THRESHOLD, color="k", ls="--", lw=0.8)
            ax.set_title(f"{NAMES[d]} — {g} (n={len(s)})", fontsize=9)
            ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def timing(rows, dets, meta) -> list[str]:
    out = ["## Tempo de inferência (CPU, por imagem)", "", "| detector | média ms | mediana | p95 |", "|---|---|---|---|"]
    for d in dets:
        t = col(rows, f"{d}_ms")
        out.append(f"| {NAMES[d]} | {t.mean():.1f} | {np.median(t):.1f} | {np.quantile(t, 0.95):.1f} |")
    out += ["", "Tempo de carga dos modelos (s): `" + json.dumps({k: round(v, 2) for k, v in meta.get("load_seconds", {}).items()}) + "`", ""]
    return out


def main(argv: list[str] | None = None) -> pathlib.Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=pathlib.Path, help="results/<run_id>")
    args = ap.parse_args(argv)

    rows_all, rows = load(args.run)
    meta = json.loads((args.run / "meta.json").read_text())
    header = rows_all[0].keys() if rows_all else []
    dets = [d for d in ("frequency", "clip_drift", "hf_classifier", "ensemble") if f"{d}_score" in header]

    counts = {g: sum(r["label"] == g for r in rows) for g in GROUPS}
    md = [f"# Benchmark — {meta['kind']} — run {meta['run_id']}", ""]
    if meta["kind"] != "original-benchmark":
        md += ["> **EXECUÇÃO PARCIAL/SMOKE: não é o benchmark do projeto original; não interpretar como resultado.**", ""]
    md += [f"- base `{meta['base_commit']}` (dirty={meta['base_dirty']}), manifesto `{meta['manifest_digest']}`",
           f"- imagens avaliadas: {counts} | com erro: {len(rows_all) - len(rows)} de {len(rows_all)}",
           f"- ambiente: `{json.dumps(meta['env'])}`", "",
           "> Estimativa estatística. Convenção: positivo = IA; REAL e EDITADA (real editada) = negativos.", ""]
    md += ["## Resumo de métricas", ""]
    md += metrics_table(rows, dets, "A. REAL vs IA (comparação principal; IC do AUC por bootstrap)", ("real", "ai"))
    if counts["edited"]:
        md += metrics_table(rows, dets, "B. REAL + EDITADA vs IA (negativos agrupados; sem IC por pseudo-replicação)", ("real", "edited", "ai"))
    md += group_tables(rows, dets)
    histograms(rows, dets, args.run / "score_distributions.png")
    md += ["## Distribuição dos scores", "", "![scores](score_distributions.png)", ""]
    md += timing(rows, dets, meta)
    text = "\n".join(md)
    (args.run / "report.md").write_text(text)
    print(text)
    return args.run / "report.md"


if __name__ == "__main__":
    main()
