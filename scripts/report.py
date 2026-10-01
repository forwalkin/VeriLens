#!/usr/bin/env python3
"""Gera results/<run>/report.md (+ figura) a partir de raw.csv.

Critério operacional INTERNO do projeto: limite inferior do IC95% do AUC > 0,60 (não é um limite
científico universal). Um AUC global alto não basta: o detector precisa também se sustentar por
gerador, sob edição e sem depender de confundidores. A decisão final é de engenharia; a tabela
"Decisão de port para Android" é apenas a aplicação mecânica da regra em evalkit/decision.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalkit import analysis as an  # noqa: E402
from evalkit.analysis import NAMES, scores  # noqa: E402
from evalkit.decision import CRITERION_LO, MIN_N, MIN_N_EDIT, AucResult, decide  # noqa: E402
from evalkit.metrics import confusion, prf, wilson_ci  # noqa: E402

THRESHOLD = 0.5
GROUPS = ("real", "edited", "ai", "ai_edited")


def f(x: float, d: int = 3) -> str:
    return "n/d" if x != x else f"{x:.{d}f}"


def ci(r: AucResult) -> str:
    return f"{f(r.auc, 3)} [{f(r.lo, 2)}, {f(r.hi, 2)}]"


def mark(r: AucResult, min_n: int = MIN_N) -> str:
    s = r.status(min_n)
    return {"ok": "ok", "fraco": "fraco", "amostra pequena": "amostra pequena (não julgado)",
            "acaso": "**PRÓXIMO DO ACASO**", "invertido": "**INVERTIDO**"}[s]


def section_global(rows, dets, n_boot) -> list[str]:
    out = ["## 1. AUC global REAL vs IA", "",
           f"Critério interno: limite inferior do IC95% > {CRITERION_LO:.2f} (operacional, não universal).", "",
           "| detector | n REAL | n IA | AUC [IC95%] | critério interno |", "|---|---|---|---|---|"]
    for d in dets:
        r = an.global_auc(rows, d, n_boot)
        out.append(f"| {NAMES[d]} | {r.n_neg} | {r.n_pos} | {ci(r)} | {mark(r)} |")
    return out + [""]


def section_generators(rows, dets, n_boot) -> list[str]:
    out = ["## 2. AUC por gerador (REAL vs cada gerador)", "",
           f"As reais são as mesmas em todos os geradores. Geradores com n < {MIN_N} não são julgados. "
           "IC por bootstrap estratificado (1000 reamostragens).", ""]
    for d in dets:
        res = an.per_generator(rows, d, n_boot)
        out += [f"### {NAMES[d]}", "", "| gerador | n IA | n REAL | AUC [IC95%] | critério interno |", "|---|---|---|---|---|"]
        out += [f"| {g or '-'} | {r.n_pos} | {r.n_neg} | {ci(r)} | {mark(r)} |" for g, r in res.items()]
        out.append("")
    return out


def section_paired(rows, dets, n_boot) -> list[str]:
    if not any(r["pair_key"] for r in rows):
        return ["## 3. Pareamento Synthbuster ↔ RAISE-1k", "", "Sem `pair_key` no dataset: análise pareada não executada.", ""]
    out = ["## 3. Pareamento Synthbuster ↔ RAISE-1k", "",
           "Taxa de vitória = P(score da IA > score da real do mesmo par). 0,5 = o detector não ordena o par; "
           "controla o conteúdo da cena. O Synthbuster foi gerado a partir de *descrições* do RAISE, então o par "
           "compartilha tema, não é a mesma imagem.", ""]
    for d in dets:
        res = an.paired(rows, d, n_boot)
        out += [f"### {NAMES[d]}", "", "| gerador | n pares | taxa de vitória [IC95%] |", "|---|---|---|"]
        out += [f"| {g} | {n} | {f(m, 3)} [{f(lo, 2)}, {f(hi, 2)}] |" for g, (n, m, lo, hi) in res.items()]
        out.append("")
    return out


def section_confounders(rows, dets, n_boot) -> list[str]:
    out = ["## 4. Confundidores", ""]
    if "format" not in rows[0]:
        return out + ["Colunas de metadados ausentes em raw.csv.", ""]
    conf = an.metadata_confounders(rows)
    out += ["### 4a. O que metadados simples já separam (REAL vs IA)", "",
            "| atributo | média REAL | média IA | AUC univariado | alerta (≥0,8 ou ≤0,2) |", "|---|---|---|---|---|"]
    out += [f"| {n} | {a:.3g} | {b:.3g} | {f(c)} | {'**CONFUNDIDOR**' if n in an.flagged_attributes(conf) else ''} |"
            for n, a, b, c in conf]
    out += ["", f"### 4b. AUC dos detectores dentro de estratos (n ≥ {MIN_N} por classe para ser julgado)", "",
            "Se o AUC cai ou some dentro de um formato/resolução, o resultado global pode vir do confundidor.", ""]
    for d in dets:
        res = an.per_stratum(rows, d, n_boot)
        out += [f"**{NAMES[d]}**", "", "| estrato | n IA | n REAL | AUC [IC95%] | situação |", "|---|---|---|---|---|"]
        out += [f"| {k} | {r.n_pos} | {r.n_neg} | {ci(r)} | {mark(r)} |" for k, r in res.items()]
        out.append("")
    return out


def section_robustness(rows, dets, n_boot) -> list[str]:
    n_e = sum(r["label"] == "edited" for r in rows)
    n_a = sum(r["label"] == "ai_edited" for r in rows)
    out = ["## 5. Robustez após compressão/edição", ""]
    if not n_e:
        return out + ["Sem imagens editadas no dataset.", ""]
    out += [f"Real editada: n={n_e}; IA editada: n={n_a}. "
            "AUC = real editada vs IA editada com a MESMA edição. Taxas a limiar 0.5. "
            f"Por edição os ICs são largos (n≈por-edição); só julgamos com n ≥ {MIN_N_EDIT}.", ""]
    for d in dets:
        grp = an.per_edit_group(rows, d, n_boot)
        out += [f"### {NAMES[d]}", "", "| grupo de edição | n IA ed. | n REAL ed. | AUC [IC95%, bootstrap por imagem-pai] | situação |", "|---|---|---|---|---|"]
        out += [f"| {k} | {r.n_pos} | {r.n_neg} | {ci(r)} | {mark(r, MIN_N_EDIT)} |" for k, r in grp.items()]
        out += ["", "| edição | n IA | n REAL | AUC [IC95%] | recall IA | falso positivo REAL |", "|---|---|---|---|---|---|"]
        for e, v in an.per_edit(rows, d, n_boot).items():
            r = v["auc"]
            out.append(f"| {e} | {r.n_pos} | {r.n_neg} | {ci(r)} | {f(v['recall'], 2)} | {f(v['fpr'], 2)} |")
        out.append("")
    return out


def threshold_tables(rows, dets) -> list[str]:
    out = [f"## 6. Métricas no limiar original ({THRESHOLD})", "",
           "Positivo = IA. REAL e EDITADA = negativos; IA editada fica fora desta tabela (ver seção 5).", ""]
    sub = [r for r in rows if r["label"] in ("real", "ai")]
    y = np.array([r["label"] == "ai" for r in sub])
    out += [f"REAL n={int((~y).sum())}, IA n={int(y.sum())}", "", "```",
            f"{'Detector':<16}{'Accuracy':>9}{'Precision':>10}{'Recall':>8}{'F1':>7}{'FPR':>7}", "-" * 57]
    cms = []
    for d in dets:
        c = confusion(y, scores(sub, d) >= THRESHOLD)
        m = prf(c)
        cms.append((d, c))
        out.append(f"{NAMES[d]:<16}{f(m['accuracy']):>9}{f(m['precision']):>10}{f(m['recall']):>8}{f(m['f1']):>7}{f(m['fpr']):>7}")
    out += ["```", "", "Matrizes de confusão (linhas = verdade):", ""]
    out += [f"- **{NAMES[d]}** — REAL→REAL: {c.tn}, REAL→IA: {c.fp} | IA→REAL: {c.fn}, IA→IA: {c.tp}" for d, c in cms]
    out += ["", "### Taxa ≥ 0.5 e distribuição dos scores por grupo", "",
            "| grupo | detector | n | taxa ≥0.5 [IC95%] | média | p10 | mediana | p90 |", "|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        gs = [r for r in rows if r["label"] == g]
        for d in dets if gs else []:
            s = scores(gs, d)
            k = int((s >= THRESHOLD).sum())
            lo, hi = wilson_ci(k, len(s))
            q = np.quantile(s, [0.1, 0.5, 0.9])
            out.append(f"| {g} | {NAMES[d]} | {len(s)} | {k / len(s):.2f} [{lo:.2f},{hi:.2f}] | {s.mean():.3f} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} |")
    return out + [""]


def histograms(rows, dets, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(len(dets), len(GROUPS), figsize=(13, 2.6 * len(dets)), squeeze=False)
    colors = {"real": "#2a9d8f", "edited": "#8ab17d", "ai": "#e76f51", "ai_edited": "#f4a261"}
    for i, d in enumerate(dets):
        for j, g in enumerate(GROUPS):
            s = [float(r[f"{d}_score"]) for r in rows if r["label"] == g]
            ax = axes[i][j]
            ax.hist(s, bins=np.linspace(0, 1, 21), color=colors[g])
            ax.axvline(THRESHOLD, color="k", ls="--", lw=0.8)
            ax.set_title(f"{NAMES[d]} — {g} (n={len(s)})", fontsize=8)
            ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def timing(rows, dets, meta) -> list[str]:
    out = ["## 7. Tempo de inferência (CPU, por imagem)", "", "| detector | média ms | mediana | p95 |", "|---|---|---|---|"]
    for d in [x for x in dets if f"{x}_ms" in rows[0]]:
        t = np.array([float(r[f"{d}_ms"]) for r in rows])
        out.append(f"| {NAMES[d]} | {t.mean():.1f} | {np.median(t):.1f} | {np.quantile(t, 0.95):.1f} |")
    return out + ["", "Carga dos modelos (s): `" + json.dumps({k: round(v, 2) for k, v in meta.get("load_seconds", {}).items()}) + "`", ""]


def section_decision(rows, dets, n_boot) -> list[str]:
    out = ["## 8. Decisão de port para Android", "",
           "**Decisão PRELIMINAR de engenharia**, aplicação mecânica da regra em `evalkit/decision.py` "
           f"(limite inferior do IC95% do AUC > {CRITERION_LO:.2f} no global, em cada gerador com n ≥ {MIN_N}, "
           "nos grupos de edição, e sem sinal de confundidor). Não é afirmação sobre a capacidade universal do "
           "detector; o limiar 0,60 é critério operacional nosso. Nenhum detector é descartado por um único número.", "",
           f"| detector | AUC global [IC95%] | pior gerador (n≥{MIN_N}) | confundidores | robustez (pior grupo) | decisão preliminar |",
           "|---|---|---|---|---|---|"]
    details = []
    for d in dets:
        ev = an.build_evidence(rows, d, n_boot)
        dec = decide(ev)
        main = {k: r for k, r in ev.generators.items() if r.n_pos >= MIN_N}
        worst = min(main.items(), key=lambda kv: kv[1].lo, default=None)
        wtxt = f"{worst[0]}: {ci(worst[1])}" if worst else "n/d"
        rob = min(ev.robustness.items(), key=lambda kv: kv[1].lo, default=None)
        rtxt = f"{rob[0]}: {ci(rob[1])}" if rob else "n/d"
        ctxt = ", ".join(ev.dataset_confounders) or "nenhum no dataset"
        bad = [k for k, r in ev.strata.items() if r.status() in ("acaso", "invertido") or ev.global_.auc - r.auc > 0.15]
        if bad:
            ctxt += f"; suspeita nos estratos: {', '.join(bad)}"
        out.append(f"| {NAMES[d]}{' (derivado dos 3 anteriores)' if d == 'ensemble' else ''} | {ci(ev.global_)} | {wtxt} | {ctxt} | {rtxt} | **{dec.label}** |")
        details.append((d, ev, dec))
    out.append("")
    for d, ev, dec in details:
        out += [f"### {NAMES[d]} — {dec.label}", ""]
        out += [f"- {g}" for g in dec.gaps] or ["- sem lacunas segundo a regra"]
        out += [f"- _nota:_ {n}" for n in dec.notes]
        out.append("")
    out += ["Antes de decidir, leia as seções 2 a 5: um detector marcado para manter ainda pode falhar fora desta amostra; "
            "um marcado para descartar/recalibrar pode ter falhado por viés do dataset (ver 4a) ou por n pequeno."]
    return out + [""]


def main(argv: list[str] | None = None) -> pathlib.Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=pathlib.Path, help="results/<run_id>")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    rows_all, rows, header = an.load_raw(args.run)
    meta = json.loads((args.run / "meta.json").read_text())
    dets = an.detectors_in(header)
    counts = {g: sum(r["label"] == g for r in rows) for g in GROUPS}

    md = [f"# Benchmark — {meta['kind']} — run {meta['run_id']}", ""]
    if meta["kind"] != "original-benchmark":
        md += ["> **EXECUÇÃO PARCIAL/SMOKE: não é o benchmark do projeto original; não interpretar como resultado.**", ""]
    md += [f"- base `{meta['base_commit']}` (dirty={meta['base_dirty']}), manifesto `{meta['manifest_digest']}`",
           f"- imagens avaliadas: {counts} | com erro: {len(rows_all) - len(rows)} de {len(rows_all)}",
           f"- ambiente: `{json.dumps(meta['env'])}`", "",
           "> Estimativa estatística, não prova. Positivo = IA. Seção 8 resume a decisão preliminar de port.", ""]
    md += section_global(rows, dets, args.n_boot)
    md += section_generators(rows, dets, args.n_boot)
    md += section_paired(rows, dets, args.n_boot)
    md += section_confounders(rows, dets, args.n_boot)
    md += section_robustness(rows, dets, args.n_boot)
    md += threshold_tables(rows, dets)
    histograms(rows, dets, args.run / "score_distributions.png")
    md += ["![scores](score_distributions.png)", ""]
    md += timing(rows, dets, meta)
    md += section_decision(rows, dets, args.n_boot)
    text = "\n".join(md)
    (args.run / "report.md").write_text(text)
    print(text)
    return args.run / "report.md"


if __name__ == "__main__":
    main()
