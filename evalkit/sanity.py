"""Checagens de sanidade do raw.csv de um benchmark (pipeline correto?), independentes de qualidade do detector."""

from __future__ import annotations

import math

TOL = 1e-6


def check(rows: list[dict], meta: dict) -> list[str]:
    """Retorna a lista de problemas (vazia = pipeline coerente)."""
    problems: list[str] = []
    errs = [r for r in rows if r["error"]]
    if errs:
        problems.append(f"{len(errs)} imagens com erro (ex.: {errs[0]['id']}: {errs[0]['error']})")
    ok = [r for r in rows if not r["error"]]
    dets = [d for d in ("frequency", "clip_drift", "hf_classifier", "ensemble") if ok and f"{d}_score" in ok[0]]
    for d in dets:
        vals = [float(r[f"{d}_score"]) for r in ok]
        if any(math.isnan(v) or not (0.0 <= v <= 1.0) for v in vals):
            problems.append(f"{d}: scores fora de [0,1] ou NaN")
        if len(set(round(v, 6) for v in vals)) == 1 and len(vals) > 5:
            problems.append(f"{d}: score constante em todas as imagens ({vals[0]:.4f}) — detector provavelmente inoperante")
    w = meta.get("weights")
    if w and "ensemble" in dets:
        for r in ok:
            exp = max(0.0, min(1.0, sum(w[d] * float(r[f"{d}_score"]) for d in w)))
            if abs(exp - float(r["ensemble_score"])) > TOL:
                problems.append(f"ensemble inconsistente com média ponderada original em {r['id']}")
                break
            if (r["ensemble_verdict"] == "AI") != (float(r["ensemble_score"]) >= 0.5):
                problems.append(f"veredito inconsistente com limiar 0.5 em {r['id']}")
                break
    det = meta.get("determinism_max_abs_diff")
    if det is not None and det > 1e-5:
        problems.append(f"pontuação não determinística (diferença máx. {det:.2e})")
    labels = {r["label"] for r in ok}
    if "real" not in labels or "ai" not in labels:
        problems.append("faltam classes real e/ou ai nas imagens avaliadas")
    return problems
