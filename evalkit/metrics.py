"""Métricas de classificação binária sem dependência de scikit-learn.

Convenção: classe positiva = IA (y=1); negativa = real/editada (y=0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Confusion:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return Confusion(
        tp=int(np.sum(y_true & y_pred)),
        fp=int(np.sum(~y_true & y_pred)),
        tn=int(np.sum(~y_true & ~y_pred)),
        fn=int(np.sum(y_true & ~y_pred)),
    )


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def prf(c: Confusion) -> dict[str, float]:
    """Accuracy, precision, recall, F1. Valores indefinidos retornam NaN (não 0)."""
    precision = _div(c.tp, c.tp + c.fp)
    recall = _div(c.tp, c.tp + c.fn)
    ok = precision == precision and recall == recall
    f1 = _div(2 * precision * recall, precision + recall) if ok else float("nan")
    return {
        "accuracy": _div(c.tp + c.tn, c.n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _div(c.fp, c.fp + c.tn),
    }


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via estatística de Mann-Whitney (empates tratados por postos médios)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """IC percentil do AUC com reamostragem estratificada por classe."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        vals[i] = auc(
            np.concatenate([np.ones(len(p)), np.zeros(len(n))]),
            np.concatenate([p, n]),
        )
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def chance_verdict(auc_value: float, ci: tuple[float, float]) -> str:
    """Resposta objetiva a 'distingue real de IA?', independente de limiar de decisão."""
    lo, hi = ci
    if auc_value != auc_value:
        return "indefinido"
    if lo <= 0.5 <= hi:
        return "indistinguível do acaso"
    return "melhor que o acaso" if lo > 0.5 else "INVERTIDO (pior que o acaso)"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para uma proporção (taxa de detecção / falso positivo por grupo)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def cluster_bootstrap_auc_ci(
    pos: np.ndarray,
    neg: np.ndarray,
    pos_clusters: np.ndarray,
    neg_clusters: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """IC do AUC reamostrando CLUSTERS (ex.: imagem-pai) dentro de cada classe.

    Necessário quando a mesma foto-pai aparece em várias edições (observações não independentes).
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pc, nc = np.asarray(pos_clusters), np.asarray(neg_clusters)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")

    def groups(vals, cl):
        out: dict = {}
        for v, c in zip(vals, cl):
            out.setdefault(c, []).append(v)
        return [np.array(v) for v in out.values()]

    gp, gn = groups(pos, pc), groups(neg, nc)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        p = np.concatenate([gp[j] for j in rng.integers(0, len(gp), len(gp))])
        n = np.concatenate([gn[j] for j in rng.integers(0, len(gn), len(gn))])
        vals[i] = auc(np.concatenate([np.ones(len(p)), np.zeros(len(n))]), np.concatenate([p, n]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_win_rate(
    ai: np.ndarray, real: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """P(score_IA > score_real) dentro de pares correspondentes (empate vale 0.5) e IC por bootstrap de pares.

    0.5 = o detector não ordena corretamente o par; 1.0 = sempre ordena a IA acima da real.
    """
    ai, real = np.asarray(ai, float), np.asarray(real, float)
    if len(ai) == 0 or len(ai) != len(real):
        return float("nan"), float("nan"), float("nan")
    w = (ai > real).astype(float) + 0.5 * (ai == real)
    rng = np.random.default_rng(seed)
    boots = np.array([w[rng.integers(0, len(w), len(w))].mean() for _ in range(n_boot)])
    return float(w.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
