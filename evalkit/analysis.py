"""Análises sobre raw.csv: AUC global/por gerador/por estrato/por edição, pareamento e confundidores.

Todas as funções são puras (recebem linhas já carregadas) para facilitar teste.
"""

from __future__ import annotations

import csv
import pathlib

import numpy as np

from .decision import AucResult, Evidence
from .edits import EDIT_GROUPS
from .metrics import auc, bootstrap_auc_ci, cluster_bootstrap_auc_ci, paired_win_rate

DETECTOR_ORDER = ("frequency", "clip_drift", "hf_classifier", "ensemble")
NAMES = {"frequency": "Frequency", "clip_drift": "CLIP", "hf_classifier": "SDXL Detector", "ensemble": "Ensemble"}


def load_raw(run: pathlib.Path) -> tuple[list[dict], list[dict], list[str]]:
    with open(run / "raw.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return rows, [r for r in rows if not r["error"]], header


def detectors_in(header: list[str]) -> list[str]:
    return [d for d in DETECTOR_ORDER if f"{d}_score" in header]


def scores(rows: list[dict], det: str) -> np.ndarray:
    return np.array([float(r[f"{det}_score"]) for r in rows])


def auc_result(pos: np.ndarray, neg: np.ndarray, n_boot: int = 1000, seed: int = 0,
               pos_clusters=None, neg_clusters=None) -> AucResult:
    if len(pos) == 0 or len(neg) == 0:
        return AucResult(float("nan"), float("nan"), float("nan"), len(pos), len(neg))
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    a = auc(y, s)
    if pos_clusters is not None:
        lo, hi = cluster_bootstrap_auc_ci(pos, neg, pos_clusters, neg_clusters, n_boot=n_boot, seed=seed)
    else:
        lo, hi = bootstrap_auc_ci(y, s, n_boot=n_boot, seed=seed)
    return AucResult(a, lo, hi, len(pos), len(neg))


def _sel(rows, label, **eq):
    return [r for r in rows if r["label"] == label and all(r[k] == v for k, v in eq.items())]


def global_auc(rows, det, n_boot=1000) -> AucResult:
    return auc_result(scores(_sel(rows, "ai"), det), scores(_sel(rows, "real"), det), n_boot)


def per_generator(rows, det, n_boot=1000) -> dict[str, AucResult]:
    """REAL (todas) vs cada gerador de IA. As reais são compartilhadas entre geradores."""
    real = scores(_sel(rows, "real"), det)
    gens = sorted({r["generator"] for r in rows if r["label"] == "ai"})
    return {g: auc_result(scores(_sel(rows, "ai", generator=g), det), real, n_boot) for g in gens}


def paired(rows, det, n_boot=1000) -> dict[str, tuple[int, float, float, float]]:
    """Por gerador: (n_pares, taxa de vitória, lo, hi) usando pair_key. Vazio se não houver pareamento."""
    real_by_key = {r["pair_key"]: float(r[f"{det}_score"]) for r in rows if r["label"] == "real" and r["pair_key"]}
    out = {}
    for g in sorted({r["generator"] for r in rows if r["label"] == "ai" and r["pair_key"]}):
        ai = {r["pair_key"]: float(r[f"{det}_score"]) for r in _sel(rows, "ai", generator=g) if r["pair_key"]}
        keys = sorted(set(ai) & set(real_by_key))
        if keys:
            m, lo, hi = paired_win_rate(np.array([ai[k] for k in keys]), np.array([real_by_key[k] for k in keys]), n_boot)
            out[g] = (len(keys), m, lo, hi)
    return out


# ---------- confundidores ----------

def metadata_features(r: dict) -> dict[str, float]:
    w, h = float(r["width"]), float(r["height"])
    return {
        "log_megapixels": float(np.log10(w * h / 1e6)),
        "bytes_per_pixel": float(r["file_bytes"]) / (w * h),
        "is_jpeg": float(r["format"] == "JPEG"),
        "has_exif": float(r["has_exif"] == "1"),
        "aspect_ratio": max(w, h) / min(w, h),
    }


def metadata_confounders(rows) -> list[tuple[str, float, float, float]]:
    """(atributo, média real, média IA, AUC univariado) em REAL vs IA."""
    sub = [r for r in rows if r["label"] in ("real", "ai")]
    if not sub:
        return []
    y = np.array([r["label"] == "ai" for r in sub])
    feats = [metadata_features(r) for r in sub]
    out = []
    for name in feats[0]:
        v = np.array([f[name] for f in feats])
        out.append((name, float(v[~y].mean()) if (~y).any() else float("nan"),
                    float(v[y].mean()) if y.any() else float("nan"), auc(y, v)))
    return out


def flagged_attributes(conf) -> list[str]:
    return [n for n, _, _, a in conf if a == a and (a >= 0.8 or a <= 0.2)]


def strata_of(rows) -> dict[str, list[dict]]:
    """Estratos de REAL+IA: por formato de arquivo e por tercil de megapixels (sobre real+IA agrupados)."""
    sub = [r for r in rows if r["label"] in ("real", "ai")]
    out: dict[str, list[dict]] = {}
    for r in sub:
        out.setdefault(f"formato:{r['format']}", []).append(r)
    if sub:
        mp = np.array([float(r["width"]) * float(r["height"]) / 1e6 for r in sub])
        q = np.quantile(mp, [1 / 3, 2 / 3])
        for r, m in zip(sub, mp):
            b = 0 if m <= q[0] else 1 if m <= q[1] else 2
            out.setdefault(f"resolução:tercil{b + 1}", []).append(r)
    return out


def per_stratum(rows, det, n_boot=1000) -> dict[str, AucResult]:
    res = {}
    for name, rs in strata_of(rows).items():
        pos, neg = scores(_sel(rs, "ai"), det), scores(_sel(rs, "real"), det)
        if len(pos) and len(neg):  # só estratos com as duas classes
            res[name] = auc_result(pos, neg, n_boot)
    return res


# ---------- robustez ----------

def per_edit(rows, det, n_boot=1000) -> dict[str, dict]:
    """Por edição: AUC real editada vs IA editada, FPR/recall a 0.5 e n."""
    out = {}
    edits = sorted({r["edit"] for r in rows if r["label"] in ("edited", "ai_edited")})
    for e in edits:
        re_, ae = _sel(rows, "edited", edit=e), _sel(rows, "ai_edited", edit=e)
        neg, pos = scores(re_, det), scores(ae, det)
        out[e] = {
            "auc": auc_result(pos, neg, n_boot),
            "fpr": float((neg >= 0.5).mean()) if len(neg) else float("nan"),
            "recall": float((pos >= 0.5).mean()) if len(pos) else float("nan"),
        }
    return out


def per_edit_group(rows, det, n_boot=1000) -> dict[str, AucResult]:
    """AUC agrupado por família de edição, com bootstrap por imagem-pai (clusters)."""
    out = {}
    for gname, edits in EDIT_GROUPS.items():
        re_ = [r for r in rows if r["label"] == "edited" and r["edit"] in edits]
        ae = [r for r in rows if r["label"] == "ai_edited" and r["edit"] in edits]
        if re_ and ae:
            out[gname] = auc_result(scores(ae, det), scores(re_, det), n_boot,
                                    pos_clusters=np.array([r["parent_id"] for r in ae]),
                                    neg_clusters=np.array([r["parent_id"] for r in re_]))
    return out


def build_evidence(rows, det, n_boot=1000) -> Evidence:
    conf = flagged_attributes(metadata_confounders(rows)) if rows and "format" in rows[0] else []
    return Evidence(
        detector=det,
        global_=global_auc(rows, det, n_boot),
        generators=per_generator(rows, det, n_boot),
        strata=per_stratum(rows, det, n_boot) if "format" in rows[0] else {},
        robustness=per_edit_group(rows, det, n_boot),
        dataset_confounders=conf,
    )
