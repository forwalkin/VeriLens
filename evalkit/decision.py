"""Regra operacional (interna) para a decisão PRELIMINAR de portar um detector para o Android.

Não é afirmação sobre a capacidade universal do detector: é a nossa regra de engenharia,
fixada ANTES de ver resultados (docs/decisions/0002). Parâmetros abaixo são ajustáveis, mas
mudar qualquer um depois de ver os resultados deve ser registrado como mudança de critério.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CRITERION_LO = 0.60   # critério INTERNO: limite inferior do IC95% do AUC > 0,60
MIN_N = 30            # amostra mínima por classe para um AUC por gerador/estrato ser julgado
MIN_N_EDIT = 20       # idem para AUC por edição (amostras menores; ICs largos)
STRATUM_DROP = 0.15   # queda de AUC num estrato (formato/resolução) vs. global => suspeita de confundidor

# Qual estrato controla cada atributo-confundidor (None = não há estrato que o controle).
STRATUM_FOR = {"is_jpeg": "formato:", "log_megapixels": "resolução:",
               "has_exif": None, "bytes_per_pixel": None, "aspect_ratio": None}

MANTER = "manter para investigação"
RECALIBRAR = "recalibrar"
DESCARTAR = "descartar"
INSUFICIENTE = "dados insuficientes"


@dataclass(frozen=True)
class AucResult:
    auc: float
    lo: float
    hi: float
    n_pos: int
    n_neg: int

    def status(self, min_n: int = MIN_N) -> str:
        """ok | fraco | acaso | invertido | amostra pequena (critério interno, não universal)."""
        if self.n_pos < min_n or self.n_neg < min_n or self.auc != self.auc:
            return "amostra pequena"
        if self.lo > CRITERION_LO:
            return "ok"
        if self.lo > 0.5:
            return "fraco"
        if self.hi < 0.5:
            return "invertido"
        return "acaso"


@dataclass
class Evidence:
    detector: str
    global_: AucResult
    generators: dict[str, AucResult] = field(default_factory=dict)   # REAL vs cada gerador
    strata: dict[str, AucResult] = field(default_factory=dict)       # REAL vs IA dentro de formato/resolução
    robustness: dict[str, AucResult] = field(default_factory=dict)   # real editada vs IA editada, por grupo de edição
    dataset_confounders: list[str] = field(default_factory=list)     # atributos com AUC metadado >=0.8 ou <=0.2


@dataclass
class Decision:
    label: str
    gaps: list[str]
    notes: list[str]


def decide(ev: Evidence) -> Decision:
    g = ev.global_
    if g.n_pos < MIN_N or g.n_neg < MIN_N:
        return Decision(INSUFICIENTE, [f"amostra global pequena (REAL n={g.n_neg}, IA n={g.n_pos})"], [])

    main = {k: r for k, r in ev.generators.items() if r.n_pos >= MIN_N and r.n_neg >= MIN_N}
    small = [k for k in ev.generators if k not in main]
    notes = [f"gerador com amostra pequena (n<{MIN_N}), não julgado: {', '.join(sorted(small))}"] if small else []
    if not main:
        return Decision(INSUFICIENTE, [f"nenhum gerador com n>={MIN_N}"], notes)

    gaps: list[str] = []
    if g.status() != "ok":
        gaps.append(f"AUC global: limite inferior do IC {g.lo:.2f} <= {CRITERION_LO:.2f} ({g.status()})")
    for name, r in sorted(main.items()):
        st = r.status()
        if st in ("acaso", "invertido"):
            gaps.append(f"gerador {name}: AUC {r.auc:.2f} [{r.lo:.2f}, {r.hi:.2f}] — próximo do acaso/invertido")
        elif st == "fraco":
            gaps.append(f"gerador {name}: IC inferior {r.lo:.2f} <= {CRITERION_LO:.2f} (fraco)")
    for name, r in sorted(ev.robustness.items()):
        st = r.status(MIN_N_EDIT)
        if st in ("acaso", "invertido", "fraco"):
            gaps.append(f"robustez ({name}): AUC {r.auc:.2f} [{r.lo:.2f}, {r.hi:.2f}] ({st})")
    judged = {k: r for k, r in ev.strata.items() if r.status() != "amostra pequena"}
    for name, r in sorted(judged.items()):
        if r.status() in ("acaso", "invertido") or g.auc - r.auc > STRATUM_DROP:
            gaps.append(f"possível confundidor ({name}): AUC {r.auc:.2f} [{r.lo:.2f}, {r.hi:.2f}] vs global {g.auc:.2f}")
    uncontrolled = [a for a in ev.dataset_confounders
                    if STRATUM_FOR.get(a) is None or not any(k.startswith(STRATUM_FOR[a]) for k in judged)]
    if uncontrolled:
        gaps.append("confundidores no dataset sem estrato avaliável para descartá-los: " + ", ".join(uncontrolled))
    if not ev.robustness:
        notes.append("robustez em imagens editadas não avaliada (sem IA editada no dataset)")

    if not gaps:
        return Decision(MANTER, [], notes)
    if g.status() != "ok" and not any(r.status() == "ok" for r in main.values()):
        return Decision(DESCARTAR, gaps, notes)
    return Decision(RECALIBRAR, gaps, notes)
