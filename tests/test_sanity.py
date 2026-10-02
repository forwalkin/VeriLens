from evalkit.manifest import Row, stratified_sample
from evalkit.sanity import check

W = {"frequency": 0.3, "clip_drift": 0.35, "hf_classifier": 0.35}


def _row(i, label, f, c, h, ens=None, verdict=None):
    ens = 0.3 * f + 0.35 * c + 0.35 * h if ens is None else ens
    return {"id": f"i{i}", "label": label, "error": "", "frequency_score": f, "clip_drift_score": c,
            "hf_classifier_score": h, "ensemble_score": ens, "ensemble_verdict": verdict or ("AI" if ens >= 0.5 else "REAL")}


def _rows():
    return [_row(i, "real" if i % 2 else "ai", 0.1 + i * 0.01, 0.2 + i * 0.02, 0.3 + i * 0.03) for i in range(10)]


def test_clean_run_has_no_problems():
    assert check(_rows(), {"weights": W}) == []


def test_detects_wrong_ensemble_error_constant_and_nondeterminism():
    r = _rows()
    r[0]["ensemble_score"] = 0.99
    assert any("ensemble inconsistente" in p for p in check(r, {"weights": W}))
    r = _rows()
    r[1]["error"] = "ValueError: x"
    assert any("com erro" in p for p in check(r, {"weights": W}))
    const = [_row(i, "real" if i % 2 else "ai", 0.5, 0.5, 0.5) for i in range(10)]
    assert any("constante" in p for p in check(const, {"weights": W}))
    assert any("determin" in p for p in check(_rows(), {"weights": W, "determinism_max_abs_diff": 0.01}))


def test_stratified_sample_is_small_reproducible_and_covers_groups():
    rows = [Row(id=f"a{i}", path="", label="ai", source="s", generator=g) for g in ("x", "y") for i in range(10)]
    rows += [Row(id=f"r{i}", path="", label="real", source="s", generator="cam") for i in range(10)]
    rows += [Row(id=f"e{i}", path="", label="edited", source="s", edit=e) for e in ("jpeg_q30", "sepia") for i in range(5)]
    a = stratified_sample(rows, 3, seed=0)
    assert a == stratified_sample(rows, 3, seed=0)
    assert len(a) == 3 * 2 + 3 + 3 * 2  # 2 geradores IA, real, 2 edições
    assert {r.generator for r in a if r.label == "ai"} == {"x", "y"}
