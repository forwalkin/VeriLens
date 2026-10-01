from evalkit.decision import DESCARTAR, INSUFICIENTE, MANTER, RECALIBRAR, AucResult, Evidence, decide


def R(a, lo, hi, n=100):
    return AucResult(a, lo, hi, n, n)


def ev(**kw):
    base = dict(detector="d", global_=R(0.9, 0.85, 0.94),
                generators={"sd": R(0.92, 0.86, 0.96), "mj": R(0.88, 0.8, 0.93)},
                robustness={"compressao": R(0.8, 0.7, 0.88)}, strata={"formato:PNG": R(0.88, 0.8, 0.93)})
    base.update(kw)
    return Evidence(**base)


def test_clean_detector_is_kept():
    assert decide(ev()).label == MANTER


def test_high_global_auc_but_chance_on_one_main_generator_is_not_kept():
    d = decide(ev(generators={"sd": R(0.95, 0.9, 0.98), "dalle3": R(0.52, 0.42, 0.62)}))
    assert d.label == RECALIBRAR and any("dalle3" in g for g in d.gaps)


def test_small_generator_is_noted_not_judged():
    d = decide(ev(generators={"sd": R(0.9, 0.85, 0.95), "tiny": AucResult(0.5, 0.2, 0.8, 8, 100)}))
    assert d.label == MANTER and any("tiny" in n for n in d.notes)


def test_chance_everywhere_is_discarded_but_with_evidence():
    d = decide(ev(global_=R(0.51, 0.44, 0.58), generators={"sd": R(0.5, 0.4, 0.6), "mj": R(0.52, 0.42, 0.62)}))
    assert d.label == DESCARTAR and d.gaps


def test_global_fail_but_one_generator_ok_is_recalibrate_not_discard():
    d = decide(ev(global_=R(0.62, 0.55, 0.69), generators={"sd": R(0.9, 0.85, 0.95), "mj": R(0.5, 0.4, 0.6)}))
    assert d.label == RECALIBRAR


def test_robustness_failure_blocks_keep():
    assert decide(ev(robustness={"compressao": R(0.51, 0.4, 0.62)})).label == RECALIBRAR


def test_stratum_drop_flags_confounder():
    d = decide(ev(strata={"formato:JPEG": R(0.6, 0.5, 0.7)}))
    assert d.label == RECALIBRAR and any("confundidor" in g for g in d.gaps)


def test_dataset_confounder_without_strata_is_a_gap():
    d = decide(ev(strata={}, dataset_confounders=["is_jpeg"]))
    assert d.label == RECALIBRAR


def test_insufficient_data():
    assert decide(ev(global_=AucResult(0.9, 0.7, 1.0, 10, 10))).label == INSUFICIENTE
    assert decide(ev(generators={})).label == INSUFICIENTE


def test_confounder_needs_matching_stratum():
    only_res = {"resolução:tercil1": R(0.9, 0.85, 0.94)}
    d = decide(ev(strata=only_res, dataset_confounders=["is_jpeg"]))
    assert d.label == RECALIBRAR and any("is_jpeg" in g for g in d.gaps)
    ok = decide(ev(strata={"formato:PNG": R(0.9, 0.85, 0.94), "formato:JPEG": R(0.88, 0.8, 0.93)},
                   dataset_confounders=["is_jpeg"]))
    assert ok.label == MANTER
    # has_exif / bytes_per_pixel nunca são controláveis por estrato
    assert decide(ev(dataset_confounders=["has_exif"])).label == RECALIBRAR
