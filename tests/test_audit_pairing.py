"""Auditoria completa do pareamento Synthbuster x prompts.csv (TASK-0002)."""
import io
import json
import threading
import http.server
import zipfile

import pytest

from scripts.audit_synthbuster_pairing import KEY_COLUMN, audit, main

KEYS = ["r01t", "r02t", "r03t", "r04t"]


def gens(**over):
    base = {g: [f"{k}.png" for k in KEYS] for g in ("genA", "genB", "genC")}
    base.update(over)
    return base


def test_perfect_pairing_is_confirmed():
    rep = audit(gens(), KEYS, expected_keys=4)
    s = rep["summary"]
    assert s["pairing_proven"] and s["reasons_not_proven"] == []
    assert s["same_key_set_in_every_generator"] and s["same_stems_in_every_generator"]
    assert s["every_key_has_exactly_one_image_in_every_generator"]
    a = rep["generators"]["genA"]
    assert (a["n_files"], a["keys_matched_exactly_once"], a["stem_matches"], a["exact_name_matches"]) == (4, 4, 4, 0)


def test_wrong_expected_key_count_is_not_proven():
    rep = audit(gens(), KEYS, expected_keys=1000)
    assert not rep["summary"]["pairing_proven"]
    assert any("1000" in r for r in rep["summary"]["reasons_not_proven"])


def test_missing_and_extra_files_are_reported_per_generator():
    rep = audit(gens(genB=["r01t.png", "r02t.png", "r03t.png"], genC=[f"{k}.png" for k in KEYS] + ["zzz.png"]), KEYS, 4)
    assert not rep["summary"]["pairing_proven"]
    assert rep["generators"]["genB"]["keys_without_image"] == ["r04t"]
    assert rep["generators"]["genC"]["files_without_key"] == ["zzz.png"]
    assert rep["generators"]["genA"]["perfect"]  # um gerador correto não prova os outros
    assert not rep["summary"]["same_key_set_in_every_generator"]


def test_same_stem_with_two_extensions_is_ambiguous_not_chosen():
    rep = audit(gens(genA=[f"{k}.png" for k in KEYS] + ["r01t.jpg"]), KEYS, 4)
    a = rep["generators"]["genA"]
    assert a["keys_with_multiple_candidate_images"] == {"r01t": ["r01t.jpg", "r01t.png"]}
    assert a["duplicate_stems"] == ["r01t"] and a["keys_matched_exactly_once"] == 3
    assert not rep["summary"]["pairing_proven"]


def test_case_and_space_differences_are_listed_but_never_counted_as_matches():
    names = ["r01t.png", "R02T.png", " r03t.png", "r04t.png"]
    rep = audit(gens(genA=names), KEYS, 4)
    a = rep["generators"]["genA"]
    assert a["keys_matched_exactly_once"] == 2  # só r01t e r04t
    assert {d["key"] for d in a["case_or_space_only_differences"]} == {"r02t", "r03t"}
    assert not rep["summary"]["pairing_proven"]


def test_exact_match_with_extension_is_reported_separately_from_stem():
    keys = ["a.png", "b"]
    rep = audit({"g": ["a.png", "b.png"]}, keys, 2)
    g = rep["generators"]["g"]
    assert (g["exact_name_matches"], g["stem_matches"], g["keys_matched_exactly_once"]) == (1, 1, 2)
    assert rep["summary"]["pairing_proven"]


def test_duplicate_csv_keys_whitespace_and_stem_collisions():
    keys = KEYS + ["r01t", " r09t", "x.png", "x.jpg"]
    rep = audit(gens(), keys, expected_keys=len(set(keys)))
    s = rep["summary"]
    assert rep["generators"]["genA"]["duplicate_csv_keys"] == ["r01t"]
    assert s["keys_with_edge_whitespace"] == [" r09t"]
    assert s["key_stem_collisions"] == {"x": ["x.jpg", "x.png"]}
    assert not s["pairing_proven"]


def test_duplicate_file_names_inside_a_folder_are_reported():
    rep = audit(gens(genA=[f"{k}.png" for k in KEYS] + ["r01t.png"]), KEYS, 4)
    assert rep["generators"]["genA"]["duplicate_file_names"] == ["r01t.png"]
    assert not rep["generators"]["genA"]["perfect"]


def _zip(mutate=None, secret="SEGREDO_DO_PROMPT"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for g in ("genA", "genB"):
            for k in KEYS:
                z.writestr(f"syn/{g}/{k}.png", b"x")
        z.writestr("syn/genA/prompts.txt", "ignorar")  # não é imagem
        rows = "\n".join(f"{k},{secret} {k},m1" for k in KEYS)
        z.writestr("syn/prompts.csv", f"{KEY_COLUMN},Prompt,Other\n{rows}\n")
        if mutate:
            mutate(z)
    return buf.getvalue()


def test_cli_local_zip_confirms_and_never_copies_prompt_text(tmp_path):
    zp = tmp_path / "s.zip"
    zp.write_bytes(_zip())
    rep = main(["--zip", str(zp), "--out", str(tmp_path / "out"), "--expected-keys", "4"])
    assert rep["summary"]["pairing_proven"] and rep["summary"]["n_generators"] == 2 and rep["summary"]["n_image_files"] == 8
    text = (tmp_path / "out" / "synthbuster_pairing_audit.json").read_text() + (tmp_path / "out" / "synthbuster_pairing_audit.md").read_text()
    assert "SEGREDO_DO_PROMPT" not in text and "Prompt" not in text and "CONFIRMADO" in text
    assert json.loads((tmp_path / "out" / "synthbuster_pairing_audit.json").read_text())["summary"]["pairing_proven"]


def test_cli_reports_not_confirmed_when_an_image_is_extra(tmp_path):
    zp = tmp_path / "s.zip"
    zp.write_bytes(_zip(lambda z: z.writestr("syn/genB/extra.png", b"x")))
    rep = main(["--zip", str(zp), "--out", str(tmp_path / "out"), "--expected-keys", "4"])
    assert not rep["summary"]["pairing_proven"] and rep["generators"]["syn/genB"]["files_without_key"] == ["extra.png"]
    assert "NÃO CONFIRMADO" in (tmp_path / "out" / "synthbuster_pairing_audit.md").read_text()


def test_cli_missing_key_column_lists_available_columns(tmp_path):
    zp = tmp_path / "s.zip"
    zp.write_bytes(_zip())
    with pytest.raises(SystemExit, match="colunas disponíveis"):
        main(["--zip", str(zp), "--out", str(tmp_path), "--key-column", "nao existe"])


def test_cli_remote_reads_only_listing_and_csv(tmp_path):
    from tests.test_remotezip import RangeHandler

    RangeHandler.data, RangeHandler.hits = _zip(), 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rep = main(["--url", f"http://127.0.0.1:{srv.server_port}/s.zip", "--out", str(tmp_path), "--expected-keys", "4"])
    finally:
        srv.shutdown()
    assert rep["summary"]["pairing_proven"]
    t = rep["transfer"]
    assert t["zip_total_bytes"] == len(RangeHandler.data) and t["http_requests"] >= 2 and t["retried"] == 0
