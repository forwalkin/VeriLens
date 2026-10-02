import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def test_setup_base_is_committed_executable():
    """Guarda contra commitar a mudança de modo 100755 -> 100644 em scripts/setup_base.sh."""
    out = subprocess.run(["git", "ls-files", "-s", "scripts/setup_base.sh"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout:
        pytest.skip("fora de um repositório git")
    assert out.stdout.split()[0] == "100755", out.stdout


def _tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout:
        pytest.skip("fora de um repositório git")
    for rel in out.stdout.split("\n"):
        p = ROOT / rel
        if rel and p.suffix in {".md", ".toml", ".py", ".txt", ".sh", ".csv", ".json"}:
            yield rel, p.read_text(encoding="utf-8", errors="ignore")


def test_old_synthbuster_doi_is_gone_from_tracked_files():
    old = "10.5281/zenodo." + "10066047"  # montado por concatenação para o teste não se auto-detectar
    assert [rel for rel, txt in _tracked_text_files() if old in txt] == []


def test_readme_only_mentions_scripts_that_exist():
    import re

    readme = (ROOT / "README.md").read_text()
    missing = [m for m in set(re.findall(r"scripts/(\w+\.(?:py|sh))", readme)) if not (ROOT / "scripts" / m).exists()]
    assert missing == []


def test_documented_source_kinds_match_implemented_ones():
    """kinds citados em sources.example.toml/README devem existir em build_dataset.py (e vice-versa)."""
    import re

    code = (ROOT / "scripts" / "build_dataset.py").read_text()
    implemented = set(re.findall(r'kind == "(\w+)"', code))
    example = (ROOT / "sources.example.toml").read_text()
    used = set(re.findall(r'^\s*#?\s*kind = "(\w+)"', example, flags=re.M))
    documented = set(re.findall(r"^#\s{2,}(\w+)\s+:", example, flags=re.M))
    assert implemented == {"local_dir", "url_zip", "remote_zip", "hf_dataset"}
    assert used <= implemented and documented == implemented


def test_readme_states_270_minimum_and_300_target():
    readme = " ".join((ROOT / "README.md").read_text().split())  # normaliza quebras de linha
    assert "9 × 30 = 270" in readme and "300 pares" in readme and "3000 imagens" in readme
    assert "pareamento por conteúdo/descrição" in readme.lower()
    assert "não é a mesma fotografia" in readme
    assert "n_pairs = 300" in (ROOT / "sources.example.toml").read_text()


def test_readme_has_the_eleven_step_flow_in_order():
    readme = (ROOT / "README.md").read_text()
    steps = ["Inspecionar Synthbuster", "Confirmar estrutura/chaves", "Configurar sources.toml", "Montar dataset",
             "Auditar confundidores", "Criar edited / ai_edited", "Benchmark pequeno com modelos reais", "sanity_check",
             "Benchmark completo", "report.md + AUDIT.md", "Decisão sobre quais detectores merecem port Android"]
    pos = [readme.index(f"{i + 1:>2}. {t}") for i, t in enumerate(steps)]
    assert pos == sorted(pos)
