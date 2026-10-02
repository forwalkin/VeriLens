"""Leitura de ZIP por Range e script de inspeção, com servidor HTTP local e ZIP SINTÉTICO.

A estrutura/nomes aqui são inventados para o teste; a estrutura real do Synthbuster é descoberta
rodando scripts/inspect_synthbuster.py contra o Zenodo.
"""

import http.server
import io
import json
import threading
import zipfile

import numpy as np
import pytest
from PIL import Image

from evalkit.remotezip import open_remote_zip
from scripts import inspect_synthbuster


class RangeHandler(http.server.BaseHTTPRequestHandler):
    data = b""
    hits = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.data)))
            self.end_headers()
            self.wfile.write(self.data)
            return
        a, b = rng.split("=")[1].split("-")
        a, b = int(a), min(int(b), len(self.data) - 1)
        chunk = self.data[a: b + 1]
        type(self).hits += 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.data)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


def _zip_bytes():
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for gen in ("genA", "genB", "genC"):
            for i in range(5):
                im = Image.fromarray(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8))
                b = io.BytesIO()
                im.save(b, "PNG")
                z.writestr(f"synth/{gen}/{i:03d}.png", b.getvalue())
        z.writestr("synth/prompts.csv", "image,prompt,gen\n000,a cat,genA\n001,a dog,genB\n")
    return buf.getvalue()


@pytest.fixture()
def server():
    RangeHandler.data = _zip_bytes()
    RangeHandler.hits = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/x.zip", len(RangeHandler.data)
    srv.shutdown()


def test_remote_zip_lists_without_downloading_everything(server):
    url, total = server
    zf, f = open_remote_zip(url)
    assert len(zf.namelist()) == 16
    assert f.bytes_fetched < total * 0.05  # só o diretório central


def test_remote_zip_reads_one_member_only(server):
    url, total = server
    zf, f = open_remote_zip(url)
    data = zf.read("synth/genB/002.png")
    assert Image.open(io.BytesIO(data)).size == (400, 400)
    assert f.bytes_fetched < total * 0.25


def test_server_without_range_is_rejected(monkeypatch):
    class NoRange(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "3")
            self.end_headers()
            self.wfile.write(b"abc")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NoRange)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with pytest.raises(RuntimeError, match="Range"):
        open_remote_zip(f"http://127.0.0.1:{srv.server_port}/x.zip")
    srv.shutdown()


def test_inspect_script_reports_structure_and_transfer(server, tmp_path):
    url, total = server
    inspect_synthbuster.main(["--url", url, "--out", str(tmp_path)])
    rep = json.loads((tmp_path / "synthbuster_inspection.json").read_text())
    assert set(rep["directories"]) == {"synth/genA", "synth/genB", "synth/genC"}
    assert all(v["n_images"] == 5 for v in rep["directories"].values())
    assert rep["prompts"]["columns"] == ["image", "prompt", "gen"]
    assert rep["transfer"]["bytes_fetched"] < total * 0.5
    assert "genA" in (tmp_path / "synthbuster_inspection.md").read_text()


def test_build_dataset_remote_zip_selects_members_without_full_download(server, tmp_path):
    from evalkit.manifest import read_manifest
    from scripts import build_dataset

    url, total = server
    (tmp_path / "sources.toml").write_text(f"""
[[source]]
name = "synth"
label = "ai"
kind = "remote_zip"
url = "{url}"
generator_from = "parent_dir"
n = 2
""")
    build_dataset.main(["--sources", str(tmp_path / "sources.toml"), "--out", str(tmp_path / "ds")])
    rows = read_manifest(tmp_path / "ds" / "manifest.csv")
    assert len(rows) == 6 and {r.generator for r in rows} == {"genA", "genB", "genC"}
    assert all((tmp_path / "ds" / r.path).exists() for r in rows)
