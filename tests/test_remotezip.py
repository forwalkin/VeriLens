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


def _serve(handler_cls):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/x.zip"


def _bad_server(mode):
    """1ª requisição Range (sondagem) é válida; as seguintes se comportam mal conforme `mode`."""
    data = _zip_bytes()

    class Bad(http.server.BaseHTTPRequestHandler):
        calls = 0
        full_body_sent = 0

        def log_message(self, *a):
            pass

        def do_GET(self):
            type(self).calls += 1
            a, b = self.headers["Range"].split("=")[1].split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            if type(self).calls == 1 or mode == "ok":
                chunk, status, cr = data[a: b + 1], 206, f"bytes {a}-{b}/{len(data)}"
            elif mode == "later_200":
                chunk, status, cr = data, 200, None
                type(self).full_body_sent = len(data)
            elif mode == "wrong_range":
                chunk, status, cr = data[0:10], 206, f"bytes 0-9/{len(data)}"
            elif mode == "short_body":
                chunk, status, cr = data[a: b + 1][:-5], 206, f"bytes {a}-{b}/{len(data)}"
            self.send_response(status)
            if cr:
                self.send_header("Content-Range", cr)
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            try:
                self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Bad, *_serve(Bad), len(data)


@pytest.mark.parametrize("mode", ["later_200", "wrong_range", "short_body"])
def test_bad_range_responses_after_probe_are_rejected(mode):
    Bad, srv, url, total = _bad_server(mode)
    try:
        with pytest.raises(RuntimeError):
            zf, f = open_remote_zip(url)
            zf.namelist()
    finally:
        srv.shutdown()


def test_later_200_is_not_read_into_memory():
    Bad, srv, url, total = _bad_server("later_200")
    try:
        from evalkit.remotezip import HttpRangeFile

        f = HttpRangeFile(url)
        f.seek(0)
        with pytest.raises(RuntimeError, match="Range"):
            f.read(100)
        assert f.bytes_fetched <= 2  # só a sondagem; o corpo de 200 não foi lido
    finally:
        srv.shutdown()


def test_probe_is_counted_in_transfer_stats(server):
    url, total = server
    from evalkit.remotezip import HttpRangeFile

    f = HttpRangeFile(url)
    assert f.requests == 1 and f.bytes_fetched == 1


# --- sondagem estrita e limite de leitura (revisão do PR #1) -----------------------------------


class _FakeResp:
    """Resposta HTTP falsa que registra os tamanhos pedidos a read()."""

    def __init__(self, status=206, headers=None, body=b""):
        self.status, self.headers, self._body = status, dict(headers or {}), body
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        self.read_sizes.append(n)
        out = self._body if n < 0 else self._body[:n]
        self._body = self._body[len(out):]
        return out


def _fake_urlopen(monkeypatch, *responses):
    queue = list(responses)
    monkeypatch.setattr("evalkit.remotezip.urllib.request.urlopen", lambda *a, **k: queue.pop(0))


def _good_probe(total=100):
    return _FakeResp(headers={"Content-Range": f"bytes 0-0/{total}", "Content-Length": "1"}, body=b"P")


@pytest.mark.parametrize(
    "headers, body, msg",
    [
        ({"Content-Range": "bytes 5-5/100", "Content-Length": "1"}, b"x", "sondagem"),      # intervalo incorreto
        ({"Content-Range": "bytes 0-9/100", "Content-Length": "10"}, b"x" * 10, "sondagem"),  # intervalo maior
        ({"Content-Range": "bytes 0-0/0", "Content-Length": "1"}, b"x", "inválido"),         # tamanho total inválido
        ({"Content-Range": "bytes 0-0/100"}, b"", "sem corpo"),                               # corpo vazio, sem Content-Length
        ({"Content-Range": "bytes 0-0/100", "Content-Length": "0"}, b"", "Content-Length"),  # Content-Length incoerente
        ({"Content-Range": "bytes 0-0/100", "Content-Length": "2"}, b"xx", "Content-Length"),
    ],
)
def test_probe_must_be_exactly_byte_zero(monkeypatch, headers, body, msg):
    from evalkit.remotezip import HttpRangeFile

    _fake_urlopen(monkeypatch, _FakeResp(headers=headers, body=body))
    with pytest.raises(RuntimeError, match=msg):
        HttpRangeFile("http://x/y.zip")


def test_probe_reads_at_most_one_byte(monkeypatch):
    from evalkit.remotezip import HttpRangeFile

    probe = _good_probe()
    _fake_urlopen(monkeypatch, probe)
    f = HttpRangeFile("http://x/y.zip")
    assert probe.read_sizes == [1] and f.size == 100 and f.requests == 1 and f.bytes_fetched == 1


def test_fetch_reads_at_most_the_requested_bytes(monkeypatch):
    from evalkit.remotezip import HttpRangeFile

    # corpo maior que o pedido, sem Content-Length: o cliente só pode pedir `want` bytes
    body = _FakeResp(headers={"Content-Range": "bytes 10-19/100"}, body=b"A" * 10 + b"EXTRA")
    _fake_urlopen(monkeypatch, _good_probe(), body)
    f = HttpRangeFile("http://x/y.zip")
    assert f._fetch(10, 19) == b"A" * 10
    assert body.read_sizes == [10]


def test_fetch_rejects_wrong_content_length_before_reading_body(monkeypatch):
    from evalkit.remotezip import HttpRangeFile

    body = _FakeResp(headers={"Content-Range": "bytes 10-19/100", "Content-Length": "11"}, body=b"A" * 11)
    _fake_urlopen(monkeypatch, _good_probe(), body)
    f = HttpRangeFile("http://x/y.zip")
    with pytest.raises(RuntimeError, match="Content-Length"):
        f._fetch(10, 19)
    assert body.read_sizes == []  # o corpo nem foi lido


# --- retentativa de respostas truncadas e falhas temporárias -----------------------------------


def _truncating_server(n_truncate):
    """Sondagem OK; as `n_truncate` primeiras requisições seguintes têm o corpo CORTADO no meio
    (Content-Length correto, conexão fechada antes do fim), como o Zenodo fez na inspeção real."""
    data = _zip_bytes()

    class Trunc(http.server.BaseHTTPRequestHandler):
        calls = 0

        def log_message(self, *a):
            pass

        def do_GET(self):
            type(self).calls += 1
            a, b = self.headers["Range"].split("=")[1].split("-")
            a, b = int(a), min(int(b), len(data) - 1)
            chunk = data[a: b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            if type(self).calls > 1 and type(self).calls <= 1 + n_truncate:
                chunk = chunk[: len(chunk) // 4]  # corta e fecha a conexão
            try:
                self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv, url = _serve(Trunc)
    return Trunc, srv, url, data


def test_truncated_body_is_retried_and_recovers():
    from evalkit.remotezip import HttpRangeFile

    Trunc, srv, url, data = _truncating_server(n_truncate=2)
    try:
        f = HttpRangeFile(url, retries=3, backoff=0)
        f.seek(0)
        assert f.read(100) == data[:100]  # recuperou após 2 truncamentos
        assert f.retried == 2
        assert f.requests == 1 + 3  # sondagem + 2 falhas + 1 sucesso
        assert f.bytes_fetched > 1  # conta também os bytes das tentativas truncadas
    finally:
        srv.shutdown()


def test_truncated_body_gives_up_after_retries():
    from evalkit.remotezip import HttpRangeFile

    Trunc, srv, url, data = _truncating_server(n_truncate=99)
    try:
        f = HttpRangeFile(url, retries=2, backoff=0)
        f.seek(0)
        with pytest.raises(RuntimeError, match="3 tentativas"):
            f.read(100)
        assert Trunc.calls == 1 + 3  # sondagem + 3 tentativas, nem uma a mais
    finally:
        srv.shutdown()


@pytest.mark.parametrize("mode", ["later_200", "wrong_range"])
def test_protocol_violations_are_never_retried(mode):
    from evalkit.remotezip import HttpRangeFile

    Bad, srv, url, total = _bad_server(mode)
    try:
        f = HttpRangeFile(url, retries=5, backoff=0)
        f.seek(0)
        with pytest.raises(RuntimeError):
            f.read(100)
        assert Bad.calls == 2  # sondagem + 1 única tentativa: sem Range, para na hora
        assert f.retried == 0
    finally:
        srv.shutdown()


def _fake_urlopen_seq(monkeypatch, *items):
    queue = list(items)

    def fake(*a, **k):
        it = queue.pop(0)
        if isinstance(it, BaseException):
            raise it
        return it

    monkeypatch.setattr("evalkit.remotezip.urllib.request.urlopen", fake)


def _http_error(code):
    import urllib.error

    return urllib.error.HTTPError("http://x/y.zip", code, "erro", {}, None)


def test_temporary_http_error_is_retried(monkeypatch):
    from evalkit.remotezip import HttpRangeFile

    good = _FakeResp(headers={"Content-Range": "bytes 10-19/100", "Content-Length": "10"}, body=b"A" * 10)
    _fake_urlopen_seq(monkeypatch, _good_probe(), _http_error(503), TimeoutError("lento"), good)
    f = HttpRangeFile("http://x/y.zip", retries=3, backoff=0)
    assert f._fetch(10, 19) == b"A" * 10
    assert f.retried == 2


def test_permanent_http_error_is_not_retried(monkeypatch):
    import urllib.error

    from evalkit.remotezip import HttpRangeFile

    _fake_urlopen_seq(monkeypatch, _good_probe(), _http_error(404))
    f = HttpRangeFile("http://x/y.zip", retries=3, backoff=0)
    with pytest.raises(urllib.error.HTTPError):
        f._fetch(10, 19)
    assert f.retried == 0
