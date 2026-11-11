"""Leitura de um ZIP remoto por requisições HTTP Range (sem baixar o arquivo inteiro).

`zipfile` precisa apenas de um objeto seekable: lemos o diretório central (fim do arquivo) e,
depois, só os bytes dos membros pedidos. Requer que o servidor suporte `Accept-Ranges: bytes`.
"""

from __future__ import annotations

import http.client
import io
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile


_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def _parse_content_range(r) -> tuple[int, int, int]:
    """Valida a resposta ANTES de ler o corpo: só aceita 206 com Content-Range bem formado."""
    if r.status != 206:
        raise RuntimeError(f"servidor não suporta Range (HTTP {r.status}); não é seguro seguir sem baixar tudo")
    m = _CONTENT_RANGE.fullmatch(r.headers.get("Content-Range", "").strip())
    if not m:
        raise RuntimeError("Content-Range ausente ou inválido na resposta Range")
    a, b, total = (int(x) for x in m.groups())
    if not (0 <= a <= b < total):  # também garante total > 0
        raise RuntimeError(f"Content-Range inválido: bytes {a}-{b}/{total}")
    return a, b, total


def _check_content_length(r, expected: int) -> None:
    """Se o servidor informou Content-Length, ele deve bater com o tamanho esperado (antes de ler o corpo)."""
    cl = r.headers.get("Content-Length")
    if cl is None:
        return
    if not cl.strip().isdigit() or int(cl) != expected:
        raise RuntimeError(f"Content-Length {cl!r} não corresponde ao esperado ({expected})")


# Códigos HTTP temporários que justificam nova tentativa. Todo o resto (404, 403...) falha na hora.
_RETRY_HTTP = {429, 500, 502, 503, 504}


class HttpRangeFile(io.RawIOBase):
    """Arquivo seekable sobre HTTP Range.

    Respostas TRUNCADAS (corpo menor que o pedido) e falhas de rede/HTTP temporárias são repetidas até
    `retries` vezes, com espera crescente (`backoff` * 1, 2, 4... s). Violações de protocolo NUNCA são
    repetidas: resposta que não seja 206, Content-Range ou Content-Length inesperados. Assim, um servidor
    sem suporte a Range continua parando a execução na hora, em vez de gastar tentativas.
    """

    def __init__(self, url: str, block: int = 1 << 20, timeout: int = 60, retries: int = 5, backoff: float = 1.0):
        self.url, self.block, self.timeout, self.pos = url, block, timeout, 0
        self.retries, self.backoff = retries, backoff
        self.bytes_fetched = 0  # todos os bytes recebidos, inclusive de tentativas truncadas
        self.requests = 0  # todas as requisições HTTP, inclusive as repetidas
        self.retried = 0  # quantas tentativas falharam e foram repetidas
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            a, b, self.size = _parse_content_range(r)
            if (a, b) != (0, 0):
                raise RuntimeError(f"sondagem: Content-Range inesperado (pedido 0-0, recebido {a}-{b}/{self.size})")
            _check_content_length(r, 1)
            probe = r.read(1)  # leitura limitada ao 1 byte pedido
        if len(probe) != 1:
            raise RuntimeError("sondagem sem corpo: esperado exatamente 1 byte")
        self.requests += 1
        self.bytes_fetched += len(probe)
        self._cache: tuple[int, bytes] = (0, b"")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        want = end - start + 1
        last = "?"
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            self.requests += 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    a, b, total = _parse_content_range(r)  # RuntimeError aqui = violação de protocolo: não repete
                    if (a, b) != (start, end) or total != self.size:
                        raise RuntimeError(
                            f"Content-Range inesperado: pedido {start}-{end}/{self.size}, recebido {a}-{b}/{total}"
                        )
                    _check_content_length(r, want)
                    data = r.read(want)  # limite duro: nunca lê mais que o pedido
            except urllib.error.HTTPError as e:
                if e.code not in _RETRY_HTTP:
                    raise
                last = f"HTTP {e.code}"
            except http.client.IncompleteRead as e:  # corpo cortado: conta os bytes parciais que chegaram
                self.bytes_fetched += len(e.partial)
                last = f"IncompleteRead: {len(e.partial)} bytes recebidos antes do corte"
            except OSError as e:  # rede, timeout, conexão cortada
                last = f"{type(e).__name__}: {e}"
            else:
                self.bytes_fetched += len(data)
                if len(data) == want:
                    return data
                last = f"corpo truncado: {len(data)} de {want} bytes"
            self.retried += 1
            print(
                f"[remotezip] tentativa {attempt + 1}/{self.retries + 1} falhou ({start}-{end}): {last}",
                file=sys.stderr,
            )
        raise RuntimeError(f"falha ao ler {start}-{end} após {self.retries + 1} tentativas: {last}")

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        c0, buf = self._cache
        if c0 <= self.pos and self.pos + n <= c0 + len(buf):
            out = buf[self.pos - c0: self.pos - c0 + n]
        else:
            size = max(n, self.block)
            data = self._fetch(self.pos, min(self.pos + size, self.size) - 1)
            self._cache = (self.pos, data)
            out = data[:n]
        self.pos += len(out)
        return out

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def open_remote_zip(url: str) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    f = HttpRangeFile(url)
    return zipfile.ZipFile(f), f
