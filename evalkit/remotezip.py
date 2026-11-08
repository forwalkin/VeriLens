"""Leitura de um ZIP remoto por requisições HTTP Range (sem baixar o arquivo inteiro).

`zipfile` precisa apenas de um objeto seekable: lemos o diretório central (fim do arquivo) e,
depois, só os bytes dos membros pedidos. Requer que o servidor suporte `Accept-Ranges: bytes`.
"""

from __future__ import annotations

import io
import re
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
    return a, b, total


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20, timeout: int = 60):
        self.url, self.block, self.timeout, self.pos = url, block, timeout, 0
        self.bytes_fetched = 0
        self.requests = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            _, _, self.size = _parse_content_range(r)
            probe = r.read(2)  # pedimos 1 byte; lê no máximo 2 para nunca consumir um corpo inesperado
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
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        want = end - start + 1
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            a, b, total = _parse_content_range(r)  # recusa 200 e Content-Range inesperado antes de ler o corpo
            if (a, b) != (start, end) or total != self.size:
                raise RuntimeError(
                    f"Content-Range inesperado: pedido {start}-{end}/{self.size}, recebido {a}-{b}/{total}"
                )
            data = r.read(want + 1)  # limite duro: nunca lê mais que o pedido
        if len(data) != want:
            raise RuntimeError(f"corpo com {len(data)} bytes; esperado {want} para o intervalo {start}-{end}")
        self.requests += 1
        self.bytes_fetched += len(data)
        return data

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
