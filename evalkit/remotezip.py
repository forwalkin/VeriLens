"""Leitura de um ZIP remoto por requisições HTTP Range (sem baixar o arquivo inteiro).

`zipfile` precisa apenas de um objeto seekable: lemos o diretório central (fim do arquivo) e,
depois, só os bytes dos membros pedidos. Requer que o servidor suporte `Accept-Ranges: bytes`.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20, timeout: int = 60):
        self.url, self.block, self.timeout, self.pos = url, block, timeout, 0
        self.bytes_fetched = 0
        self.requests = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 206:
                raise RuntimeError(f"servidor não suporta Range (HTTP {r.status}); não é seguro seguir sem baixar tudo")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
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
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = r.read()
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
