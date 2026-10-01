"""Edições determinísticas aplicadas a fotos reais (categoria 'edited').

Cada edição é uma função (Image, rng) -> (Image, formato_de_saída, kwargs_de_save).
Edições de compressão salvam em JPEG; as demais salvam em PNG para isolar o efeito
da edição (sem adicionar compressão extra).

'screenshot_sim' é uma SIMULAÇÃO (moldura + reescala); capturas reais de tela devem
ser adicionadas manualmente (ver README).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

Edit = Callable[[Image.Image, np.random.Generator], tuple[Image.Image, str, dict]]
_PNG = ("PNG", {})


def _jpeg(q: int) -> Edit:
    return lambda im, rng: (im, "JPEG", {"quality": q})


def _resize_down(f: float) -> Edit:
    def fn(im, rng):
        w, h = im.size
        return im.resize((max(1, int(w * f)), max(1, int(h * f))), Image.LANCZOS), *_PNG
    return fn


def _down_up(im, rng):
    w, h = im.size
    small = im.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
    return small.resize((w, h), Image.BICUBIC), *_PNG


def _crop(frac: float, random: bool) -> Edit:
    def fn(im, rng):
        w, h = im.size
        cw, ch = int(w * frac), int(h * frac)
        x = int(rng.integers(0, w - cw + 1)) if random else (w - cw) // 2
        y = int(rng.integers(0, h - ch + 1)) if random else (h - ch) // 2
        return im.crop((x, y, x + cw, y + ch)), *_PNG
    return fn


def _enh(cls, factor: float) -> Edit:
    return lambda im, rng: (cls(im).enhance(factor), *_PNG)


def _warm(im, rng):
    a = np.asarray(im, dtype=np.float32)
    a[..., 0] *= 1.12
    a[..., 2] *= 0.88
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), *_PNG


def _grayscale(im, rng):
    return im.convert("L").convert("RGB"), *_PNG


def _sepia(im, rng):
    a = np.asarray(im, dtype=np.float32)
    m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    return Image.fromarray(np.clip(a @ m.T, 0, 255).astype(np.uint8)), *_PNG


def _vignette(im, rng):
    w, h = im.size
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    mask = np.clip(1.0 - 0.45 * r**2, 0.35, 1.0)[..., None]
    a = np.asarray(im, dtype=np.float32) * mask
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), *_PNG


def _blur(im, rng):
    return im.filter(ImageFilter.GaussianBlur(radius=2)), *_PNG


def _sharpen(im, rng):
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2)), *_PNG


def _social(im, rng):
    """Aproximação de pipeline de rede social: lado maior 1080 px + JPEG q70."""
    w, h = im.size
    s = 1080 / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return im, "JPEG", {"quality": 70}


def _screenshot_sim(im, rng):
    """Tela de celular 1080x2340: barra de status, imagem reescalada ao centro, barra inferior."""
    cw, ch = 1080, 2340
    canvas = Image.new("RGB", (cw, ch), (18, 18, 18))
    s = cw / im.width
    im2 = im.resize((cw, max(1, int(im.height * s))), Image.BICUBIC)
    canvas.paste(im2, (0, (ch - im2.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, cw, 90), fill=(0, 0, 0))
    d.rectangle((0, ch - 140, cw, ch), fill=(0, 0, 0))
    return canvas, *_PNG


EDITS: dict[str, Edit] = {
    "jpeg_q75": _jpeg(75),
    "jpeg_q50": _jpeg(50),
    "jpeg_q30": _jpeg(30),
    "resize_50": _resize_down(0.5),
    "down_up_2x": _down_up,
    "crop_center_70": _crop(0.7, random=False),
    "crop_random_60": _crop(0.6, random=True),
    "brightness_contrast": lambda im, rng: (ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(1.25)).enhance(1.3), *_PNG),
    "saturation_150": _enh(ImageEnhance.Color, 1.5),
    "warm_tone": _warm,
    "grayscale": _grayscale,
    "sepia": _sepia,
    "vignette": _vignette,
    "blur_r2": _blur,
    "sharpen": _sharpen,
    "social_1080_q70": _social,
    "screenshot_sim": _screenshot_sim,
}

EXT = {"JPEG": ".jpg", "PNG": ".png"}


def apply_edit(name: str, im: Image.Image, seed: int):
    """Aplica a edição *name* de forma determinística (mesmo seed => mesma saída)."""
    return EDITS[name](im.convert("RGB"), np.random.default_rng(seed))


# Grupos usados no relatório de robustez (agrupamento operacional, não taxonomia científica).
EDIT_GROUPS: dict[str, tuple[str, ...]] = {
    "compressao": ("jpeg_q75", "jpeg_q50", "jpeg_q30", "social_1080_q70"),
    "geometria": ("resize_50", "down_up_2x", "crop_center_70", "crop_random_60", "screenshot_sim"),
    "tonal_filtros": ("brightness_contrast", "saturation_150", "warm_tone", "grayscale",
                      "sepia", "vignette", "blur_r2", "sharpen"),
}
assert sorted(sum(EDIT_GROUPS.values(), ())) == sorted(EDITS), "toda edição deve estar em um grupo"
