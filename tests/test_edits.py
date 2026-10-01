import numpy as np
from PIL import Image

from evalkit.edits import EDITS, apply_edit


def _img(w=320, h=240):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_all_edits_run_and_return_valid_image():
    for name in EDITS:
        out, fmt, kw = apply_edit(name, _img(), seed=1)
        assert out.mode == "RGB" and out.width > 0 and out.height > 0, name
        assert fmt in ("JPEG", "PNG"), name


def test_edits_are_deterministic():
    for name in EDITS:
        a, *_ = apply_edit(name, _img(), seed=7)
        b, *_ = apply_edit(name, _img(), seed=7)
        assert np.array_equal(np.asarray(a), np.asarray(b)), name


def test_random_crop_depends_on_seed():
    a, *_ = apply_edit("crop_random_60", _img(), seed=1)
    b, *_ = apply_edit("crop_random_60", _img(), seed=2)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_geometry_edits_change_size():
    assert apply_edit("resize_50", _img(), 0)[0].size == (160, 120)
    assert apply_edit("down_up_2x", _img(), 0)[0].size == (320, 240)
    assert apply_edit("crop_center_70", _img(), 0)[0].size == (224, 168)
