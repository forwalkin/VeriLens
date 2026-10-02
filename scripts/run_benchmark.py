#!/usr/bin/env python3
"""BENCHMARK DO PROJETO ORIGINAL: executa a implementação atual, sem alterações.

- Os detectores são os do pacote ai_image_scanner (commit fixado, árvore limpa).
- O ensemble vem de ai_image_scanner.ensemble.score_image (pesos e limiar 0.5 originais);
  os detectores são embrulhados em proxies apenas para cronometrar e capturar cada score.
- Saída: results/<run_id>/raw.csv + meta.json. Nada aqui altera calibração, pesos ou limiares.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalkit.manifest import manifest_digest, read_manifest, stratified_sample  # noqa: E402

EXPECTED_BASE_COMMIT = "4ec7edc"
DETECTORS = ("frequency", "clip_drift", "hf_classifier")


class Timed:
    """Proxy que cronometra .score() e guarda o último resultado, sem alterar o detector."""

    def __init__(self, det):
        self._det, self.last_ms, self.last_score = det, 0.0, float("nan")

    def score(self, img):
        t0 = time.perf_counter()
        res = self._det.score(img)
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.last_score = float(res["score"] if isinstance(res, dict) else res)
        return res


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def base_info(allow_dirty: bool) -> dict:
    import ai_image_scanner

    repo = pathlib.Path(ai_image_scanner.__file__).resolve().parent.parent
    commit, dirty = _git(repo, "rev-parse", "--short", "HEAD"), bool(_git(repo, "status", "--porcelain"))
    if not commit.startswith(EXPECTED_BASE_COMMIT):
        sys.exit(f"base em {commit}, esperado {EXPECTED_BASE_COMMIT}. Rode scripts/setup_base.sh")
    if dirty and not allow_dirty:
        sys.exit("base com alterações locais: o benchmark exige a implementação original")
    cal = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12]
           for p in sorted((repo / "calibration").glob("*.np*"))}
    from ai_image_scanner import config

    weights = {"frequency": config.WEIGHT_FREQUENCY, "clip_drift": config.WEIGHT_CLIP, "hf_classifier": config.WEIGHT_CLASSIFIER}
    return {"base_commit": commit, "base_dirty": dirty, "calibration_sha256": cal, "weights": weights}


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count()}
    for mod in ("numpy", "torch", "transformers", "open_clip", "PIL"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except ImportError:
            info[mod] = None
    try:
        import torch
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def build_detectors(names: tuple[str, ...]) -> tuple[dict, dict]:
    """Constrói só os detectores pedidos; retorna (proxies, tempos de carga em s)."""
    from ai_image_scanner import config

    loaders = {
        "frequency": lambda: __import__("ai_image_scanner.detectors.frequency", fromlist=["x"]).FrequencyDetector(
            calibration_path=config.FREQUENCY_CALIBRATION),
        "clip_drift": lambda: __import__("ai_image_scanner.detectors.clip_drift", fromlist=["x"]).CLIPDriftDetector(
            real_centroid_path=config.CLIP_CENTROID_REAL, ai_centroid_path=config.CLIP_CENTROID_AI),
        "hf_classifier": lambda: __import__("ai_image_scanner.detectors.hf_classifier", fromlist=["x"]).HFClassifierDetector(
            model_id=config.HF_CLASSIFIER_MODEL),
    }
    dets, load_s = {}, {}
    for n in names:
        t0 = time.perf_counter()
        dets[n] = Timed(loaders[n]())
        load_s[n] = time.perf_counter() - t0
    # Warm-up: o HF carrega o pipeline na 1ª chamada e o CLIP/torch aquecem; medir à parte.
    warm = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
    for n, d in dets.items():
        t0 = time.perf_counter()
        d.score(warm)
        load_s[f"{n}_first_call"] = time.perf_counter() - t0
    return dets, load_s


def main(argv: list[str] | None = None) -> pathlib.Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("dataset"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results"))
    ap.add_argument("--detectors", default=",".join(DETECTORS), help="subconjunto (smoke tests)")
    ap.add_argument("--limit", type=int, default=0, help="processa só as N primeiras (depuração)")
    ap.add_argument("--sample-per-group", type=int, default=0,
                    help="validação pequena: até N imagens por (rótulo, gerador|edição), seed fixa")
    ap.add_argument("--determinism-check", type=int, default=3,
                    help="re-pontua as N primeiras imagens e registra a maior diferença (0 desliga)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args(argv)

    names = tuple(n for n in args.detectors.split(",") if n)
    full = set(names) == set(DETECTORS)
    meta = {"run_id": args.run_id, "kind": "original-benchmark" if full else "PARTIAL-SMOKE",
            "detectors": names, **base_info(args.allow_dirty), "env": env_info(),
            "manifest_digest": manifest_digest(args.dataset / "manifest.csv")}

    psum = args.dataset / "pairing_summary.json"
    if psum.exists():
        meta["pairing"] = json.loads(psum.read_text())
    rows = read_manifest(args.dataset / "manifest.csv")
    if args.sample_per_group:
        rows = stratified_sample(rows, args.sample_per_group)
        meta["kind"] = "SMALL-VALIDATION" if full else meta["kind"]
    if args.limit:
        rows = rows[: args.limit]

    dets, meta["load_seconds"] = build_detectors(names)
    if full:
        from ai_image_scanner.ensemble import score_image  # implementação original do ensemble

    out_dir = args.out / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = (["id", "label", "generator", "edit", "source", "parent_id", "pair_key", "width", "height", "format", "file_bytes", "has_exif", "error"]
            + [f"{n}_score" for n in names] + [f"{n}_ms" for n in names]
            + (["ensemble_score", "ensemble_verdict"] if full else []))
    t_start = time.time()
    with open(out_dir / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            rec = {"id": r.id, "label": r.label, "generator": r.generator, "edit": r.edit,
                   "source": r.source, "parent_id": r.parent_id, "pair_key": r.pair_key, "error": ""}
            try:
                fpath = args.dataset / r.path
                raw = Image.open(fpath)
                rec["format"], rec["file_bytes"], rec["has_exif"] = raw.format, fpath.stat().st_size, int(bool(raw.getexif()))
                img = raw.convert("RGB")  # mesmo carregamento de api.py (Image.open(...).convert("RGB"))
                rec["width"], rec["height"] = img.size
                if full:
                    res = score_image(img, detectors=dets)
                    rec["ensemble_score"], rec["ensemble_verdict"] = res["score"], res["verdict"]
                else:
                    for d in dets.values():
                        d.score(img)
                for n, d in dets.items():
                    rec[f"{n}_score"], rec[f"{n}_ms"] = d.last_score, round(d.last_ms, 2)
            except Exception as exc:  # registra e segue: falhas são parte do resultado
                rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
            w.writerow(rec)
            f.flush()
            if i % 25 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] {time.time() - t_start:.0f}s", flush=True)
    if args.determinism_check and full:
        diffs = []
        for r in rows[: args.determinism_check]:
            try:
                img = Image.open(args.dataset / r.path).convert("RGB")
                first = {n: None for n in names}
                res = score_image(img, detectors=dets)
                for n, d in dets.items():
                    first[n] = d.last_score
                res2 = score_image(img, detectors=dets)
                diffs.append(max([abs(res["score"] - res2["score"])] + [abs(first[n] - dets[n].last_score) for n in names]))
            except Exception:
                pass
        meta["determinism_max_abs_diff"] = max(diffs) if diffs else None
    meta["n_images"], meta["wall_seconds"] = len(rows), round(time.time() - t_start, 1)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"resultados em {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
