#!/usr/bin/env python3
"""Valida um run (results/<id>): sem erros, scores em [0,1], ensemble == média ponderada original, determinismo."""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from evalkit.analysis import load_raw  # noqa: E402
from evalkit.sanity import check  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=pathlib.Path)
    args = ap.parse_args(argv)
    rows, _, _ = load_raw(args.run)
    meta = json.loads((args.run / "meta.json").read_text())
    problems = check(rows, meta)
    for p in problems:
        print("PROBLEMA:", p)
    print("pipeline coerente" if not problems else f"{len(problems)} problema(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
