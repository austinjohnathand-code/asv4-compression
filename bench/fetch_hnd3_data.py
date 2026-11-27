#!/usr/bin/env python3
"""
Download and verify SHA-256 integrity of all public datasets used in HND3 benchmarks.

Usage:
    python3 bench/fetch_hnd3_data.py          # fetch missing files + verify SHA-256
    python3 bench/fetch_hnd3_data.py --check  # verify existing phase0_data/ files only
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "phase0_data"

# Expected SHA-256 checksums of the exact benchmark inputs in phase0_data/
MANIFEST = {
    "Indian_pines.mat": {
        "sha256": "ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939",
        "urls": [
            "https://github.com/gokriznastic/HybridSN/raw/master/data/Indian_pines.mat",
            "https://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat",
        ],
    },
    "Salinas.mat": {
        "sha256": "5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d",
        "urls": [
            "https://github.com/gokriznastic/HybridSN/raw/master/data/Salinas.mat",
            "https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat",
        ],
    },
    "PaviaU.mat": {
        "sha256": "28447fa87f7a5797845e9a189c0da85e23b1d06a4ba7361e5ff44efbf834d2fb",
        "urls": [
            "https://github.com/gokriznastic/HybridSN/raw/master/data/PaviaU.mat",
            "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        ],
    },
    "cells3d.tif": {
        "sha256": "afc7c7d80d38bfde09788b4064ac1e64ec14e88454ab785ebdc8dbba5ca3b222",
        "urls": [
            "https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif",
        ],
    },
    "brain.tiff": {
        "sha256": "bcdbaf424fbad7b1fb0f855f608c68e5a838f35affc323ff04ea17f678eef5c6",
        "urls": [
            "https://gitlab.com/scikit-image/data/-/raw/master/brain.tiff",
        ],
    },
    "aviris_holdout.npy": {
        "sha256": "94eb7f716a2dbfd22ef28cc7c4ec6862d019e9fe0d4e386da1f01712110de0d6",
        "parquet_url": "https://huggingface.co/datasets/danjacobellis/aviris_1k_val/resolve/main/data/validation-00003-of-00066.parquet",
        "note": (
            "Row 0 extracted from validation-00003-of-00066.parquet, shape (224, 745, 398). "
            "The parquet stores raw NASA AVIRIS big-endian (>i2) radiance bytes decoded as little-endian (<i2); "
            "byte-swapping (.astype('<i2').byteswap()) restores authentic radiance in [-50, 2033]."
        ),
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(urls, dst: Path):
    last_err = None
    for url in urls:
        try:
            print(f"Downloading {dst.name} from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "asv4-hnd3-bench/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
            return
        except Exception as e:
            last_err = e
            print(f"  mirror failed ({e}), trying next...")
    raise RuntimeError(f"Failed to download {dst.name}: {last_err}")


def fetch_aviris_holdout(dst: Path, url: str):
    import pyarrow.parquet as pq
    tmp_pq = dst.with_suffix(".parquet.tmp")
    download_file([url], tmp_pq)
    print(f"  raw parquet SHA-256: {sha256_file(tmp_pq)}")
    tbl = pq.read_table(tmp_pq)
    row0 = tbl.to_pylist()[0]
    arr = np.asarray(row0["data"] if "data" in row0 else row0[list(row0.keys())[0]])
    if arr.ndim == 3 and arr.shape[2] == 224:
        arr = np.transpose(arr, (2, 0, 1))
    arr_be_fixed = np.ascontiguousarray(arr.astype("<i2").byteswap(), dtype="<i4")
    np.save(dst, arr_be_fixed)
    tmp_pq.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Verify existing files only")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, meta in MANIFEST.items():
        path = DATA_DIR / name
        if not path.exists():
            if args.check:
                print(f"[MISSING] {name}")
                ok = False
                continue
            if "parquet_url" in meta:
                fetch_aviris_holdout(path, meta["parquet_url"])
            else:
                download_file(meta["urls"], path)
        got = sha256_file(path)
        exp = meta["sha256"]
        if got == exp:
            print(f"[OK] {name:<20} {path.stat().st_size:>12,} B  sha256={got[:16]}...")
        else:
            print(f"[SHA MISMATCH] {name}: got {got}, expected {exp}")
            ok = False
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
