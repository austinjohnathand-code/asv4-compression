"""Download and assemble the 6 real public benchmark datasets in v3/corpus_real/.

Datasets created (~2.15 GiB total; requires ~2.5 GiB free disk space):
  1. real_gpt2_f32.safetensors      (522.7 MB) — OpenAI GPT-2 production FP32 weights (HuggingFace)
  2. real_smollm2_bf16.safetensors  (256.6 MB) — HuggingFace SmolLM2-135M production BF16 weights
  3. real_hubble_eagle_f32.fits     ( 29.5 MB) — 3 NASA/ESA Hubble WFPC2 Eagle Nebula 1600x1600 >f4 FITS frames
  4. real_nyc_taxi_struct104.bin    (256.0 MB) — 2,581,110 real NYC TLC Yellow Taxi 2024-01/02 104B structs
  5. real_nyc_taxi_columnar.arrow   (803.1 MB) — 5,972,150 real NYC TLC trips uncompressed Arrow IPC stream
  6. real_mixed_public.bin          (283.9 MB) — Non-4MB-aligned mix of real GPT-2, source.tar, taxi, enwik9, Arrow

Prerequisites:
  - Run `python3 v3/build_corpus.py` first (needed for `neg_source_tar.tar` and `neg_enwik9.xml`
    slices used inside `real_mixed_public.bin`).
  - Python packages: numpy, pandas, pyarrow.
"""

from __future__ import annotations

import shutil
import struct
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

V3_DIR = Path(__file__).resolve().parent
CORPUS_DIR = V3_DIR / "corpus"
REAL_DIR = V3_DIR / "corpus_real"
RASTERS_DIR = REAL_DIR / "rasters"
MB = 1024 * 1024


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [SKIP] {dest.name} ({dest.stat().st_size / MB:.2f} MB) already exists")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [GET ] {url} -> {dest.name}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "asv4-benchmark/4.1"})
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    tmp.replace(dest)
    print(f"  [DONE] {dest.name} ({dest.stat().st_size / MB:.2f} MB)")


def build_hubble_fits() -> None:
    out_path = REAL_DIR / "real_hubble_eagle_f32.fits"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [SKIP] {out_path.name} ({out_path.stat().st_size / MB:.2f} MB) already exists")
        return
    RASTERS_DIR.mkdir(parents=True, exist_ok=True)
    fits_files = ["502nmos.fits", "656nmos.fits", "673nmos.fits"]
    base_url = "https://cdn.spacetelescope.org/archives/fitsimages/data/fits/opo9544/"
    local_paths: List[Path] = []
    for fname in fits_files:
        p = RASTERS_DIR / fname
        download(base_url + fname, p)
        local_paths.append(p)
    with out_path.open("wb") as out:
        for p in local_paths:
            out.write(p.read_bytes())
    print(f"  [DONE] {out_path.name} ({out_path.stat().st_size / MB:.2f} MB)")


def build_nyc_taxi() -> None:
    p1 = REAL_DIR / "yellow_2024_01.parquet"
    p2 = REAL_DIR / "yellow_2024_02.parquet"
    download("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet", p1)
    download("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet", p2)

    struct_path = REAL_DIR / "real_nyc_taxi_struct104.bin"
    arrow_path = REAL_DIR / "real_nyc_taxi_columnar.arrow"

    if not arrow_path.exists() or not struct_path.exists():
        t1 = pq.read_table(p1)
        t2 = pq.read_table(p2)
        tbl = pa.concat_tables([t1, t2])

        if not arrow_path.exists():
            print(f"  [MAKE] {arrow_path.name} ({tbl.num_rows:,} rows uncompressed Arrow IPC)...")
            with pa.OSFile(str(arrow_path), "wb") as sink:
                with ipc.new_stream(sink, tbl.schema) as writer:
                    for batch in tbl.to_batches(max_chunksize=65536):
                        writer.write_batch(batch)
            print(f"  [DONE] {arrow_path.name} ({arrow_path.stat().st_size / MB:.2f} MB)")

        if not struct_path.exists():
            print(f"  [MAKE] {struct_path.name} (256.0 MB fixed-width 104B structs)...")
            target_rows = (256 * MB) // 104
            df = tbl.slice(0, target_rows).to_pandas()
            dt = np.dtype([
                ("tpep_pickup", "<i8"), ("tpep_dropoff", "<i8"),
                ("vendor_id", "<i4"), ("pu_loc", "<i4"), ("do_loc", "<i4"), ("payment_type", "<i4"),
                ("passenger_cnt", "<f4"), ("ratecode_id", "<f4"),
                ("trip_dist", "<f8"), ("fare_amt", "<f8"), ("extra", "<f8"), ("mta_tax", "<f8"),
                ("tip_amt", "<f8"), ("tolls_amt", "<f8"), ("imp_surcharge", "<f8"), ("total_amt", "<f8"),
            ])
            assert dt.itemsize == 104
            arr = np.empty(len(df), dtype=dt)
            arr["tpep_pickup"] = df["tpep_pickup_datetime"].astype("int64").to_numpy()
            arr["tpep_dropoff"] = df["tpep_dropoff_datetime"].astype("int64").to_numpy()
            arr["vendor_id"] = df["VendorID"].fillna(0).astype(np.int32).to_numpy()
            arr["pu_loc"] = df["PULocationID"].fillna(0).astype(np.int32).to_numpy()
            arr["do_loc"] = df["DOLocationID"].fillna(0).astype(np.int32).to_numpy()
            arr["payment_type"] = df["payment_type"].fillna(0).astype(np.int32).to_numpy()
            arr["passenger_cnt"] = df["passenger_count"].fillna(0).astype(np.float32).to_numpy()
            arr["ratecode_id"] = df["RatecodeID"].fillna(0).astype(np.float32).to_numpy()
            arr["trip_dist"] = df["trip_distance"].fillna(0).astype(np.float64).to_numpy()
            arr["fare_amt"] = df["fare_amount"].fillna(0).astype(np.float64).to_numpy()
            arr["extra"] = df["extra"].fillna(0).astype(np.float64).to_numpy()
            arr["mta_tax"] = df["mta_tax"].fillna(0).astype(np.float64).to_numpy()
            arr["tip_amt"] = df["tip_amount"].fillna(0).astype(np.float64).to_numpy()
            arr["tolls_amt"] = df["tolls_amount"].fillna(0).astype(np.float64).to_numpy()
            arr["imp_surcharge"] = df["improvement_surcharge"].fillna(0).astype(np.float64).to_numpy()
            arr["total_amt"] = df["total_amount"].fillna(0).astype(np.float64).to_numpy()
            struct_path.write_bytes(arr.tobytes())
            print(f"  [DONE] {struct_path.name} ({struct_path.stat().st_size / MB:.2f} MB)")


def build_mixed_public() -> None:
    out_path = REAL_DIR / "real_mixed_public.bin"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [SKIP] {out_path.name} ({out_path.stat().st_size / MB:.2f} MB) already exists")
        return
    src_tar = CORPUS_DIR / "neg_source_tar.tar"
    enwik9 = CORPUS_DIR / "neg_enwik9.xml"
    if not src_tar.exists() or not enwik9.exists():
        raise FileNotFoundError(
            "Run `python3 v3/build_corpus.py` first so neg_source_tar.tar and neg_enwik9.xml exist."
        )
    parts = [
        (REAL_DIR / "real_gpt2_f32.safetensors", 67_213_312),    # ~64.1 MB (non-4MB-aligned)
        (src_tar, 48_000_000),                                   # ~45.8 MB
        (REAL_DIR / "real_nyc_taxi_struct104.bin", 67_320_000),  # ~64.2 MB
        (enwik9, 48_000_000),                                    # ~45.8 MB
        (REAL_DIR / "real_nyc_taxi_columnar.arrow", 67_124_736), # ~64.0 MB
    ]
    with out_path.open("wb") as out:
        for p, nbytes in parts:
            with p.open("rb") as f:
                out.write(f.read(nbytes))
    print(f"  [DONE] {out_path.name} ({out_path.stat().st_size / MB:.2f} MB)")


def main() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading & assembling real public benchmark dumps into v3/corpus_real/...")
    download(
        "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
        REAL_DIR / "real_gpt2_f32.safetensors",
    )
    download(
        "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors",
        REAL_DIR / "real_smollm2_bf16.safetensors",
    )
    build_hubble_fits()
    build_nyc_taxi()
    build_mixed_public()
    print("\nAll 6 real public datasets are ready in v3/corpus_real/!")


if __name__ == "__main__":
    main()
