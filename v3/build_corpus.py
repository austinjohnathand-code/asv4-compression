"""Pre-Registered Benchmark Corpus Builder for ASV3 (v3 Spec).

Builds and locks all 10 pre-registered files in `v3/corpus/` before any
Stage 1-3 tuning:

Primary Target Datasets:
  1. target_numeric_f32.bin      (256 MB) - IEEE-754 float32 3D scientific field
  2. target_struct64.bin         (256 MB) - 64-byte fixed-width binary telemetry structs
  3. target_sqlite.db            (256 MB) - Real SQLite 3 database (4096 B B-tree pages)
  4. target_mixed_structured.bin (256 MB) - Non-4MB-aligned mixed structured + real regimes
  5. target_snapshots.bin        (256 MB) - Versioned 8 MB binary snapshots (32 MB stride, 1.5% in-place edits)

Negative Controls (must pass <= 0.15% worst-case tax vs zstd -19 --long=27):
  6. neg_enwik9.xml             (1000 MB) - Official enwik9 (1 GB Wikipedia XML)
  7. neg_source_tar.tar          (256 MB) - Real uncompressed .tar of source code & docs

Scorer-Adversarial Controls (must pass <= 0.15% worst-case tax):
  8. adv_binaries.bin            (160 MB) - Real concatenated ARM64 Mach-O executables/dylibs
  9. adv_compressed_media.tar    (128 MB) - Already-compressed files + periodic tar/container headers
 10. adv_random.bin               (64 MB) - Pure os.urandom(64 MB)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import zstandard as zstd

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
MB = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * MB)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_numeric_f32(path: Path, size_mb: int = 256) -> None:
    if path.exists() and path.stat().st_size == size_mb * MB:
        return
    print(f"Building {path.name} ({size_mb} MB)...")
    rng = np.random.default_rng(1001)
    # 3D climate/atmospheric float32 grid: slices of (Y=256, X=1024) = 1 MB per slice
    y_dim, x_dim = 256, 1024
    yy, xx = np.meshgrid(
        np.linspace(-np.pi, np.pi, y_dim, dtype=np.float32),
        np.linspace(-np.pi, np.pi, x_dim, dtype=np.float32),
        indexing="ij",
    )
    with open(path, "wb") as f:
        for t in range(size_mb):
            phase = np.float32(t * 0.04)
            # Smooth multi-scale geophysical field (realistic IEEE-754 exponent concentration + smooth mantissas)
            field = (
                285.0
                + 18.0 * np.cos(yy * 0.8)
                + 6.5 * np.sin(2.0 * xx + phase) * np.cos(1.5 * yy - 0.5 * phase)
                + 2.2 * np.cos(5.0 * xx - 3.0 * yy + 1.3 * phase)
                + rng.normal(0.0, 0.015, size=(y_dim, x_dim)).astype(np.float32)
            ).astype(np.float32)
            f.write(field.tobytes())


def make_struct64(path: Path, size_mb: int = 256) -> None:
    if path.exists() and path.stat().st_size == size_mb * MB:
        return
    print(f"Building {path.name} ({size_mb} MB)...")
    rng = np.random.default_rng(2002)
    rec_size = 64
    chunk_mb = 16
    n_rec_chunk = (chunk_mb * MB) // rec_size

    ts_cur = np.int64(1_730_000_000_000_000)
    seq_cur = np.uint32(0)
    cnt_cur = np.array([10000, 20000, 30000, 40000], dtype=np.int32)
    tag = np.frombuffer(b"TICK_V3\x00", dtype=np.uint8)

    with open(path, "wb") as f:
        for c in range(size_mb // chunk_mb):
            buf = np.zeros((n_rec_chunk, rec_size), dtype=np.uint8)
            dt = 1000 + rng.integers(-6, 7, size=n_rec_chunk, dtype=np.int64)
            ts = ts_cur + np.cumsum(dt, dtype=np.int64)
            ts_cur = int(ts[-1])
            buf[:, 0:8] = ts.view(np.uint8).reshape(n_rec_chunk, 8)

            seq = np.arange(seq_cur, seq_cur + n_rec_chunk, dtype=np.uint32)
            seq_cur = np.uint32(seq_cur + n_rec_chunk)
            buf[:, 8:12] = seq.view(np.uint8).reshape(n_rec_chunk, 4)

            sym = (np.arange(n_rec_chunk, dtype=np.uint16) & 63) + 1000
            flags = np.full(n_rec_chunk, 0x0104, dtype=np.uint16)
            flags[rng.random(n_rec_chunk) < 0.015] ^= 0x0020
            buf[:, 12:14] = sym.view(np.uint8).reshape(n_rec_chunk, 2)
            buf[:, 14:16] = flags.view(np.uint8).reshape(n_rec_chunk, 2)

            t = np.linspace(c * 10.0, (c + 1) * 10.0, n_rec_chunk, dtype=np.float32)
            for k in range(6):
                walk = np.cumsum(rng.normal(0.0, 0.0025, size=n_rec_chunk)).astype(np.float32)
                val = (100.0 + 15.0 * k + 3.5 * np.sin((0.4 + 0.15 * k) * t + k) + walk).astype(np.float32)
                buf[:, 16 + 4 * k:20 + 4 * k] = val.view(np.uint8).reshape(n_rec_chunk, 4)

            for k in range(4):
                inc = rng.integers(0, 4 + k, size=n_rec_chunk, dtype=np.int32)
                cnt = cnt_cur[k] + np.cumsum(inc, dtype=np.int32)
                cnt_cur[k] = int(cnt[-1])
                buf[:, 40 + 4 * k:44 + 4 * k] = cnt.view(np.uint8).reshape(n_rec_chunk, 4)

            buf[:, 56:64] = tag
            buf[:, 63] = (seq & 0xFF).astype(np.uint8)
            f.write(buf.tobytes())


def make_sqlite_db(path: Path, target_mb: int = 256) -> None:
    if path.exists() and path.stat().st_size >= (target_mb - 8) * MB:
        return
    if path.exists():
        path.unlink()
    print(f"Building {path.name} (~{target_mb} MB real SQLite 3 DB)...")
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("PRAGMA page_size = 4096")
    cur.execute("PRAGMA journal_mode = OFF")
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute(
        """
        CREATE TABLE telemetry (
            id INTEGER PRIMARY KEY,
            ts INTEGER NOT NULL,
            device_id INTEGER NOT NULL,
            region TEXT NOT NULL,
            temp REAL NOT NULL,
            pressure REAL NOT NULL,
            voltage REAL NOT NULL,
            status INTEGER NOT NULL,
            payload BLOB NOT NULL
        )
        """
    )
    rng = np.random.default_rng(3003)
    regions = ("us-west-2", "us-east-1", "eu-central-1", "ap-northeast-1")
    batch_sz = 25000
    row_id = 1
    ts = 1_730_000_000_000
    target_bytes = target_mb * MB

    while True:
        dts = rng.integers(5, 25, size=batch_sz)
        devs = rng.integers(1000, 1128, size=batch_sz)
        ridx = rng.integers(0, 4, size=batch_sz)
        temps = 22.0 + np.sin(np.linspace(0, 20, batch_sz)) * 4.0 + rng.normal(0, 0.05, size=batch_sz)
        press = 1013.2 + rng.normal(0, 0.4, size=batch_sz)
        volts = 3.30 + rng.normal(0, 0.005, size=batch_sz)
        stats = np.where(rng.random(batch_sz) < 0.98, 200, 503)

        rows = []
        for i in range(batch_sz):
            ts += int(dts[i])
            dev = int(devs[i])
            # 24-byte structured binary sensor payload inside each row
            blob = (
                int(ts & 0xFFFFFFFF).to_bytes(4, "little")
                + dev.to_bytes(4, "little")
                + int(row_id & 0xFFFFFFFF).to_bytes(4, "little")
                + b"SENS\x01\x00\x02\x00"
                + (i & 0xFFFF).to_bytes(4, "little")
            )
            rows.append((
                row_id,
                ts,
                dev,
                regions[int(ridx[i])],
                round(float(temps[i]), 3),
                round(float(press[i]), 2),
                round(float(volts[i]), 4),
                int(stats[i]),
                blob,
            ))
            row_id += 1

        cur.executemany(
            "INSERT INTO telemetry VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        if path.stat().st_size >= target_bytes:
            break

    cur.execute("CREATE INDEX idx_dev_ts ON telemetry(device_id, ts)")
    conn.commit()
    conn.close()
    print(f"  -> {path.name}: {path.stat().st_size / MB:.1f} MB")


def make_source_tar(path: Path, target_mb: int = 256) -> None:
    if path.exists() and path.stat().st_size >= (target_mb - 4) * MB:
        return
    print(f"Building {path.name} (~{target_mb} MB real uncompressed source .tar)...")
    src_root = Path("/opt/anaconda3/lib/python3.11/site-packages")
    exts = {".py", ".c", ".h", ".cpp", ".hpp", ".md", ".rst", ".txt", ".json", ".html", ".css", ".js", ".xml", ".yml", ".yaml", ".csv"}
    files: List[Path] = []
    for root, _, fnames in os.walk(src_root):
        for fn in sorted(fnames):
            p = Path(root) / fn
            if p.suffix.lower() in exts and not p.is_symlink():
                try:
                    sz = p.stat().st_size
                    if 64 <= sz <= 4 * MB:
                        files.append(p)
                except OSError:
                    pass
    files.sort()
    target_bytes = target_mb * MB
    written = 0
    with tarfile.open(path, "w") as tf:
        for p in files:
            try:
                sz = p.stat().st_size
                arcname = str(p.relative_to(src_root))
                tf.add(p, arcname=arcname, recursive=False)
                written += sz + 512
                if written >= target_bytes:
                    break
            except OSError:
                continue
    print(f"  -> {path.name}: {path.stat().st_size / MB:.1f} MB")


def make_binaries(path: Path, target_mb: int = 160) -> None:
    if path.exists() and path.stat().st_size == target_mb * MB:
        return
    print(f"Building {path.name} ({target_mb} MB real concatenated ARM64 Mach-O binaries)...")
    roots = [Path("/opt/anaconda3/lib"), Path("/opt/anaconda3/bin")]
    bins: List[Path] = []
    seen_inodes = set()
    for r in roots:
        for root, _, fnames in os.walk(r):
            for fn in sorted(fnames):
                p = Path(root) / fn
                if p.is_symlink():
                    continue
                if p.suffix in (".dylib", ".so") or r.name == "bin":
                    try:
                        st = p.stat()
                        if st.st_ino in seen_inodes or st.st_size < 16384:
                            continue
                        with open(p, "rb") as f:
                            magic = f.read(4)
                        # Mach-O 64-bit little-endian or fat binary magic
                        if magic in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf"):
                            seen_inodes.add(st.st_ino)
                            bins.append(p)
                    except OSError:
                        continue
    bins.sort()
    target_bytes = target_mb * MB
    written = 0
    with open(path, "wb") as out:
        for p in bins:
            data = p.read_bytes()
            rem = target_bytes - written
            if rem <= 0:
                break
            chunk = data[:rem]
            out.write(chunk)
            written += len(chunk)
    print(f"  -> {path.name}: {path.stat().st_size / MB:.1f} MB from {len(bins)} Mach-O files")


def make_compressed_media_tar(path: Path, target_mb: int = 128) -> None:
    if path.exists() and path.stat().st_size >= (target_mb - 2) * MB:
        return
    print(f"Building {path.name} (~{target_mb} MB compressed media/archives w/ periodic headers)...")
    # Gather real compressed assets (.png, .jpg, .gz, .zip, .whl, .bz2, .xz) from anaconda,
    # then top up to target_mb with zstd-compressed chunks wrapped in periodic 4 KB container frames
    # + 512 B POSIX tar headers (classic scorer-adversarial periodic header over high-entropy payload).
    src_root = Path("/opt/anaconda3")
    exts = {".png", ".jpg", ".jpeg", ".gif", ".gz", ".zip", ".whl", ".bz2", ".xz", ".pdf"}
    assets: List[Path] = []
    for root, _, fnames in os.walk(src_root):
        for fn in sorted(fnames):
            p = Path(root) / fn
            if p.suffix.lower() in exts and not p.is_symlink():
                try:
                    sz = p.stat().st_size
                    if 1024 <= sz <= 8 * MB:
                        assets.append(p)
                except OSError:
                    pass
        if len(assets) >= 4000:
            break
    assets.sort()

    target_bytes = target_mb * MB
    written = 0
    rng = np.random.default_rng(5005)
    c_zstd = zstd.ZstdCompressor(level=6)

    with tarfile.open(path, "w") as tf:
        for i, p in enumerate(assets):
            try:
                sz = p.stat().st_size
                tf.add(p, arcname=f"real/{i:05d}_{p.name}", recursive=False)
                written += sz + 512
                if written >= target_bytes // 2:
                    break
            except OSError:
                continue

        # Add MP4/media-like container entries: repeating 16-byte frame headers every 4096 bytes
        # surrounding incompressible compressed payloads (adversarial for M(4096) lag detector!)
        idx = 0
        while written < target_bytes:
            # Build 256 KB container object with periodic 16-byte sync headers every 4096 B
            n_frames = 64
            payload = bytearray()
            for fr in range(n_frames):
                hdr = b"MDAT\x00\x00\x10\x00" + idx.to_bytes(4, "big") + fr.to_bytes(4, "big")
                raw_chunk = rng.integers(0, 256, size=4080, dtype=np.uint8).tobytes()
                # Make it truly compressed bytes
                comp_chunk = c_zstd.compress(raw_chunk)[:4080]
                if len(comp_chunk) < 4080:
                    comp_chunk += rng.bytes(4080 - len(comp_chunk))
                payload.extend(hdr)
                payload.extend(comp_chunk)
            info = tarfile.TarInfo(name=f"media/clip_{idx:05d}.mp4")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
            written += len(payload) + 512
            idx += 1
    print(f"  -> {path.name}: {path.stat().st_size / MB:.1f} MB")


def make_random(path: Path, size_mb: int = 64) -> None:
    if path.exists() and path.stat().st_size == size_mb * MB:
        return
    print(f"Building {path.name} ({size_mb} MB os.urandom)...")
    with open(path, "wb") as f:
        for _ in range(size_mb):
            f.write(os.urandom(MB))


def make_snapshots(path: Path, size_mb: int = 256) -> None:
    if path.exists() and path.stat().st_size == size_mb * MB:
        return
    print(f"Building {path.name} ({size_mb} MB versioned 8 MB snapshots @ 32 MB stride)...")
    rng = np.random.default_rng(6006)
    snap_sz = 8 * MB
    # 4 distinct 8 MB base binary snapshots (semi-random 4KB pages with structured headers)
    bases = []
    for k in range(4):
        arr = rng.integers(0, 256, size=snap_sz, dtype=np.uint8)
        pages = arr.reshape(-1, 4096)
        pages[:, 0:8] = np.frombuffer(f"SNAP_V{k}_".encode("ascii"), dtype=np.uint8)
        pages[:, 8:12] = np.arange(pages.shape[0], dtype=np.uint32).view(np.uint8).reshape(-1, 4)
        pages[:, 12:32] = 0x00
        bases.append(arr)

    # 8 generations x 4 snapshots = 32 x 8 MB = 256 MB
    # Each successive generation mutates 1.5% of random bytes in-place (small additive deltas)
    # Distance between consecutive versions of snapshot k = 32 MB (inside 128 MB window!)
    current = [b.copy() for b in bases]
    with open(path, "wb") as f:
        for gen in range(size_mb // 32):
            for k in range(4):
                if gen > 0:
                    arr = current[k]
                    n_mut = int(snap_sz * 0.015)
                    idx = rng.integers(0, snap_sz, size=n_mut)
                    delta = rng.integers(1, 17, size=n_mut, dtype=np.uint8)
                    arr[idx] = ((arr[idx].astype(np.uint16) + delta) & 0xFF).astype(np.uint8)
                f.write(current[k].tobytes())


def make_mixed_structured(
    path: Path,
    f32_path: Path,
    struct_path: Path,
    tar_path: Path,
    enwik_path: Path,
    bin_path: Path,
) -> None:
    if path.exists() and path.stat().st_size == 256 * MB:
        return
    print(f"Building {path.name} (256 MB non-4MB-aligned mixed structured + real regimes)...")
    # Deliberately choose non-4MB-aligned region sizes (still record-aligned internally)
    # so fixed 4 MB blocks straddle transitions and drift phase, testing Stage 1 DP segmentation!
    # Sum of lengths = 256 MB exactly (268,435,456 bytes).
    lens = [
        29 * MB + 123_456,   # float32 scientific grid
        31 * MB + 512_000,   # real source tar
        34 * MB + 262_144,   # 64-byte binary structs
        28 * MB + 700_000,   # real enwik9 XML
        33 * MB + 300_000,   # int16 3D scientific field
        30 * MB + 400_000,   # real ARM64 binaries
        35 * MB + 600_000,   # 64-byte binary structs (second region)
    ]
    rem = 256 * MB - sum(lens)
    lens.append(rem)         # float32 scientific grid (second region)
    assert sum(lens) == 256 * MB

    # Quick int16 3D field for segment 4
    rng = np.random.default_rng(7007)
    z_d, y_d, x_d = 64, 64, 64
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, z_d, dtype=np.float32),
        np.linspace(-1, 1, y_d, dtype=np.float32),
        np.linspace(-1, 1, x_d, dtype=np.float32),
        indexing="ij",
    )
    base_v = 12000.0 * np.exp(-2.0 * (xx ** 2 + yy ** 2 + 0.5 * zz ** 2))
    vol_chunks = []
    need_v = lens[4]
    while sum(len(x) for x in vol_chunks) < need_v:
        v = np.clip(base_v + rng.normal(0, 3, size=(z_d, y_d, x_d)), -30000, 30000).astype(np.int16)
        vol_chunks.append(v.tobytes())
    vol_bytes = b"".join(vol_chunks)[:need_v]

    def read_head(p: Path, n: int, offset: int = 0) -> bytes:
        with open(p, "rb") as f:
            f.seek(offset)
            return f.read(n)

    with open(path, "wb") as out:
        out.write(read_head(f32_path, lens[0], 0))
        out.write(read_head(tar_path, lens[1], 0))
        out.write(read_head(struct_path, lens[2], 0))
        out.write(read_head(enwik_path, lens[3], 0))
        out.write(vol_bytes)
        out.write(read_head(bin_path, lens[5], 0))
        out.write(read_head(struct_path, lens[6], 64 * MB))
        out.write(read_head(f32_path, lens[7], 64 * MB))
    assert path.stat().st_size == 256 * MB


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    enwik_zip = CORPUS_DIR / "enwik9.zip"
    enwik_xml = CORPUS_DIR / "neg_enwik9.xml"
    if not enwik_xml.exists():
        if not enwik_zip.exists():
            raise FileNotFoundError(
                f"Expected {enwik_zip} to be downloaded first (https://mattmahoney.net/dc/enwik9.zip)"
            )
        print(f"Extracting {enwik_zip.name} -> {enwik_xml.name} ...")
        with zipfile.ZipFile(enwik_zip, "r") as zf:
            with zf.open("enwik9") as src, open(enwik_xml, "wb") as dst:
                while True:
                    buf = src.read(8 * MB)
                    if not buf:
                        break
                    dst.write(buf)
        enwik_zip.unlink()

    f32_p = CORPUS_DIR / "target_numeric_f32.bin"
    s64_p = CORPUS_DIR / "target_struct64.bin"
    sql_p = CORPUS_DIR / "target_sqlite.db"
    snp_p = CORPUS_DIR / "target_snapshots.bin"
    tar_p = CORPUS_DIR / "neg_source_tar.tar"
    bin_p = CORPUS_DIR / "adv_binaries.bin"
    med_p = CORPUS_DIR / "adv_compressed_media.tar"
    rnd_p = CORPUS_DIR / "adv_random.bin"
    mix_p = CORPUS_DIR / "target_mixed_structured.bin"

    make_numeric_f32(f32_p, 256)
    make_struct64(s64_p, 256)
    make_sqlite_db(sql_p, 256)
    make_snapshots(snp_p, 256)
    make_source_tar(tar_p, 256)
    make_binaries(bin_p, 160)
    make_compressed_media_tar(med_p, 128)
    make_random(rnd_p, 64)
    make_mixed_structured(mix_p, f32_p, s64_p, tar_p, enwik_xml, bin_p)

    # Lock pre-registered manifest with sizes and SHA-256
    manifest: List[Dict] = []
    ordered = [
        ("target_numeric_f32.bin", "target", "f32_le_transpose_xor"),
        ("target_struct64.bin", "target", "struct64_transpose_xor"),
        ("target_sqlite.db", "target", "sqlite_page4096"),
        ("target_mixed_structured.bin", "target_mixed", "mixed_oracle"),
        ("target_snapshots.bin", "target_snapshot", "snapshot_delta"),
        ("neg_enwik9.xml", "negative", "identity"),
        ("neg_source_tar.tar", "negative", "identity"),
        ("adv_binaries.bin", "adversarial", "identity"),
        ("adv_compressed_media.tar", "adversarial", "identity"),
        ("adv_random.bin", "adversarial", "identity"),
    ]
    print("\nPre-Registered Benchmark Corpus Manifest:")
    for fname, cat, oracle_hint in ordered:
        p = CORPUS_DIR / fname
        sz = p.stat().st_size
        h = sha256_file(p)
        manifest.append({
            "name": fname,
            "category": cat,
            "bytes": sz,
            "mb": round(sz / MB, 2),
            "sha256": h,
            "oracle_hint": oracle_hint,
        })
        print(f"  {fname:<30} {cat:<16} {sz/MB:7.2f} MB  sha256={h[:16]}...")

    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nLocked manifest at {CORPUS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
