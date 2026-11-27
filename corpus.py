"""Deterministic synthetic structured datasets and 1 GB+ multi-regime file generator.

Generates:
  1. Five 32 MB synthetic datasets spanning distinct structural regimes:
       - tabular_telemetry    (64-byte multi-stride binary sensor/telemetry structs)
       - volumetric_3d        (3D int16 scientific voxel grids with shared motifs)
       - distant_records      (mutated snapshots recurring 8-24 MB apart across filler)
       - mixed_text_logs      (realistic JSON/HTTP application logs)
       - mixed_binary_entropy (sharp transitions between structs, voxels, logs, & PRNG)
  2. A 1.05 GB (1,056 MB) on-disk multi-regime file (`bench_data/corpus_1gb.bin`)
     for realistic multi-offset 32-64 MB slice experiments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
import numpy as np


def gen_tabular_telemetry(size_bytes: int, seed: int = 42) -> bytes:
    """64-byte multi-stride telemetry structs (timestamps, seq IDs, floats, counters)."""
    rng = np.random.default_rng(seed)
    rec_size = 64
    n_rec = size_bytes // rec_size
    buf = np.zeros((n_rec, rec_size), dtype=np.uint8)

    # Bytes 0..7: uint64 microsecond timestamps with steady step + tiny jitter
    dt = 1000 + rng.integers(-4, 5, size=n_rec, dtype=np.int64)
    ts = np.cumsum(dt, dtype=np.int64) + np.int64(1_700_000_000_000_000)
    buf[:, 0:8] = ts.view(np.uint8).reshape(n_rec, 8)

    # Bytes 8..11: uint32 sequence ID
    seq = np.arange(n_rec, dtype=np.uint32)
    buf[:, 8:12] = seq.view(np.uint8).reshape(n_rec, 4)

    # Bytes 12..15: uint16 device_id + uint16 flags
    dev = (np.arange(n_rec, dtype=np.uint16) & 31) + 100
    flags = np.full(n_rec, 0x0102, dtype=np.uint16)
    flips = rng.random(n_rec) < 0.01
    flags[flips] ^= 0x0010
    buf[:, 12:14] = dev.view(np.uint8).reshape(n_rec, 2)
    buf[:, 14:16] = flags.view(np.uint8).reshape(n_rec, 2)

    # Bytes 16..39: 6 smooth float32 sensor readings (sinusoid + small drift/noise)
    t = np.linspace(0.0, 200.0, n_rec, dtype=np.float32)
    for k in range(6):
        freq = 0.25 + 0.17 * k
        walk = np.cumsum(rng.normal(0.0, 0.002, size=n_rec)).astype(np.float32)
        val = (20.0 + 5.0 * k + 2.5 * np.sin(freq * t + k) + walk).astype(np.float32)
        buf[:, 16 + 4 * k:20 + 4 * k] = val.view(np.uint8).reshape(n_rec, 4)

    # Bytes 40..55: 4 int32 smoothly incrementing counters/accumulators
    for k in range(4):
        inc = rng.integers(0, 3 + k, size=n_rec, dtype=np.int32)
        cnt = np.cumsum(inc, dtype=np.int32) + (k + 1) * 10000
        buf[:, 40 + 4 * k:44 + 4 * k] = cnt.view(np.uint8).reshape(n_rec, 4)

    # Bytes 56..63: ASCII header tag + small rolling checksum byte
    tag = np.frombuffer(b"SENS_V2\x00", dtype=np.uint8)
    buf[:, 56:64] = tag
    buf[:, 63] = (seq & 0xFF).astype(np.uint8)

    raw = buf.tobytes()
    if len(raw) < size_bytes:
        raw += b"\x00" * (size_bytes - len(raw))
    return raw[:size_bytes]


def gen_volumetric_3d(size_bytes: int, seed: int = 101) -> bytes:
    """3D volumetric int16 scientific field chunks (Z=64, Y=64, X=64 = 512 KB/volume)."""
    rng = np.random.default_rng(seed)
    z_dim, y_dim, x_dim = 64, 64, 64
    vol_bytes = z_dim * y_dim * x_dim * 2  # 524,288 bytes (512 KB)
    n_vols = max(1, (size_bytes + vol_bytes - 1) // vol_bytes)

    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, z_dim, dtype=np.float32),
        np.linspace(-1.0, 1.0, y_dim, dtype=np.float32),
        np.linspace(-1.0, 1.0, x_dim, dtype=np.float32),
        indexing="ij",
    )
    # Shared 2D/3D calibration background field
    base_field = (
        12000.0 * np.exp(-2.2 * (xx ** 2 + yy ** 2 + 0.5 * zz ** 2))
        + 2500.0 * np.sin(3.0 * xx) * np.cos(2.5 * yy)
    )

    chunks = []
    for v in range(n_vols):
        cx = float(rng.uniform(-0.3, 0.3))
        cy = float(rng.uniform(-0.3, 0.3))
        amp = float(rng.uniform(0.85, 1.15))
        blob = 3000.0 * np.exp(-4.0 * ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - 0.1 * (v % 5)) ** 2))
        noise = rng.integers(-6, 7, size=(z_dim, y_dim, x_dim), dtype=np.int16)
        vol = np.clip(amp * base_field + blob, -30000, 30000).astype(np.int16) + noise
        chunks.append(vol.tobytes())

    return b"".join(chunks)[:size_bytes]


def gen_mixed_text_logs(size_bytes: int, seed: int = 202) -> bytes:
    """Realistic structured JSON/HTTP access logs with timestamps, IPs, routes, latencies."""
    rng = np.random.default_rng(seed)
    routes = [
        "/api/v2/ingest/telemetry",
        "/api/v2/query/timeseries",
        "/api/v2/models/predict",
        "/api/v2/users/profile",
        "/healthz",
        "/metrics/prometheus",
        "/api/v2/storage/compaction",
    ]
    methods = ["GET", "POST", "POST", "GET", "PUT", "DELETE"]
    statuses = [200, 200, 200, 200, 201, 204, 400, 404, 500]
    regions = ["us-west-2", "us-east-1", "eu-central-1", "ap-northeast-1"]
    agents = ["grpc-go/1.59", "python-httpx/0.27", "curl/8.4.0", "envoy-proxy/1.28"]

    # Build a rich 256 KB block of varied log lines, then evolve timestamps/IDs per block
    out = bytearray()
    ts_sec = 1732600000
    req_id = 1000000
    while len(out) < size_bytes:
        # Generate batch of 2000 lines using fast vectorized random indices
        batch = 2000
        r_idx = rng.integers(0, len(routes), size=batch)
        m_idx = rng.integers(0, len(methods), size=batch)
        s_idx = rng.integers(0, len(statuses), size=batch)
        reg_idx = rng.integers(0, len(regions), size=batch)
        ag_idx = rng.integers(0, len(agents), size=batch)
        ip3 = rng.integers(1, 254, size=batch)
        ip4 = rng.integers(1, 254, size=batch)
        lat = rng.integers(1, 450, size=batch)
        b_sent = rng.integers(128, 65536, size=batch)
        dt = rng.integers(0, 3, size=batch)

        lines = []
        for i in range(batch):
            ts_sec += int(dt[i])
            req_id += 1
            lines.append(
                f'{{"ts":{ts_sec},"req":"r-{req_id:09d}","region":"{regions[reg_idx[i]]}",'
                f'"ip":"10.24.{ip3[i]}.{ip4[i]}","method":"{methods[m_idx[i]]}",'
                f'"path":"{routes[r_idx[i]]}","status":{statuses[s_idx[i]]},'
                f'"ms":{lat[i]},"bytes":{b_sent[i]},"ua":"{agents[ag_idx[i]]}"}}\n'
            )
        out.extend("".join(lines).encode("ascii"))

    return bytes(out[:size_bytes])


def gen_distant_records(size_bytes: int, seed: int = 303) -> bytes:
    """Structured snapshot pages recurring at distant offsets (>8-24 MB apart) with sparse edits.

    Standard zstd (level 3, ~4 MB window) misses cross-distance matches, while
    Tree 2 (Similarity/Delta Forest) and zstd+LDM capture them.
    """
    rng = np.random.default_rng(seed)
    block_sz = 1024 * 1024  # 1 MB blocks

    # Create 4 rich semi-high-entropy base snapshots (realistic binary database/checkpoint pages)
    bases = []
    for b in range(4):
        # Mix deterministic structured pages with pseudo-random payload fields so
        # each individual snapshot is moderately hard to compress on its own (~1.3x),
        # while mutated copies of the same snapshot differ in only ~1-2% of bytes!
        arr = rng.integers(0, 256, size=block_sz, dtype=np.uint8)
        # Imprint 64-byte page/record headers every 256 bytes
        rec = arr.reshape(-1, 256)
        rec[:, 0:8] = np.frombuffer(f"PAGE_0{b}_".encode("ascii"), dtype=np.uint8)
        rec[:, 8:12] = np.arange(rec.shape[0], dtype=np.uint32).view(np.uint8).reshape(-1, 4)
        rec[:, 12:24] = 0x20
        bases.append(arr)

    def mutate_copy(base: np.ndarray, frac: float = 0.015) -> bytes:
        c = base.copy()
        n_mut = int(c.shape[0] * frac)
        idx = rng.integers(0, c.shape[0], size=n_mut)
        delta = rng.integers(1, 9, size=n_mut, dtype=np.uint8)
        c[idx] = ((c[idx].astype(np.uint16) + delta) & 0xFF).astype(np.uint8)
        return c.tobytes()

    # Layout over 32 MB:
    #   [0..4 MB]:  Base snapshots 0, 1, 2, 3
    #   [4..12 MB]: Unrelated filler (4 MB logs + 4 MB telemetry -> >8 MB separation!)
    #   [12..16 MB]: Mutated copies of snapshots 0, 1, 2, 3
    #   [16..24 MB]: Unrelated filler (4 MB voxels + 4 MB logs)
    #   [24..28 MB]: Second generation mutated copies of snapshots 0, 1, 2, 3
    #   [28..32 MB]: Mutated copies of snapshots 0, 1, 2, 3
    parts = [
        bases[0].tobytes(),
        bases[1].tobytes(),
        bases[2].tobytes(),
        bases[3].tobytes(),
        gen_mixed_text_logs(4 * block_sz, seed=seed + 10),
        gen_tabular_telemetry(4 * block_sz, seed=seed + 11),
        mutate_copy(bases[0], 0.012),
        mutate_copy(bases[1], 0.012),
        mutate_copy(bases[2], 0.015),
        mutate_copy(bases[3], 0.015),
        gen_volumetric_3d(4 * block_sz, seed=seed + 12),
        gen_mixed_text_logs(4 * block_sz, seed=seed + 13),
        mutate_copy(bases[0], 0.018),
        mutate_copy(bases[1], 0.018),
        mutate_copy(bases[2], 0.020),
        mutate_copy(bases[3], 0.020),
        mutate_copy(bases[0], 0.010),
        mutate_copy(bases[1], 0.010),
        mutate_copy(bases[2], 0.010),
        mutate_copy(bases[3], 0.010),
    ]
    raw = b"".join(parts)
    if len(raw) < size_bytes:
        # Repeat pattern if larger requested
        raw = (raw * ((size_bytes // len(raw)) + 1))[:size_bytes]
    return raw[:size_bytes]


def gen_mixed_binary_entropy(size_bytes: int, seed: int = 404) -> bytes:
    """Interleaved structured binary, volumetric, text logs, and incompressible PRNG."""
    rng = np.random.default_rng(seed)
    seg = max(512 * 1024, size_bytes // 8)
    t1 = gen_tabular_telemetry(seg, seed=seed + 1)
    r1 = rng.bytes(seg)  # High-entropy incompressible
    l1 = gen_mixed_text_logs(seg, seed=seed + 2)
    v1 = gen_volumetric_3d(seg, seed=seed + 3)
    r2 = rng.bytes(seg)  # High-entropy incompressible
    # Slightly mutated copy of t1 (16 MB later!)
    t1_arr = np.frombuffer(t1, dtype=np.uint8).copy()
    idx = rng.integers(0, t1_arr.shape[0], size=max(1, t1_arr.shape[0] // 100))
    t1_arr[idx] ^= 0x03
    t2 = t1_arr.tobytes()
    l2 = gen_mixed_text_logs(seg, seed=seed + 4)
    r3 = rng.bytes(seg)

    raw = b"".join([t1, r1, l1, v1, r2, t2, l2, r3])
    if len(raw) < size_bytes:
        raw += rng.bytes(size_bytes - len(raw))
    return raw[:size_bytes]


def build_synthetic_suite(size_mb: int = 32) -> Dict[str, bytes]:
    sz = size_mb * 1024 * 1024
    return {
        "tabular_telemetry": gen_tabular_telemetry(sz, seed=42),
        "volumetric_3d": gen_volumetric_3d(sz, seed=101),
        "distant_records": gen_distant_records(sz, seed=303),
        "mixed_text_logs": gen_mixed_text_logs(sz, seed=202),
        "mixed_binary_entropy": gen_mixed_binary_entropy(sz, seed=404),
    }


def ensure_1gb_corpus(path: Path, target_mb: int = 1056) -> Path:
    """Create a realistic 1.05 GB (1,056 MB) multi-regime file on disk if not present.

    Composed of 66 x 16 MB blocks combining tabular telemetry, volumetric 3D grids,
    distant recurring checkpoint pages, structured JSON logs, and random/encrypted blocks
    so slicing at arbitrary offsets yields realistic multi-regime data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = target_mb * 1024 * 1024
    if path.exists() and path.stat().st_size == target_bytes:
        return path

    print(f"Generating {target_mb} MB multi-regime corpus at {path} ...")
    rng = np.random.default_rng(777)
    block_mb = 16
    block_sz = block_mb * 1024 * 1024
    n_blocks = target_mb // block_mb

    # Pre-generate reusable 4 MB base pages for realistic cross-file recurrence
    page_a = rng.integers(0, 256, size=4 * 1024 * 1024, dtype=np.uint8)
    page_a.reshape(-1, 256)[:, 0:8] = np.frombuffer(b"SNAP_AA_", dtype=np.uint8)
    page_b = rng.integers(0, 256, size=4 * 1024 * 1024, dtype=np.uint8)
    page_b.reshape(-1, 256)[:, 0:8] = np.frombuffer(b"SNAP_BB_", dtype=np.uint8)

    with open(path, "wb") as f:
        for b in range(n_blocks):
            regime = b % 6
            if regime == 0:
                chunk = gen_tabular_telemetry(block_sz, seed=1000 + b)
            elif regime == 1:
                chunk = gen_volumetric_3d(block_sz, seed=2000 + b)
            elif regime == 2:
                # Mixed: 4 MB mutated page_a + 4 MB logs + 4 MB telemetry + 4 MB mutated page_b
                pa = page_a.copy()
                idx = rng.integers(0, pa.shape[0], size=pa.shape[0] // 80)
                pa[idx] ^= 0x05
                pb = page_b.copy()
                idx2 = rng.integers(0, pb.shape[0], size=pb.shape[0] // 80)
                pb[idx2] ^= 0x07
                chunk = b"".join([
                    pa.tobytes(),
                    gen_mixed_text_logs(4 * 1024 * 1024, seed=3000 + b),
                    gen_tabular_telemetry(4 * 1024 * 1024, seed=3500 + b),
                    pb.tobytes(),
                ])
            elif regime == 3:
                chunk = gen_mixed_text_logs(block_sz, seed=4000 + b)
            elif regime == 4:
                chunk = gen_mixed_binary_entropy(block_sz, seed=5000 + b)
            else:
                # Another realistic mix: 8 MB telemetry + 4 MB 3D + 4 MB random
                chunk = b"".join([
                    gen_tabular_telemetry(8 * 1024 * 1024, seed=6000 + b),
                    gen_volumetric_3d(4 * 1024 * 1024, seed=6500 + b),
                    rng.bytes(4 * 1024 * 1024),
                ])
            f.write(chunk)

    assert path.stat().st_size == target_bytes
    return path


def read_file_slice(path: Path, offset_mb: int, length_mb: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset_mb * 1024 * 1024)
        return f.read(length_mb * 1024 * 1024)
