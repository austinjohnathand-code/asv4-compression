"""Phase 0: Headroom & Baseline Harness (bench.py for ASV3 v3 Spec).

Runs the pre-registered corpus across ALL baselines and ablations in a flat
multi-core task pool (saturating all CPU cores cleanly even when zstd -19 --long=27/31
runs single-threaded per invocation):
  - B1:  lz4 -1
  - B2:  zstd -3
  - B3a: zstd -19 --long=27  (Primary Apples-to-Apples Competitor, 128 MB window)
  - B3b: zstd -19 --long=31  (Unconstrained long-range competitor, 2 GB window)
  - B4a: xz -6
  - B4b: xz --lzma2=preset=6,pb=2,lp=2,lc=2
  - B5:  OpenZL v0.3.0 (best schema profile + serial)
  - B6:  Blosc2 (shuffle/bitshuffle + zstd)
  - O1:  Oracle Ablation (hand-picked true layout/transform + zstd -19 --long=27)
  - A0:  Global Transform (single auto-detected transform for entire file + zstd -19 --long=27)
  - A1:  Fixed 4 MB Blocks + Stage 2 Auto-Transform + Coalesced zstd -19 --long=27

Evaluates Kill Gate 0:
  O1 must beat B3a (zstd -19 --long=27) by >= 5.0% on real target datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import blosc2
import numpy as np
import openzl.ext as ozl
import zstandard as zstd

from transforms import (
    T0_IDENTITY,
    T1_TRANSPOSE,
    T2_BITSHUFFLE,
    T2_ELEM_DELTA,
    T2_XOR_TRANSPOSE,
    T3_2D_PAETH,
    TransformSpec,
    TwoTierScorer,
    apply_transform_forward,
    apply_transform_inverse,
)

V3_DIR = Path(__file__).resolve().parent
CORPUS_DIR = V3_DIR / "corpus"
MB = 1024 * 1024


def _run_cli_compress_size(cmd: List[str], input_path: Path) -> int:
    with open(input_path, "rb") as fin:
        proc = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        total = 0
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4 * MB)
            if not chunk:
                break
            total += len(chunk)
        _, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Command {cmd} failed: {err.decode(errors='replace')}")
    return total


def _zstd19_l27_stream_transformed(p: Path, transform_fn, block_bytes: int = 16 * MB) -> int:
    proc = subprocess.Popen(
        ["zstd", "-19", "--long=27", "-T1", "-q", "-c"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None
    out_bytes = [0]

    def _drain():
        while True:
            d = proc.stdout.read(4 * MB)
            if not d:
                break
            out_bytes[0] += len(d)

    th = threading.Thread(target=_drain, daemon=True)
    th.start()
    with open(p, "rb") as fin:
        while True:
            raw = fin.read(block_bytes)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=np.uint8)
            proc.stdin.write(transform_fn(arr).tobytes())
    proc.stdin.close()
    th.join()
    err = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"zstd pipe failed: {err.decode(errors='replace')}")
    return out_bytes[0]


def run_b5_openzl(p: Path, oracle_hint: str) -> Tuple[int, str]:
    c = ozl.Compressor()
    if oracle_hint == "f32_le_transpose_xor":
        g = ozl.nodes.ConvertSerialToNumLE32()(c, ozl.graphs.Compress()(c))
        profile_used = "le32"
    elif oracle_hint == "struct64_transpose_xor":
        g = ozl.nodes.ConvertSerialToStruct(64)(
            c, ozl.nodes.TransposeSplit()(c, ozl.graphs.Compress()(c))
        )
        profile_used = "struct64+transpose"
    else:
        g = ozl.graphs.Compress()(c)
        profile_used = "serial"
    c.select_starting_graph(g)
    c.set_parameter(ozl.CParam.FormatVersion, ozl.MAX_FORMAT_VERSION)

    total = 0
    chunk_sz = 16 * MB
    align = 64 if profile_used.startswith("struct64") else (4 if profile_used == "le32" else 1)
    with open(p, "rb") as f:
        while True:
            buf = f.read(chunk_sz)
            if not buf:
                break
            body = (len(buf) // align) * align
            if body > 0:
                cctx = ozl.CCtx()
                cctx.ref_compressor(c)
                comp = cctx.compress([ozl.Input(ozl.Type.Serial, buf[:body])])
                total += len(comp)
            if body < len(buf):
                total += len(buf) - body
    return total, profile_used


def run_b6_blosc2(p: Path, oracle_hint: str) -> Tuple[int, str]:
    if oracle_hint == "f32_le_transpose_xor":
        typesize, filt, tag = 4, blosc2.Filter.BITSHUFFLE, "bitshuffle4"
    elif oracle_hint == "struct64_transpose_xor":
        typesize, filt, tag = 64, blosc2.Filter.SHUFFLE, "shuffle64"
    else:
        typesize, filt, tag = 4, blosc2.Filter.SHUFFLE, "shuffle4"

    cparams = {
        "codec": blosc2.Codec.ZSTD,
        "clevel": 7,
        "filters": [filt],
        "nthreads": 2,
    }
    total = 0
    chunk_sz = 16 * MB
    with open(p, "rb") as f:
        while True:
            buf = f.read(chunk_sz)
            if not buf:
                break
            body = (len(buf) // typesize) * typesize
            if body > 0:
                comp = blosc2.compress2(buf[:body], typesize=typesize, cparams=cparams)
                total += len(comp)
            if body < len(buf):
                total += len(buf) - body
    return total, tag


def run_o1_oracle(p: Path, oracle_hint: str) -> Tuple[int | None, str]:
    """Returns (compressed_bytes or None if equal to B3a, tag)."""
    if oracle_hint == "identity":
        return None, "T0"
    if oracle_hint == "f32_le_transpose_xor":
        with open(p, "rb") as f:
            s16 = np.frombuffer(f.read(16 * MB), dtype=np.uint8)
        cands = [
            TransformSpec(T2_BITSHUFFLE, 4, 0),
            TransformSpec(T2_XOR_TRANSPOSE, 4, 0),
            TransformSpec(T3_2D_PAETH, 4, 1024),
        ]
        c3 = zstd.ZstdCompressor(level=3)
        best_sp = min(cands, key=lambda sp: len(c3.compress(apply_transform_forward(s16, sp).tobytes())))
        return _zstd19_l27_stream_transformed(p, lambda a: apply_transform_forward(a, best_sp)), best_sp.label()
    if oracle_hint == "struct64_transpose_xor":
        sp = TransformSpec(T2_XOR_TRANSPOSE, 64, 0)
        return _zstd19_l27_stream_transformed(p, lambda a: apply_transform_forward(a, sp)), sp.label()
    if oracle_hint == "sqlite_page4096":
        return None, "T0"
    if oracle_hint == "mixed_oracle":
        lens = [
            29 * MB + 123_456,
            31 * MB + 512_000,
            34 * MB + 262_144,
            28 * MB + 700_000,
            33 * MB + 300_000,
            30 * MB + 400_000,
            35 * MB + 600_000,
        ]
        lens.append(256 * MB - sum(lens))
        specs = [
            TransformSpec(T3_2D_PAETH, 4, 1024),
            TransformSpec(T0_IDENTITY, 1, 0),
            TransformSpec(T2_XOR_TRANSPOSE, 64, 0),
            TransformSpec(T0_IDENTITY, 1, 0),
            TransformSpec(T3_2D_PAETH, 2, 64),
            TransformSpec(T0_IDENTITY, 1, 0),
            TransformSpec(T2_XOR_TRANSPOSE, 64, 0),
            TransformSpec(T3_2D_PAETH, 4, 1024),
        ]
        raw_parts: List[bytes] = []
        xform_parts: List[bytes] = []
        with open(p, "rb") as f:
            for seg_len, sp in zip(lens, specs):
                chunk = f.read(seg_len)
                if sp.tid == T0_IDENTITY:
                    raw_parts.append(chunk)
                else:
                    arr = np.frombuffer(chunk, dtype=np.uint8)
                    xform_parts.append(apply_transform_forward(arr, sp).tobytes())

        def _z19(b: bytes) -> int:
            if not b:
                return 0
            pr = subprocess.run(
                ["zstd", "-19", "--long=27", "-T1", "-q", "-c"],
                input=b,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return len(pr.stdout)

        return _z19(b"".join(raw_parts)) + _z19(b"".join(xform_parts)) + 256, "mixed_true_oracle"
    if oracle_hint == "snapshot_delta":
        snap_sz = 8 * MB
        prev = [None, None, None, None]
        proc = subprocess.Popen(
            ["zstd", "-19", "--long=27", "-T1", "-q", "-c"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        out_bytes = [0]

        def _drain():
            while True:
                d = proc.stdout.read(4 * MB)
                if not d:
                    break
                out_bytes[0] += len(d)

        th = threading.Thread(target=_drain, daemon=True)
        th.start()
        with open(p, "rb") as f:
            idx = 0
            while True:
                buf = f.read(snap_sz)
                if not buf:
                    break
                arr = np.frombuffer(buf, dtype=np.uint8)
                k = idx % 4
                if prev[k] is None:
                    proc.stdin.write(buf)
                else:
                    res = ((arr.astype(np.int16) - prev[k].astype(np.int16)) & 0xFF).astype(np.uint8)
                    proc.stdin.write(res.tobytes())
                prev[k] = arr.copy()
                idx += 1
        proc.stdin.close()
        th.join()
        proc.wait()
        return out_bytes[0] + 512, "oracle_sub_32MB"
    return None, "T0"


def run_a0_global(p: Path) -> Tuple[int | None, str]:
    scorer = TwoTierScorer()
    sz = p.stat().st_size
    votes: Dict[TransformSpec, int] = {}
    with open(p, "rb") as f:
        for frac in (0.0, 0.25, 0.5):
            off = int(sz * frac)
            f.seek(off)
            buf = np.frombuffer(f.read(min(65536, sz - off)), dtype=np.uint8)
            if buf.shape[0] < 16384:
                continue
            pr = scorer.score_probe(buf[:16384], buf if buf.shape[0] >= 32768 else None)
            votes[pr.best_spec] = votes.get(pr.best_spec, 0) + 1
    if not votes:
        return None, "T0"
    chosen = max(votes.items(), key=lambda kv: kv[1])[0]
    if chosen.tid == T0_IDENTITY:
        return None, "T0"
    return _zstd19_l27_stream_transformed(p, lambda a: apply_transform_forward(a, chosen)) + 32, chosen.label()


def run_a1_fixed4m(p: Path) -> Tuple[int | None, str]:
    """Fixed 4 MB blocks + Stage 2 Auto-Transform + Coalesced zstd -19 --long=27.

    First scans 4 MB block probes (fast: <1s even on 1 GB); if 0 blocks transform,
    returns (None, 'xform=0/N') so B3a + manifest_est is used directly without redundant pass.
    """
    scorer = TwoTierScorer()
    block_sz = 4 * MB
    block_specs: List[TransformSpec] = []
    with open(p, "rb") as f:
        while True:
            head = f.read(65536)
            if not head:
                break
            arr = np.frombuffer(head, dtype=np.uint8)
            if arr.shape[0] >= 16384:
                pr = scorer.score_probe(arr[:16384], arr if arr.shape[0] >= 32768 else None)
                block_specs.append(pr.best_spec)
            else:
                block_specs.append(TransformSpec(T0_IDENTITY, 1, 0))
            # Skip rest of 4 MB block
            rem = block_sz - len(head)
            if rem > 0:
                f.seek(rem, 1)

    n_blocks = len(block_specs)
    xform_blocks = sum(1 for sp in block_specs if sp.tid != T0_IDENTITY)
    meta_est = 32 + n_blocks * 12
    if xform_blocks == 0:
        return None, f"xform=0/{n_blocks}"

    def _start_zstd():
        pr = subprocess.Popen(
            ["zstd", "-19", "--long=27", "-T1", "-q", "-c"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cnt = [0]

        def _dr():
            assert pr.stdout is not None
            while True:
                d = pr.stdout.read(4 * MB)
                if not d:
                    break
                cnt[0] += len(d)

        th = threading.Thread(target=_dr, daemon=True)
        th.start()
        return pr, th, cnt

    p_raw, t_raw, c_raw = _start_zstd() if xform_blocks < n_blocks else (None, None, [0])
    p_xf, t_xf, c_xf = _start_zstd()

    with open(p, "rb") as f:
        for sp in block_specs:
            buf = f.read(block_sz)
            if not buf:
                break
            if sp.tid == T0_IDENTITY:
                assert p_raw is not None and p_raw.stdin is not None
                p_raw.stdin.write(buf)
            else:
                arr = np.frombuffer(buf, dtype=np.uint8)
                assert p_xf.stdin is not None
                p_xf.stdin.write(apply_transform_forward(arr, sp).tobytes())

    if p_raw is not None:
        assert p_raw.stdin is not None and t_raw is not None
        p_raw.stdin.close()
        t_raw.join()
        p_raw.wait()
    assert p_xf.stdin is not None
    p_xf.stdin.close()
    t_xf.join()
    p_xf.wait()

    return c_raw[0] + c_xf[0] + meta_est, f"xform={xform_blocks}/{n_blocks}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 Parallel Headroom & Baseline Harness")
    parser.add_argument("--workers", type=int, default=12, help="Parallel task workers")
    parser.add_argument("--resume", action="store_true", help="Reuse cached phase0_results.json")
    args = parser.parse_args()

    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())

    # Warmup Numba JIT on main thread before spawning worker threads
    scorer = TwoTierScorer()
    dummy = np.arange(65536, dtype=np.uint8)
    scorer.score_probe(dummy[:16384], dummy)
    for sp in (
        TransformSpec(T1_TRANSPOSE, 4),
        TransformSpec(T2_XOR_TRANSPOSE, 4),
        TransformSpec(T2_ELEM_DELTA, 4),
        TransformSpec(T2_BITSHUFFLE, 4),
        TransformSpec(T3_2D_PAETH, 4, 64),
    ):
        assert np.array_equal(apply_transform_inverse(apply_transform_forward(dummy, sp), sp), dummy)

    out_json = V3_DIR / "phase0_results.json"
    cache_path = V3_DIR / "phase0_task_cache.json"
    rows_by_name: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_json.exists():
        for r in json.loads(out_json.read_text()):
            rows_by_name[r["name"]] = r

    task_cache: Dict[str, Any] = {}
    if args.resume and cache_path.exists():
        task_cache = json.loads(cache_path.read_text())

    # Build flat list of independent (file, method) jobs for files not yet complete
    jobs: List[Tuple[str, str, Path, str]] = []
    todo_files = [m for m in manifest if m["name"] not in rows_by_name]
    methods = ["B1_lz4", "B2_zstd3", "B3a_zstd19_l27", "B3b_zstd19_l31", "B4a_xz6", "B4b_xz_pb2", "B5_openzl", "B6_blosc2", "O1", "A0", "A1"]

    for m in todo_files:
        p = CORPUS_DIR / m["name"]
        for meth in methods:
            key = f"{m['name']}::{meth}"
            if key not in task_cache:
                jobs.append((m["name"], meth, p, m["oracle_hint"]))

    # Schedule faster/smaller files first so progress checkpoints every few seconds,
    # while running heavy zstd/xz jobs with --workers 3 to avoid LLC/DRAM thrashing.
    def _priority(j: Tuple[str, str, Path, str]) -> int:
        fname, meth, p, _ = j
        sz_mb = p.stat().st_size // MB
        heavy = 3 if meth in ("B3a_zstd19_l27", "B3b_zstd19_l31", "O1", "A1", "B4a_xz6", "B4b_xz_pb2") else 1
        return sz_mb * heavy

    jobs.sort(key=_priority)

    print("=" * 116)
    print(f"PHASE 0 PARALLEL HARNESS | cached_files={len(rows_by_name)} | todo_files={len(todo_files)} | jobs={len(jobs)} | workers={args.workers}")
    print("=" * 116, flush=True)

    lock = threading.Lock()
    t_start = time.perf_counter()

    def _exec_job(fname: str, meth: str, p: Path, hint: str) -> Tuple[str, Any]:
        t0 = time.perf_counter()
        if meth == "B1_lz4":
            val: Any = _run_cli_compress_size(["lz4", "-1", "-c"], p)
        elif meth == "B2_zstd3":
            val = _run_cli_compress_size(["zstd", "-3", "-T4", "-q", "-c"], p)
        elif meth == "B3a_zstd19_l27":
            val = _run_cli_compress_size(["zstd", "-19", "--long=27", "-T4", "-q", "-c"], p)
        elif meth == "B3b_zstd19_l31":
            val = _run_cli_compress_size(["zstd", "-19", "--long=31", "-T4", "-q", "-c"], p)
        elif meth == "B4a_xz6":
            val = _run_cli_compress_size(["xz", "-6", "-T8", "-c"], p)
        elif meth == "B4b_xz_pb2":
            val = _run_cli_compress_size(["xz", "--lzma2=preset=6,pb=2,lp=2,lc=2", "-T8", "-c"], p)
        elif meth == "B5_openzl":
            val = list(run_b5_openzl(p, hint))
        elif meth == "B6_blosc2":
            val = list(run_b6_blosc2(p, hint))
        elif meth == "O1":
            val = list(run_o1_oracle(p, hint))
        elif meth == "A0":
            val = list(run_a0_global(p))
        elif meth == "A1":
            val = list(run_a1_fixed4m(p))
        else:
            raise ValueError(meth)
        dt = time.perf_counter() - t0
        key = f"{fname}::{meth}"
        print(f"  [{time.perf_counter()-t_start:6.1f}s] DONE {fname:<28} {meth:<15} ({dt:5.1f}s) -> {val}", flush=True)
        return key, val

    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_exec_job, *j) for j in jobs]
            for fut in as_completed(futs):
                key, val = fut.result()
                with lock:
                    task_cache[key] = val
                    cache_path.write_text(json.dumps(task_cache, indent=2))

    # Assemble complete rows in manifest order
    final_rows: List[Dict[str, Any]] = []
    for m in manifest:
        fname = m["name"]
        if fname in rows_by_name:
            final_rows.append(rows_by_name[fname])
            continue
        p = CORPUS_DIR / fname
        orig = p.stat().st_size
        b1 = task_cache[f"{fname}::B1_lz4"]
        b2 = task_cache[f"{fname}::B2_zstd3"]
        b3a = task_cache[f"{fname}::B3a_zstd19_l27"]
        b3b = task_cache[f"{fname}::B3b_zstd19_l31"]
        b4a = task_cache[f"{fname}::B4a_xz6"]
        b4b = task_cache[f"{fname}::B4b_xz_pb2"]
        b5, b5_prof = task_cache[f"{fname}::B5_openzl"]
        b6, b6_tag = task_cache[f"{fname}::B6_blosc2"]
        o1_raw, o1_tag = task_cache[f"{fname}::O1"]
        a0_raw, a0_tag = task_cache[f"{fname}::A0"]
        a1_raw, a1_tag = task_cache[f"{fname}::A1"]

        o1 = b3a if o1_raw is None else o1_raw
        a0 = (b3a + 32) if a0_raw is None else a0_raw
        n_blk = (orig + 4 * MB - 1) // (4 * MB)
        a1 = (b3a + 32 + n_blk * 12) if a1_raw is None else a1_raw

        def pct(x: int) -> float:
            return 100.0 * (x - b3a) / b3a if b3a > 0 else 0.0

        final_rows.append({
            "name": fname,
            "category": m["category"],
            "orig_bytes": orig,
            "B1_lz4": b1,
            "B2_zstd3": b2,
            "B3a_zstd19_l27": b3a,
            "B3b_zstd19_l31": b3b,
            "B4a_xz6": b4a,
            "B4b_xz_pb2": b4b,
            "B5_openzl": b5,
            "B5_profile": b5_prof,
            "B6_blosc2": b6,
            "B6_tag": b6_tag,
            "O1_oracle": o1,
            "O1_tag": o1_tag,
            "O1_vs_B3a_pct": round(pct(o1), 3),
            "A0_global": a0,
            "A0_tag": a0_tag,
            "A0_vs_B3a_pct": round(pct(a0), 3),
            "A1_fixed4m": a1,
            "A1_tag": a1_tag,
            "A1_vs_B3a_pct": round(pct(a1), 3),
        })

    out_json.write_text(json.dumps(final_rows, indent=2))
    out_csv = V3_DIR / "phase0_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        w.writeheader()
        w.writerows(final_rows)

    print("\n" + "=" * 130)
    print(f"{'Dataset':<28} {'OrigMB':>7} {'B3a(z19)':>9} {'B3b(l31)':>9} {'B4b(xz)':>8} {'B5(OZL)':>9} {'B6(Bls)':>9} {'O1 vs B3a':>10} {'A0 vs B3a':>10} {'A1 vs B3a':>10}")
    print("=" * 130)
    for r in final_rows:
        print(
            f"{r['name']:<28} {r['orig_bytes']/MB:7.1f} {r['B3a_zstd19_l27']/MB:9.2f} "
            f"{r['B3b_zstd19_l31']/MB:9.2f} {r['B4b_xz_pb2']/MB:8.2f} {r['B5_openzl']/MB:9.2f} "
            f"{r['B6_blosc2']/MB:9.2f} {r['O1_vs_B3a_pct']:+9.2f}% {r['A0_vs_B3a_pct']:+9.2f}% {r['A1_vs_B3a_pct']:+9.2f}%"
        )
    print("=" * 130)

    print("\nKILL GATE 0 EVALUATION (O1 must beat B3a by >= 5.0% on real target datasets):")
    wins = []
    for r in final_rows:
        if r["category"].startswith("target"):
            gain = -r["O1_vs_B3a_pct"]
            st = "PASS (>=5%)" if gain >= 5.0 else "LOW (<5%)"
            print(f"  {r['name']:<30} O1 gain={gain:+6.2f}%  [{st}]  (O1={r['O1_tag']}, A1={r['A1_vs_B3a_pct']:+.2f}%)")
            if gain >= 5.0:
                wins.append(r["name"])
    if wins:
        print(f"\n>>> KILL GATE 0: PASSED ({len(wins)} target datasets exceed 5.0% headroom)! Proceed to Phase 1A.")
    else:
        print("\n>>> KILL GATE 0: FAILED! Stop.")


if __name__ == "__main__":
    main()
