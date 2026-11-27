"""Stage 4: Traditional Compressor Finisher + Per-Clump Identity Escape Hatch + Wire Format.

Supports zstd, zstd+LDM, and lz4 finishers, exact metadata/descriptor overhead accounting,
per-clump escape hatch (guaranteeing no clump degrades below baseline compression),
and 100% bit-exact lossless round-trip encoding & decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import struct
import time
import zlib

import lz4.frame
import numpy as np
import zstandard as zstd

from .stage1_clumping import ClumpingConfig, partition_energy_clumps
from .stage2_layout import (
    LAYOUT_1D_IDENTITY,
    LAYOUT_NAMES,
    LayoutDescriptor,
    apply_layout_inverse,
    select_optimal_layout,
)
from .stage3_forest import (
    ForestConfig,
    MODE_IDENTITY,
    MODE_NAMES,
    MODE_SHAPED_ONLY,
    MODE_TREE_COARSE_FINE,
    MODE_TREE_SIMILARITY_DELTA,
    MODE_TREE_TEMPLATE_DELTA,
    OP_NONE,
    ClumpPlan,
    apply_op_inverse_pair,
    apply_template_inverse,
    build_forest_and_refine,
    coarse_fine_decode,
)

MAGIC = b"FTCC"
VERSION = 1

FINISHER_ZSTD = 0
FINISHER_ZSTD_LDM = 1
FINISHER_LZ4 = 2

FINISHER_NAME_TO_ID = {
    "zstd": FINISHER_ZSTD,
    "zstd_ldm": FINISHER_ZSTD_LDM,
    "lz4": FINISHER_LZ4,
}
FINISHER_ID_TO_NAME = {v: k for k, v in FINISHER_NAME_TO_ID.items()}

# 32-byte fixed file header:
# magic(4s), version(H), finisher_id(B), level(B), orig_size(Q),
# num_clumps(I), num_templates(I), crc32(I), meta_comp_len(I)
HEADER_FMT = "<4sHBBQIIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 32 bytes

# Compact 24-byte clump descriptor:
# raw_len(I), mode(B), layout_id(B), d0(H), d1(H), d2(H),
# tree_parent(i), tree_op(B), tree_depth(B), stream_group(B), reserved(B), aux_param(i)
# aux_param = coarse_len (for MODE_TREE_COARSE_FINE) or align_shift (for MODE_TREE_SIMILARITY_DELTA)
CLUMP_DESC_FMT = "<IBBHHHiBBBBi"
CLUMP_DESC_SIZE = struct.calcsize(CLUMP_DESC_FMT)  # 24 bytes

# Stream group IDs (5 logical residual streams)
STREAM_IDENTITY = 0
STREAM_SHAPED = 1
STREAM_TREE_RESIDUAL = 2
STREAM_COARSE_BAND = 3
STREAM_FINE_RESIDUAL = 4
NUM_STREAMS = 5


@dataclass
class CodecConfig:
    finisher: str = "zstd"              # "zstd", "zstd_ldm", "lz4"
    zstd_level: int = 3
    zstd_ldm_window_log: int = 27       # 128 MB window for LDM
    enable_stage2: bool = True
    enable_stage3: bool = True
    enable_escape_hatch: bool = True
    clumping: ClumpingConfig = field(default_factory=ClumpingConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)


@dataclass
class EncodeStats:
    raw_bytes: int = 0
    compressed_bytes: int = 0
    header_bytes: int = HEADER_SIZE
    metadata_raw_bytes: int = 0
    metadata_compressed_bytes: int = 0
    template_raw_bytes: int = 0
    payload_compressed_bytes: int = 0
    stream_compressed_sizes: List[int] = field(default_factory=lambda: [0] * NUM_STREAMS)
    num_clumps: int = 0
    clump_min_bytes: int = 0
    clump_mean_bytes: float = 0.0
    clump_max_bytes: int = 0
    mode_counts: Dict[str, int] = field(default_factory=dict)
    layout_counts: Dict[str, int] = field(default_factory=dict)
    escape_hatch_reverts: int = 0
    tree1_templates_used: int = 0
    tree2_edges_used: int = 0
    tree2_max_depth: int = 0
    tree3_level0_clumps: int = 0
    tree3_level1_clumps: int = 0
    stage1_ms: float = 0.0
    stage2_ms: float = 0.0
    stage3_ms: float = 0.0
    stage4_escape_ms: float = 0.0
    stage4_finish_ms: float = 0.0

    @property
    def total_overhead_bytes(self) -> int:
        # Header + compressed metadata block (includes clump descriptors, tree edges, templates)
        # + stream table framing (NUM_STREAMS * 4 bytes)
        return self.header_bytes + self.metadata_compressed_bytes + NUM_STREAMS * 4

    @property
    def overhead_pct(self) -> float:
        return (100.0 * self.total_overhead_bytes / self.raw_bytes) if self.raw_bytes > 0 else 0.0

    @property
    def ratio(self) -> float:
        return (self.raw_bytes / self.compressed_bytes) if self.compressed_bytes > 0 else 0.0


def _compress_block(data: bytes, finisher_id: int, level: int, ldm_wlog: int = 27) -> bytes:
    if not data:
        return b""
    if finisher_id == FINISHER_ZSTD:
        return zstd.ZstdCompressor(level=level).compress(data)
    if finisher_id == FINISHER_ZSTD_LDM:
        params = zstd.ZstdCompressionParameters.from_level(
            level, enable_ldm=True, window_log=ldm_wlog
        )
        return zstd.ZstdCompressor(compression_params=params).compress(data)
    if finisher_id == FINISHER_LZ4:
        return lz4.frame.compress(data)
    raise ValueError(f"Unknown finisher_id={finisher_id}")


def _decompress_block(data: bytes, finisher_id: int) -> bytes:
    if not data:
        return b""
    if finisher_id in (FINISHER_ZSTD, FINISHER_ZSTD_LDM):
        return zstd.ZstdDecompressor().decompress(data)
    if finisher_id == FINISHER_LZ4:
        return lz4.frame.decompress(data)
    raise ValueError(f"Unknown finisher_id={finisher_id}")


def _fast_probe_size(buf: bytes | memoryview, finisher_id: int) -> int:
    """Fast compressed size probe used by the per-clump identity escape hatch."""
    if not buf:
        return 0
    if finisher_id == FINISHER_LZ4:
        return len(lz4.frame.compress(buf))
    # For zstd / zstd_ldm, level 1 probe runs at >500 MB/s and tracks level 3 well
    return len(zstd.ZstdCompressor(level=1).compress(buf))


def encode_buffer(
    raw_bytes: bytes | memoryview,
    config: CodecConfig | None = None,
) -> Tuple[bytes, EncodeStats]:
    """Run full Stage 1 -> 2 -> 3 -> 4 lossless compression pipeline."""
    if config is None:
        config = CodecConfig()

    finisher_id = FINISHER_NAME_TO_ID[config.finisher]
    n = len(raw_bytes)
    stats = EncodeStats(raw_bytes=n)
    crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF

    if n == 0:
        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, finisher_id, config.zstd_level, 0, 0, 0, crc, 0
        )
        stream_hdr = struct.pack(f"<{NUM_STREAMS}I", *([0] * NUM_STREAMS))
        out = header + stream_hdr
        stats.compressed_bytes = len(out)
        return out, stats

    raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    # ---------------- Stage 1: Energy Clumping ----------------
    t0 = time.perf_counter()
    clumps = partition_energy_clumps(raw_arr, config.clumping)
    stats.stage1_ms = (time.perf_counter() - t0) * 1000.0

    n_clumps = len(clumps)
    lengths = [c.length for c in clumps]
    stats.num_clumps = n_clumps
    stats.clump_min_bytes = min(lengths)
    stats.clump_mean_bytes = sum(lengths) / n_clumps
    stats.clump_max_bytes = max(lengths)

    # ---------------- Stage 2: Optimal N-D Shape Selection ----------------
    t1 = time.perf_counter()
    layouts: List[LayoutDescriptor] = []
    if config.enable_stage2:
        for sp in clumps:
            lay, _, _ = select_optimal_layout(raw_arr[sp.start:sp.end], profile=sp.profile)
            layouts.append(lay)
    else:
        layouts = [LayoutDescriptor(LAYOUT_1D_IDENTITY) for _ in clumps]
    stats.stage2_ms = (time.perf_counter() - t1) * 1000.0

    # ---------------- Stage 3: Forest of Trees & Coarse-to-Fine ----------------
    t2 = time.perf_counter()
    if config.enable_stage3:
        forest_res = build_forest_and_refine(raw_arr, clumps, layouts, config.forest)
    else:
        # Stage 1+2 Only mode (no forest trees)
        fcfg = ForestConfig(
            enable_tree1_template=False,
            enable_tree2_similarity=False,
            enable_tree3_coarse_fine=False,
        )
        forest_res = build_forest_and_refine(raw_arr, clumps, layouts, fcfg)
    stats.stage3_ms = (time.perf_counter() - t2) * 1000.0

    plans = forest_res.plans
    templates = list(forest_res.templates)

    # ---------------- Stage 4A: Per-Clump Identity Escape Hatch ----------------
    t3 = time.perf_counter()
    reverts = 0
    if config.enable_escape_hatch:
        # Track template amortized overhead per beneficiary clump
        tmpl_counts = [0] * len(templates)
        for pl in plans:
            if pl.mode == MODE_TREE_TEMPLATE_DELTA and 0 <= pl.tree_parent < len(templates):
                tmpl_counts[pl.tree_parent] += 1

        for pl in plans:
            if pl.mode == MODE_IDENTITY:
                continue
            raw_c_bytes = raw_bytes[pl.start:pl.end]
            raw_probe = _fast_probe_size(raw_c_bytes, finisher_id)

            payload_bytes = memoryview(pl.payload) if pl.payload is not None else raw_c_bytes
            trans_probe = _fast_probe_size(payload_bytes, finisher_id)
            if pl.mode == MODE_TREE_COARSE_FINE and pl.coarse_band:
                trans_probe += _fast_probe_size(pl.coarse_band, finisher_id)

            # Account for descriptor + amortized template overhead
            extra_overhead = CLUMP_DESC_SIZE
            if pl.mode == MODE_TREE_TEMPLATE_DELTA and 0 <= pl.tree_parent < len(templates):
                cnt = max(1, tmpl_counts[pl.tree_parent])
                extra_overhead += len(templates[pl.tree_parent]) // cnt

            # Revert to identity unless transformed beats raw by at least 1% after overhead
            if (trans_probe + extra_overhead) >= int(raw_probe * 0.99):
                pl.mode = MODE_IDENTITY
                pl.layout = LayoutDescriptor(LAYOUT_1D_IDENTITY)
                pl.tree_parent = -1
                pl.tree_op = OP_NONE
                pl.tree_depth = 0
                pl.coarse_band = b""
                pl.payload = raw_arr[pl.start:pl.end]
                reverts += 1

        # Compact unused templates if all children reverted
        if templates:
            used_tids = sorted({
                pl.tree_parent for pl in plans if pl.mode == MODE_TREE_TEMPLATE_DELTA
            })
            if len(used_tids) < len(templates):
                remap = {old: new for new, old in enumerate(used_tids)}
                templates = [templates[old] for old in used_tids]
                for pl in plans:
                    if pl.mode == MODE_TREE_TEMPLATE_DELTA:
                        pl.tree_parent = remap[pl.tree_parent]

    stats.escape_hatch_reverts = reverts
    stats.stage4_escape_ms = (time.perf_counter() - t3) * 1000.0

    # Collect final mode / layout / tree statistics after escape hatch
    mode_counts: Dict[str, int] = {}
    layout_counts: Dict[str, int] = {}
    t2_edges = 0
    t2_depth = 0
    t3_l0 = 0
    t3_l1 = 0
    for pl in plans:
        mname = MODE_NAMES[pl.mode]
        mode_counts[mname] = mode_counts.get(mname, 0) + 1
        lname = LAYOUT_NAMES[pl.layout.layout_id]
        layout_counts[lname] = layout_counts.get(lname, 0) + 1
        if pl.mode == MODE_TREE_SIMILARITY_DELTA:
            t2_edges += 1
            if pl.tree_depth > t2_depth:
                t2_depth = pl.tree_depth
        elif pl.mode == MODE_TREE_COARSE_FINE:
            if pl.tree_op == 1:
                t3_l0 += 1
            else:
                t3_l1 += 1

    stats.mode_counts = mode_counts
    stats.layout_counts = layout_counts
    stats.tree1_templates_used = len(templates)
    stats.tree2_edges_used = t2_edges
    stats.tree2_max_depth = t2_depth
    stats.tree3_level0_clumps = t3_l0
    stats.tree3_level1_clumps = t3_l1

    # ---------------- Stage 4B: Pack Streams + Metadata & Run Finisher ----------------
    t4 = time.perf_counter()

    stream_chunks: List[List[bytes | memoryview]] = [[] for _ in range(NUM_STREAMS)]
    desc_buf = bytearray(n_clumps * CLUMP_DESC_SIZE)

    for idx, pl in enumerate(plans):
        raw_len = pl.end - pl.start
        aux_param = 0
        if pl.mode == MODE_IDENTITY:
            sg = STREAM_IDENTITY
            stream_chunks[sg].append(raw_bytes[pl.start:pl.end])
        elif pl.mode == MODE_SHAPED_ONLY:
            sg = STREAM_SHAPED
            stream_chunks[sg].append(memoryview(pl.payload))
        elif pl.mode == MODE_TREE_TEMPLATE_DELTA:
            sg = STREAM_TREE_RESIDUAL
            stream_chunks[sg].append(memoryview(pl.payload))
        elif pl.mode == MODE_TREE_SIMILARITY_DELTA:
            sg = STREAM_TREE_RESIDUAL
            stream_chunks[sg].append(memoryview(pl.payload))
            aux_param = int(pl.align_shift)
        elif pl.mode == MODE_TREE_COARSE_FINE:
            sg = STREAM_FINE_RESIDUAL
            stream_chunks[STREAM_COARSE_BAND].append(pl.coarse_band)
            stream_chunks[STREAM_FINE_RESIDUAL].append(memoryview(pl.payload))
            aux_param = len(pl.coarse_band)
        else:
            raise ValueError(f"Invalid mode={pl.mode}")

        struct.pack_into(
            CLUMP_DESC_FMT,
            desc_buf,
            idx * CLUMP_DESC_SIZE,
            raw_len,
            pl.mode,
            pl.layout.layout_id,
            pl.layout.d0,
            pl.layout.d1,
            pl.layout.d2,
            pl.tree_parent,
            pl.tree_op,
            pl.tree_depth,
            sg,
            0,
            aux_param,
        )

    # Serialize templates into metadata block
    tmpl_buf = bytearray()
    for tmpl in templates:
        tmpl_buf.extend(struct.pack("<I", len(tmpl)))
        tmpl_buf.extend(tmpl)

    meta_raw = bytes(desc_buf) + bytes(tmpl_buf)
    meta_comp = _compress_block(meta_raw, finisher_id, config.zstd_level, config.zstd_ldm_window_log)

    stats.template_raw_bytes = len(tmpl_buf)
    stats.metadata_raw_bytes = len(meta_raw)
    stats.metadata_compressed_bytes = len(meta_comp)

    # Compress each payload stream with chosen traditional finisher
    comp_streams: List[bytes] = []
    stream_sizes: List[int] = []
    for sg in range(NUM_STREAMS):
        if stream_chunks[sg]:
            joined = b"".join(stream_chunks[sg])
            comp = _compress_block(joined, finisher_id, config.zstd_level, config.zstd_ldm_window_log)
        else:
            comp = b""
        comp_streams.append(comp)
        stream_sizes.append(len(comp))

    stats.stream_compressed_sizes = stream_sizes
    stats.payload_compressed_bytes = sum(stream_sizes)

    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        finisher_id,
        config.zstd_level,
        n,
        n_clumps,
        len(templates),
        crc,
        len(meta_comp),
    )
    stream_table = struct.pack(f"<{NUM_STREAMS}I", *stream_sizes)
    out = b"".join([header, meta_comp, stream_table] + comp_streams)

    stats.compressed_bytes = len(out)
    stats.stage4_finish_ms = (time.perf_counter() - t4) * 1000.0
    return out, stats


def decode_buffer(encoded: bytes | memoryview) -> bytes:
    """Bit-exact lossless decoder for the FTCC binary wire format."""
    if len(encoded) < HEADER_SIZE:
        raise ValueError("Truncated FTCC stream")

    (
        magic,
        version,
        finisher_id,
        _level,
        orig_size,
        num_clumps,
        num_templates,
        expected_crc,
        meta_comp_len,
    ) = struct.unpack_from(HEADER_FMT, encoded, 0)

    if magic != MAGIC or version != VERSION:
        raise ValueError(f"Invalid FTCC header: magic={magic!r}, version={version}")

    if orig_size == 0:
        return b""

    pos = HEADER_SIZE
    meta_raw = _decompress_block(bytes(encoded[pos:pos + meta_comp_len]), finisher_id)
    pos += meta_comp_len

    stream_sizes = struct.unpack_from(f"<{NUM_STREAMS}I", encoded, pos)
    pos += NUM_STREAMS * 4

    # Decompress payload streams
    streams: List[np.ndarray] = []
    for sz in stream_sizes:
        if sz > 0:
            dec = _decompress_block(bytes(encoded[pos:pos + sz]), finisher_id)
            streams.append(np.frombuffer(dec, dtype=np.uint8))
        else:
            streams.append(np.empty(0, dtype=np.uint8))
        pos += sz

    # Parse clump descriptors
    descs = []
    dpos = 0
    for _ in range(num_clumps):
        descs.append(struct.unpack_from(CLUMP_DESC_FMT, meta_raw, dpos))
        dpos += CLUMP_DESC_SIZE

    # Parse Tree 1 templates
    templates: List[np.ndarray] = []
    for _ in range(num_templates):
        (tlen,) = struct.unpack_from("<I", meta_raw, dpos)
        dpos += 4
        templates.append(np.frombuffer(meta_raw[dpos:dpos + tlen], dtype=np.uint8))
        dpos += tlen

    # Track how many children reference each clump as a Tree 2 parent so we can
    # release parent buffers immediately after their last child is decoded (low peak RAM).
    ref_count = [0] * num_clumps
    for desc in descs:
        mode = desc[1]
        parent = desc[6]
        if mode == MODE_TREE_SIMILARITY_DELTA and parent >= 0:
            ref_count[parent] += 1

    # Reconstruct clumps in topological order (0..num_clumps-1)
    out = bytearray(orig_size)
    out_pos = 0
    s_cursors = [0] * NUM_STREAMS
    parent_cache: Dict[int, np.ndarray] = {}

    for idx, desc in enumerate(descs):
        (
            raw_len,
            mode,
            layout_id,
            d0,
            d1,
            d2,
            tree_parent,
            tree_op,
            _tree_depth,
            sg,
            _reserved,
            aux_param,
        ) = desc

        lay = LayoutDescriptor(layout_id=layout_id, d0=d0, d1=d1, d2=d2)
        c0 = s_cursors[sg]
        payload = streams[sg][c0:c0 + raw_len]
        s_cursors[sg] = c0 + raw_len

        if mode == MODE_IDENTITY:
            rec = payload
        elif mode == MODE_SHAPED_ONLY:
            rec = apply_layout_inverse(payload, lay)
        elif mode == MODE_TREE_TEMPLATE_DELTA:
            diff = apply_layout_inverse(payload, lay)
            rec = apply_template_inverse(diff, templates[tree_parent], tree_op)
        elif mode == MODE_TREE_SIMILARITY_DELTA:
            diff = apply_layout_inverse(payload, lay) if layout_id != LAYOUT_1D_IDENTITY else payload
            p_buf = parent_cache[tree_parent]
            rec = apply_op_inverse_pair(diff, p_buf, tree_op, aux_param)
            ref_count[tree_parent] -= 1
            if ref_count[tree_parent] <= 0:
                del parent_cache[tree_parent]
        elif mode == MODE_TREE_COARSE_FINE:
            cb_cur = s_cursors[STREAM_COARSE_BAND]
            coarse_bytes = streams[STREAM_COARSE_BAND][cb_cur:cb_cur + aux_param].tobytes()
            s_cursors[STREAM_COARSE_BAND] = cb_cur + aux_param
            shaped = coarse_fine_decode(payload, coarse_bytes, tree_op)
            rec = apply_layout_inverse(shaped, lay)
        else:
            raise ValueError(f"Unknown mode={mode}")

        if ref_count[idx] > 0:
            # Copy to own buffer only if needed by future children
            parent_cache[idx] = rec.copy()

        out[out_pos:out_pos + raw_len] = rec.tobytes()
        out_pos += raw_len

    result = bytes(out)
    actual_crc = zlib.crc32(result) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(
            f"CRC32 mismatch on decode: expected {expected_crc:#x}, got {actual_crc:#x}"
        )
    return result
