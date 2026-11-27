"""FTCC: Forest-Tree, Shape-Adaptive Multi-Level Lossless Compression Prototype."""

from .stage1_clumping import ClumpSpec, ClumpingConfig, partition_energy_clumps
from .stage2_layout import (
    LAYOUT_1D_IDENTITY,
    LAYOUT_1D_DELTA,
    LAYOUT_2D_TRANSPOSE,
    LAYOUT_2D_TRANSPOSE_DELTA,
    LAYOUT_3D_CUBE_PERMUTE,
    LAYOUT_3D_CUBE_PERMUTE_DELTA,
    LAYOUT_NAMES,
    LayoutDescriptor,
    select_optimal_layout,
    apply_layout_forward,
    apply_layout_inverse,
)
from .stage3_forest import (
    ForestConfig,
    ForestResult,
    ClumpPlan,
    MODE_IDENTITY,
    MODE_SHAPED_ONLY,
    MODE_TREE_TEMPLATE_DELTA,
    MODE_TREE_SIMILARITY_DELTA,
    MODE_TREE_COARSE_FINE,
    MODE_NAMES,
)
from .codec import (
    CodecConfig,
    EncodeStats,
    encode_buffer,
    decode_buffer,
)

__all__ = [
    "ClumpSpec",
    "ClumpingConfig",
    "partition_energy_clumps",
    "LAYOUT_1D_IDENTITY",
    "LAYOUT_1D_DELTA",
    "LAYOUT_2D_TRANSPOSE",
    "LAYOUT_2D_TRANSPOSE_DELTA",
    "LAYOUT_3D_CUBE_PERMUTE",
    "LAYOUT_3D_CUBE_PERMUTE_DELTA",
    "LAYOUT_NAMES",
    "LayoutDescriptor",
    "select_optimal_layout",
    "apply_layout_forward",
    "apply_layout_inverse",
    "ForestConfig",
    "ForestResult",
    "ClumpPlan",
    "MODE_IDENTITY",
    "MODE_SHAPED_ONLY",
    "MODE_TREE_TEMPLATE_DELTA",
    "MODE_TREE_SIMILARITY_DELTA",
    "MODE_TREE_COARSE_FINE",
    "MODE_NAMES",
    "CodecConfig",
    "EncodeStats",
    "encode_buffer",
    "decode_buffer",
]
