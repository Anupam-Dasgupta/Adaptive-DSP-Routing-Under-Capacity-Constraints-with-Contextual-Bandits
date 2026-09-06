"""Adaptive DSP routing research package."""

from .data import (
    AvazuFeaturePipeline,
    chronological_split,
    chronological_sample,
    load_avazu,
    sort_chronologically,
)
from .ctr import CTRModelResult, append_ctr_feature, fit_ctr_model

__all__ = [
    "AvazuFeaturePipeline",
    "chronological_split",
    "chronological_sample",
    "load_avazu",
    "sort_chronologically",
    "CTRModelResult",
    "append_ctr_feature",
    "fit_ctr_model",
]

__version__ = "0.1.0"
