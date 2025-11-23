"""Distribution shift detection utilities for lane perception datasets."""

from ._vendor import bootstrap as _bootstrap_vendor
from .shift_experiment import ShiftExperiment, extract_features
from .data.data_logging import JsonExperimentManager, JsonDict, JsonStyle

_bootstrap_vendor()

__all__ = [
    "ShiftExperiment",
    "extract_features",
    "JsonExperimentManager",
    "JsonDict",
    "JsonStyle",
]

__version__ = "0.1.0"
