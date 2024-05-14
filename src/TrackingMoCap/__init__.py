"""
TrackingMoCap
"""
from __future__ import annotations

from .Tracking import compute_marker_distances

from importlib.metadata import version

__all__ = ("__version__", 'compute_marker_distances')
__version__ = version(__name__) 
