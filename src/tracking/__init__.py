"""Tracking subpackage — ByteTrack multi-object tracker and target lock.

Exports:
    ByteTracker: Pure-Python ByteTrack implementation with 3-tier association.
    TargetLock:  Exemplar-based target lock with golden template ReID matching.
"""

from src.tracking.byte_tracker import ByteTracker
from src.tracking.target_lock import TargetLock

__all__ = ["ByteTracker", "TargetLock"]
