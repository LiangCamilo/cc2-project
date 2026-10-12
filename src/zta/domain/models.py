from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

class ListeningMode(str, Enum):
    FAMILIAR = "familiar"
    DISCOVERY = "discovery"
    EXPLORATION = "exploration"

@dataclass(frozen=True)
class ZoneBand:
    lower: float
    upper: float

    def contains(self, distances: np.ndarray) -> np.ndarray:
        return (distances > self.lower) & (distances <= self.upper)