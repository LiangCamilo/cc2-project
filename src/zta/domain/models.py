from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from zta.data.loader import CatalogContractError

class ListeningMode(str, Enum):
    FAMILIAR = "familiar"
    DISCOVERY = "discovery"
    EXPLORATION = "exploration"

@dataclass(frozen=True)
class ZoneBand:
    lower: float
    upper: float
    include_lower: bool = False

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (self.lower, self.upper)
        ) or not 0 <= self.lower <= self.upper:
            raise CatalogContractError("Zone bounds must be finite, nonnegative and ordered.")
        if not isinstance(self.include_lower, bool):
            raise CatalogContractError("The lower-bound inclusion flag must be boolean.")

    def contains(self, distances: np.ndarray) -> np.ndarray:
        lower = distances >= self.lower if self.include_lower else distances > self.lower
        return lower & (distances <= self.upper)
