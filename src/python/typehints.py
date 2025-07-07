from dataclasses import dataclass
from typing import Annotated

import numpy as np


@dataclass
class Range:
  min_value: float
  max_value: float

  def __contains__(self, value):
    return self.min_value <= value <= self.max_value


PolarAngle = Annotated[float, Range(0, np.pi / 2)]
AzimuthalAngle = Annotated[float, Range(0, 2 * np.pi)]
