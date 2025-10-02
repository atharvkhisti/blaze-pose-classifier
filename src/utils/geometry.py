import math
import numpy as np


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle (degrees) at point b formed by a-b-c.
    a,b,c are 2D or 3D arrays. NaNs propagate to NaN.
    """
    ba = a - b
    bc = c - b
    if np.any(np.isnan(ba)) or np.any(np.isnan(bc)):
        return float('nan')
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return float('nan')
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def safe_norm(p: np.ndarray) -> float:
    if np.any(np.isnan(p)):
        return float('nan')
    return float(np.linalg.norm(p))
