"""specfuel."""

from . import corr, loss
from .stock import Feedstock

__version__ = "0.0.0"

__all__ = ["Feedstock", "corr", "loss"]
