"""DyHuCoG: Dynamic Hybrid Recommender via Graph-based Cooperative Games"""

__version__ = "1.0.0"
__author__ = "Mouad Louhichi"
__email__ = "mouad_louhichi@um5.ac.ma"

from . import models
from . import data
from . import utils
from . import explainability

__all__ = ['models', 'data', 'utils', 'explainability']