# init file of gastrometry

from .math_toolbox import biweight_median, biweight_mad
from .math_toolbox import median_check_finite
from .math_toolbox import vcorr, xiB
from .math_toolbox import return_var_map

from . import plotting

from .input import read_input

from .config import read_config
from .config import gastrogp

