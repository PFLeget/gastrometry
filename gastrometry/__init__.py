# init file of gastrometry

from .math_toolbox import biweight_median, biweight_mad
from .math_toolbox import median_check_finite
from .math_toolbox import vcorr, xiB
from .math_toolbox import return_var_map

from . import plotting

from .input import read_input
from .read_output import gather_input_all, write_output
from .machine_gun_jobs import launch_jobs_ccin2p3
from .gpastro import gpastro
from .comp_mean_function import run_ma_poule_mean
from .meanify1D import meanify1D_wrms

from .config import read_config
from .config import gastrogp
from .config import gastrify

