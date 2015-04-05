"""
========================================================
Finite Mixture Model Image Analysis (:mod:`mango.fmm`)
========================================================

.. currentmodule:: mango.fmm

Statistical image analysis module.

Statistical Tests
=================

.. autosummary::
   :toctree: generated/

   gaussian_pvalue - Generates a Normal-Distribution P-value image.
   chi_squared_pvalue - Generates a Chi-Squared-Distribution P-value image.
   non_central_chi_squared_pvalue - Generates a Non-Central-Chi-Squared-Distribution P-value image.
   generalised_chi_squared_pvalue - Generates a Non-Central-Chi-Squared-Distribution P-value image.
   PValueSidednessType - Used to specify whether *two-sided* or *one-sided* P-values are generated.
   gchisqrd - Generalised Chi-Squared distribution factory function.
   
   
Classes
=======

.. autosummary::
   :toctree: generated/

   GenChiSqrd - Generalised Chi-Squared distribution.
"""
import mango
import sys
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_open_fmm
    sys.setdlopenflags(_flags)
else:
    from . import _mango_open_fmm

from ._mango_open_fmm import *
from mango.utils import ModuleObjectFactory as _ModuleObjectFactory

_factory = _ModuleObjectFactory(_mango_open_fmm)

# Sphinx doco class only.
class GenChiSqrd(_mango_open_fmm._gchisqrd_float64):
    __doc__ = _mango_open_fmm._gchisqrd_float64.__doc__


def gchisqrd(coeffs, noncentralities, dtype="float64"):
    """
    Returns Generalised Chi-Squared distribution object (:obj:`GenChiSqrd`).
    
    :type coeffs: :obj:`numpy.array`
    :param coeffs: :samp:`(n,)` shaped array of sum coefficients.
    :type noncentralities: :obj:`numpy.array`
    :param noncentralities: :samp:`(n,)` shaped array of non-centrality (:samp:`(mean/stdd)**2`) parameters.
    
    :rtype: :obj:`GenChiSqrd`
    :return: Generalised Chi-Squared distribution object.
    
    """
    return _factory.create(classNamePrefix="_gchisqrd_", dtype=dtype, **{"coeffs":coeffs, "noncentralities":noncentralities})

if (mango.haveRestricted):
    from ._Hist2dPeaks import *

__all__ = [s for s in dir() if not s.startswith('_')]

from .MixtureModel import *
from .Gaussian import *
from .MixtureModelFitPlotter import *

if (mango.haveRestricted):
    from .SkewNormal import *
    from .SummedMixtureModel import *
    from ._MaxLikelihoodUtils import *
    from ._BinnedGmmEm import *
    from ._BinnedSummedGmmEm import *
    from ._PwGaussianMixelMixtureModel import *
    from ._fmm import *
