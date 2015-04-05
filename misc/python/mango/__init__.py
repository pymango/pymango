from . import mpi
from . import core
from .core import *
from .core import _commonDocString
from . import utils
from . import io

if (not haveReconstructionOnly):
    from . import application
else:
    application = None

if (haveReconstruction):
    from . import recon
else:
    recon = None

__miscAttribDoc__ = \
"""
Miscellaneous Attributes
========================

See :ref:`Miscellaneous Attributes <mango-core-misc-attrib-section>` from the :mod:`mango.core` module.

"""
__doc__ =\
"""
=================================================
Mango 3D Image Processing Package  (:mod:`mango`)
=================================================

.. currentmodule:: mango

Mango MPI distributed image processing package. The :mod:`mango`
module imports many of it's functions and classes from the
:mod:`mango.core` module.

%s

%s

""" % (_commonDocString,  __miscAttribDoc__)


if (not haveReconstructionOnly):
    from . import image
    from . import fmm

__all__ = [s for s in dir() if not s.startswith('_')]

