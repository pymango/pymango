__doc__ = \
"""
=======================================================================================
Main-driver :obj:`LogStream` variables (:mod:`mango.application.main_driver.logstream`)
=======================================================================================

.. currentmodule:: mango.application.main_driver.logstream

Logging objects/attributes for :obj:`mango.application.main_driver.MainDriverFilter` filters.

Classes
=======

.. autosummary::
   :toctree: generated/
   
   LogStream - Message logging for :obj:`mango.application.main_driver.MainDriverFilter` filters.


Attributes
==========

.. autodata:: log

.. autodata:: mstLog

.. autodata:: mstOut

.. autodata:: warnLog

.. autodata:: errLog


"""

import mango
import mango.mpi as mpi

import os
import os.path
import sys
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_main_driver as _mango_main_driver_so
    sys.setdlopenflags(_flags)
else:
    from . import _mango_main_driver as _mango_main_driver_so

from mango.core import LogStream

#: Messages sent to stdout, prefixed with :samp:`'P<RANK>'`, where :samp:`<RANK>` is MPI process world rank.
log  = _mango_main_driver_so._log

#: Messages sent to stdout, prefixed with :samp:`'MST'`, and messages also saved to history-meta-data.
mstLog  = _mango_main_driver_so._mstLog

#: Messages sent to stdout, prefixed with :samp:`'OUT'`.
mstOut  = _mango_main_driver_so._mstOut

#: Messages sent to stderr, prefixed with :samp:`'WARNING'`.
warnLog = _mango_main_driver_so._warnLog

#: Messages sent to stderr, prefixed with :samp:`'ERROR'`.
errLog  = _mango_main_driver_so._errLog

__all__ = [s for s in dir() if not s.startswith('_')]
