import mango
import sys
import inspect
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_open_main_driver as _mango_open_main_driver_so
    if (mango.haveRegistration):
        from . import _mango_reg_main_driver as _mango_reg_main_driver_so
    if (mango.haveRestricted and (not mango.haveReconstructionOnly)):
        from . import _mango_rest_main_driver as _mango_rest_main_driver_so
    if (mango.haveReconstruction):
        from . import _mango_recon_main_driver as _mango_recon_main_driver_so
    if (mango.haveContrib):
        from . import _mango_contrib_main_driver as _mango_contrib_main_driver_so
        
    sys.setdlopenflags(_flags)
else:
    from . import _mango_open_main_driver as _mango_open_main_driver_so
    if (mango.haveRegistration):
        from . import _mango_reg_main_driver as _mango_reg_main_driver_so
    if (mango.haveRestricted and (not mango.haveReconstructionOnly)):
        from . import _mango_rest_main_driver as _mango_rest_main_driver_so
    if (mango.haveReconstruction):
        from . import _mango_recon_main_driver as _mango_recon_main_driver_so
    if (mango.haveContrib):
        from . import _mango_contrib_main_driver as _mango_contrib_main_driver_so

from mango.application.main_driver._mango_main_driver import MainDriverFilter

from ._mango_open_main_driver import *

if (mango.haveRegistration):
    from ._mango_reg_main_driver import *

if (mango.haveRestricted and (not mango.haveReconstructionOnly)):
    from ._mango_rest_main_driver import *

if (mango.haveReconstruction):
    from ._mango_recon_main_driver import *

if (mango.haveContrib):
    from ._mango_contrib_main_driver import *

_filterClsList = []
for _member in inspect.getmembers(sys.modules[__name__]):
    if (inspect.isclass(_member[1]) and issubclass(_member[1], MainDriverFilter)):
        _filterClsList.append(_member)

_tableEntryWidth = max([len(_name) for _name in [_m[0] for _m in _filterClsList]])
_rowEndStr = "+-" + ("-"*(_tableEntryWidth)) + "-+\n"

_filtClsDocStr = ""
_filtClsDocStr += _rowEndStr
for _member in _filterClsList:
    _filtClsDocStr += ("| %%-%ds |\n" % _tableEntryWidth) % _member[0]
    _filtClsDocStr += _rowEndStr

__doc__ = \
"""
=========================================================================
Main driver filter classes (:mod:`mango.application.main_driver.filters`)
=========================================================================

.. currentmodule:: mango.application.main_driver.filters

Classes
=======

.. autosummary::
   :toctree: generated/
   
   MainDriverFilter - Base class for main-driver filters.

Filter List
===========

%s

""" % _filtClsDocStr


__all__ = [s for s in dir() if not s.startswith('_')]

