__doc__ = \
"""
=============================================================================================
Executes filter sequences from (qmango) parameter file (:mod:`mango.application.main_driver`)
=============================================================================================

.. currentmodule:: mango.application.main_driver

Replacement for *mango* binary executable.
This module can be run as a script as follows::

   mpirun -np 32 python -m mango.application.main_driver -base parameter_file

which will execute the filter sequence specified in the :samp:`parameter_file.in`
parameter file.

Functions
=========

.. autosummary::
   :toctree: generated/

   getArgumentParser - Returns command-line argument parser used by :samp:`__main__` execution.   
   run_main_driver - Runs filter sequence from a specified parameter file.
   registerMainDriverFilter - Register a :obj:`MainDriverFilter` class for lookup by :obj:`MainDriver`.
   findMainDriverFilter - Look-up a :obj:`MainDriverFilter` class by name and data-type.
   getAllRegisteredMainDriverFilterClasses - Returns list of all registered filter classes.

Classes
=======

.. autosummary::
   :toctree: generated/
   
   InputParmManager - Top-level filter-sequence parameter dictionary.
   InputParmSection - Filter parameter dictionary.
   MainDriver - Class for controlling filter execution sequence.
   MainDriverFilter - Base class for filters which can be executed by :obj:`MainDriver`.


"""

from ._main_driver import *
from . import filters
from . import logstream

__all__ = [s for s in dir() if not s.startswith('_')]
