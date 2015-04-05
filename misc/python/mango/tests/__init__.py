"""
===========================================
Top Level Unit Testing (:mod:`mango.tests`)
===========================================

.. currentmodule:: mango.tests

See the python :mod:`unittest` module for unit-test framework details.

Run all :mod:`mango` unit-tests (single MPI process) by executing this module::

   python -m mango.tests
   
Execute all :mod:`mango` unit-tests using 8-MPI-process parallelism as::

   mpirun -np 8 python -m mango.tests

See options for running unit-test cases::

   python -m mango.tests --help

This module imports unit-tests (test-cases) from a
number of mango unit-test submodules.

.. autosummary::
   :toctree: generated/

"""
import mango

from mango.coreTest            import *
from mango.ioTest              import *
from mango.mpiTest             import *

if (not mango.haveReconstructionOnly):
    from mango.dataTest        import *
    from mango.imageTest       import *
    from mango.fmmTest         import *
    from mango.applicationTest import *

if (mango.haveReconstruction):
    from mango.recon.ioTest    import *

__all__ = [s for s in dir() if not s.startswith('_')]
