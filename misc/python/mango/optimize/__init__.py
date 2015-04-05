"""
==============================================
Optimization Utilities (:mod:`mango.optimize`)
==============================================
.. currentmodule:: mango.optimize

MPI related optimization, distributed function optimization
and distributed mutli-start optimization.


Functions
=========

.. autosummary::
   :toctree: generated/

   distributed_minimize - Wrapper function for :func:`scipy.optimize.minimize`.
   
Classes
=======
.. autosummary::
   :toctree: generated/

   DistributedMetricEvaluator - Wrapper class to evaluate serial functions as a reduction (MPI reduce) over multiple MPI processes. 
   OptimizeResult - Optimization result object (e.g. returned by :meth:`SingleStartOptimizer.minimize`).
   SingleStartOptimizer - Performs local optimization/minimization from for a given starting parameter set.
   MultiStartOptimizer - Multi-start optimization taking advantage of MPI parallelism.
"""
from ._optimize import *

__all__ = [s for s in dir() if not s.startswith('_')]
