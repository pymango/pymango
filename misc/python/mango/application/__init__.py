
from . import plot
from . import io
from . import shale

__doc__ =\
"""
========================================================
Application Specific Modules  (:mod:`mango.application`)
========================================================

.. currentmodule:: mango.application

Module for application specific image analysis.


Modules
=======

:mod:`mango.application.io`
   Read and write functions for non-image data I/O.
:mod:`mango.application.plot`
   Data plotting utilities.
:mod:`mango.application.shale`
   Specific analysis for shale multi-mode imaging. 
:mod:`mango.application.cylinder_fit`
   Fit cylinder(s) to an image. 

"""

__all__ = [s for s in dir() if not s.startswith('_')]
