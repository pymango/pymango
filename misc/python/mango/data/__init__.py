"""
==================================================
Test Images and Image Creation (:mod:`mango.data`)
==================================================

.. currentmodule:: mango.data

Functions for generating images with content.


Functions
=========

.. autosummary::
   :toctree: generated/

   fill_annular_circular_cylinder - Voxels which lie inside an annular circular cylinder are assigned a specified fill value.
   fill_box - Voxels which lie inside a box are assigned a specified fill value.
   fill_circular_cylinder - Voxels which lie inside a circular cylinder are assigned a specified fill value.
   fill_ellipsoid - Voxels which lie inside an ellipsoid are assigned a specified fill value. 
   
   createCheckerDds - Creates a 3D checker-board :obj:`Dds` image.
   chi_squared_noise - Creates a :obj:`mango.Dds` of Chi-Squared distributed noise.
   gaussian_noise - Creates a :obj:`mango.Dds` of Gaussian (Normal) distributed noise.
   gaussian_noise_like - Creates a :obj:`mango.Dds` of Gaussian (Normal) distributed noise.

"""

from ._factory import *

__all__ = [s for s in dir() if not s.startswith('_')]
