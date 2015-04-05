"""
======================================================
Mathematical Functions and Classes (:mod:`mango.math`)
======================================================

.. currentmodule:: mango.math

This module contains various mathematical utilities.


Rotation
========

Rotation related functions.

.. autosummary::
   :toctree: generated/

   rotation_matrix - Returns a rotation matrix for a given angle and coordinate-axis of rotation.
   rotation_matrix_from_cross_prod - Returns rotation matrix for rotating one 3D vector onto another 3D vector.
   axis_angle_from_cross_prod - Returns :samp:(axis,angle)` rotation for rotating one 3D vector onto another 3D vector.
   axis_angle_from_rotation_matrix - Returns the :samp:(axis,angle)` rotation equivalent of a specified rotation matrix.
   axis_angle_to_rotation_matrix - Converts :samp:(axis,angle)` rotation to rotation matrix.
"""
from ._rotation import *

__all__ = [s for s in dir() if not s.startswith('_')]
