"""
====================================================
Image Registration (:mod:`mango.image.registration`)
====================================================

.. currentmodule:: mango.image.registration

This module contains various functions and classes for
3D image registration.


Functions
=========

.. autosummary::
   :toctree: generated/

   affine_transform - Apply an affine transformation to an image.

"""

from ._registration import *

__all__ = [s for s in dir() if not s.startswith('_')]
