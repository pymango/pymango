"""
==================================================
Image Processing and Analysis (:mod:`mango.image`)
==================================================

.. currentmodule:: mango.image

This module contains various functions for 3D image
processing and analysis.


Convolution-like Filters
========================

.. autosummary::
   :toctree: generated/

   convolve - 3D convolution with a specified 3D kernel/weight array.
   sobel - 3D Sobel image gradient magnitude
   discrete_gaussian - 3D convolution with Discrete-Gaussian kernel.
   discrete_gaussian_kernel - 3D Discrete-Gaussian kernel.
   discrete_gaussian_gradient_kernel - 3D Discrete-Gaussian gradient kernel.
   discrete_gaussian_mean_stdd - 3D Discrete-Gaussian weighted mean and standard-deviation.
   discrete_gaussian_gradient_magnitude - 3D Discrete-Gaussian convolved gradient magnitude.

Neighbourhood (structuring element) Filters
===========================================

Filter Functions
----------------

.. autosummary::
   :toctree: generated/
   
   mean_filter - Average intensity over neighbourhood. 
   median_filter - Median intensity over neighbourhood.
   stdd_filter - Standard deviation over neighbourhood.
   mad_filter - Median absolute deviation over neighbourhood.
   bilateral_filter - Gaussian weighted bilateral filtering.

Structuring Element Factory Functions
-------------------------------------

.. autosummary::
   :toctree: generated/

   se - Create an arbitrarily shaped structuring element.
   sphere_se - Create a spherical structuring element.
   order_se - Create a spherical structuring element (using *neighbourhood order*).
   box_se - Create a rectangular/box shaped structuring element.

Structuring Element Classes
---------------------------

.. autosummary::
   :toctree: generated/

    StructuringElement - Arbitrarily shaped structuring element.
    SphereStructuringElement - Spherical structuring element.
    OrderStructuringElement - Spherical structuring element.
    BoxStructuringElement - Rectangular/box shaped structuring element.


Interpolation
=============

.. autosummary::
   :toctree: generated/

   resample - re-sample image on specified grid.
   gaussian_downsample - re-sample an image on a coarse grid.
   InterpolationType - types of interpolation.
   
   affine_transform - Applies an affine transformation to an image.
   rotate - Applies a rotation transformation to an image.

Measurement
===========

.. autosummary::
   :toctree: generated/

   histogramdd - Calculates :samp:`d` dimensional histogram from :samp:`d` :obj:`mango.Dds` arrays.
   centre_of_mass - Calculate a centre of mass coordinate for an image.
   moment_of_inertia - Calculate principal moment of inertia tensor for an image.
   intensity_spherical_histogram - Populate a :func:`spherical_histogram` with *intensity counts*.
   distance_spherical_histogram - Populate a :func:`spherical_histogram` with *distance counts*.
   intensity_mult_distance_spherical_histogram - Populate a :func:`spherical_histogram` with *intensity times distance counts*.
   label - Generates an image where each connected component has a unique label.
   eliminate_labels_by_size - Removes labels from a labeled image whose size (number of voxels) lies in a specified range.
   convex_hull_2d - Calculates per-slice convex hull of non-masked voxels in an image.
   convex_hull_3d - Calculates convex hull of non-masked voxels in an image.
   
Morphology
==========

.. autosummary::
   :toctree: generated/

   distance_transform_edt - Calculates Euclidean distance transform.
   max_covering_radius - Calculates maximal covering sphere transform of a Euclidean distance transform image.

Miscellaneous
=============

.. autosummary::
   :toctree: generated/

   crop - Crop an image.
   auto_crop - Crop an image to minimal bounding box of non-masked values.
   subset - Crop an image (same as :func:`crop`).
   subsample - sample image on a regular sub-grid.
   gather_slice - copy a 2D slice from a 3D :obj:`mango.Dds` to a single MPI process.
   SphericalHistogram - Histogram of triangulated sphere surface (triangular bins).
   spherical_histogram - Factory method for creating a :obj:`SphericalHistogram` instance.

"""
from ._dds_open_filters import *
import mango
if (mango.haveRegistration):
    from . import registration
    from .registration import affine_transform, rotate, rotation_matrix

from ._utils import *

__all__ = [s for s in dir() if not s.startswith('_')]

if (mango.haveRestricted):
    from ._filters import *
    from ._DiscreteGaussian import *
    from ._FmmImage import *
