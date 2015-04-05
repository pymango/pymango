# This doc string is used in the top-level mango.__doc__ string.
_commonDocString = \
"""
Classes
=======

.. autosummary::
   :toctree: generated/

   Dds - 3D image array
   mtype - mango data type
   DdsNonHaloSubDomain - MPI domain decomposed non-halo-sub-domain properties.
   DdsHaloSubDomain - MPI domain decomposed halo-sub-domain properties.
   DdsMpiInfo - MPI cartesian grid domain decomposition properties.
   DdsMetaData - Meta-data associated with a :obj:`Dds` object.


Factory functions
=================

.. autosummary::
   :toctree: generated/

   empty - Create uninitialised :obj::`Dds`
   empty_like - Create uninitialised :obj::`Dds` with same attributes as a another :obj::`Dds` object 
   zeros - Create zero-initialised :obj::`Dds`
   zeros_like - Create zero-initialised :obj::`Dds` with same attributes as a another :obj::`Dds` object 
   ones - Create one-initialised :obj::`Dds`
   ones_like - Create one-initialised :obj::`Dds` with same attributes as a another :obj::`Dds` object 
   copy - Copy-construct/clone a :obj:`Dds` object.

Reduction Operations
====================

   These reduction operations use MPI *all-reduce* operations to calculate reductions
   over all non-halo parts of the sub-domains.

.. autosummary::
   :toctree: generated/

   minmax - Returns an :obj:`Dds` element-wise :samp:`(minimum,maximum)` for whole array.
   sum - Returns sum of all non-masked :obj:`Dds` elements.
   sum2 - Returns sum of non-masked :obj:`Dds` elements-squared.
   count_equal_to - Returns the number of voxels which have intensity equal to the specified value.
   count_not_equal_to - Returns the number of voxels which have intensity not equal to the specified value.
   count_less_than - Returns the number of voxels which have intensity strictly less than the specified value.
   count_masked - Returns the number of masked voxels.
   count_non_masked - Returns the number of non-masked voxels.
   unique - Returns the unique elements of a :obj:`Dds` array.
   itemfreq - Similar to :func:`scipy.stats.itemfreq`, calculates array element frequencies/counts.
   have_same_subd_decomp - Returns whether two :obj:`Dds` objects have identical sub-domain decomposition.

Miscellaneous functions
=======================

.. autosummary::
   :toctree: generated/
   
   isDds - Return whether an object is an instance of a :obj:`mango.core.Dds`.
   setLoggingVerbosityLevel - set mango c++ logging verbosity level.
   getDdsMtypeList - Returns list of all possible :obj:`mango.mtype` objects.
   gather - Copy sub-domain arrays to a single global array on a specified MPI process.
   copy_masked - Copy the masked elements from a source :obj:`Dds` array to a destination :obj:`Dds`.
   copy_non_masked - Copy the non-masked elements from a source :obj:`Dds` array to a destination :obj:`Dds`.
   map_element_values - Maps each element of an input array to a new value according to a specified mapping function.
"""

__miscAttribDoc__ = \
"""
.. _mango-core-misc-attrib-section:

Miscellaneous Attributes
========================


.. autodata:: usingMpi

.. autodata:: haveRestricted

.. autodata:: haveReconstruction

.. autodata:: haveReconstructionOnly

.. autodata:: haveRegistration

.. autodata:: usingRangeCheckedDds

.. autodata:: haveFloat16

.. autodata:: haveCGAL

.. autodata:: haveVTK

"""

__doc__ = \
"""
==============================================================
Fundamental Data Structures and Functions  (:mod:`mango.core`)
==============================================================

.. currentmodule:: mango.core

Mango core data structures and functions
of the :mod:`mango` 3D image processing package.

%s

%s
""" % (_commonDocString, __miscAttribDoc__)

from . import _core
from ._core import *

#: If :samp:`True`, support for MPI parallelism has been
#: compiled in the python shared-object modules.
usingMpi = _core.usingMpi
   
#: If :samp:`True`, development/prototype API is
#: available in various modules.
haveRestricted = _core.haveRestricted 

#: If :samp:`True`, the :mod:`mango.recon` module API is available.   
haveReconstruction = _core.haveReconstruction

#: If :samp:`True`, only the :mod:`mango.core` module API and
#: the  :mod:`mango.recon` module API are available.
haveReconstructionOnly = _core.haveReconstructionOnly

#: If :samp:`True`, the :mod:`mango.image.registration` module API is available.
haveRegistration = _core.haveRegistration

#: If :samp:`True`, mango has support for the 16 bit floating
#: point types :samp:`mtype("float16")` and :obj:`numpy.float16` . 
haveFloat16 = _core.haveFloat16

#: If :samp:`True`, parts of the mango source can utilise `CGAL <http://cgal.org>`_.
haveCGAL = _core.haveCGAL

#: If :samp:`True`, parts of the mango source can utilise `VTK <http://vtk.org>`_.
haveVTK = _core.haveVTK

#: If :samp:`True`, all distributed array element accesses are range-checked.
usingRangeCheckedDds = _core.usingRangeCheckedDds


__all__ = [s for s in dir() if not s.startswith('_')]

