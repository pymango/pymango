"""
"""
import mango
import mango.core
import mango.math
from mango.math import rotation_matrix
import mango.image
import mango.mpi as mpi
import sys
if mango.haveRegistration:
    if sys.platform.startswith('linux'):
        import DLFCN as dl
        _flags = sys.getdlopenflags()
        sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
        from . import _mango_reg_core
        sys.setdlopenflags(_flags)
    else:
        from . import _mango_reg_core
else:
    _mango_reg_core = None

from mango.image._mango_open_filters import InterpolationType
from mango.image._dds_open_filters import _DdsMangoFilterApplier
import scipy as sp
import numpy as np

logger, rootLogger = mpi.getLoggers(__name__)


def affine_transform(input, matrix, shift=None, offset=None, interptype=InterpolationType.CATMULL_ROM_CUBIC_SPLINE, fill=None):
    """
    Applies an affine transformation to an image. This is the forward transform
    so that (conceptually)::
       
       idx  = scipy.array((i,j,k), dtype="int32")
       tidx = matrix.dot((idx-offset).T) + offset + shift
       out[tuple(tidx)] = input[tuple(idx)]
    
    This differs from the :func:`scipy.ndimage.interpolation.affine_transform` function
    which does (conceptually, ignoring shift)::
    
       idx  = scipy.array((i,j,k), dtype="int32")
       tidx = matrix.dot((idx-offset).T) + offset
       out[tuple(idx)] = input[tuple(tidx)]

    
    :type input: :obj:`mango.Dds`
    :param input: Image to be transformed.
    :type matrix: :obj:`numpy.ndarray`
    :param matrix: A :samp:`(3,3)` shaped affine transformation matrix.
    :type shift: 3-sequence
    :param shift: The translation (number of voxels), can be :obj:`float` elements.
    :type offset: 3-sequence
    :param offset: The centre-point of the affine transformation (relative to input.origin).
       If :samp:`None`, the centre of the image is used as the centre of affine transformation.
       Elements can be :obj:`float`.
    :type interptype: :obj:`mango.image.InterpolationType`
    :param interptype: Interpolation type.
    :type fill: numeric
    :param fill: The value used for elements outside the image-domain.
       If :samp:`None` uses the :samp:`input.mtype.maskValue()` or :samp:`0`
       if there is no :samp:`input.mtype` attribute.
    :rtype: :obj:`mango.Dds`
    :return: Affine-transformed :obj:`mango.Dds` image.
    
    """
    if (_mango_reg_core is None):
        raise Exception("This mango build has not been compiled with registration support.")
    if (sp.any(input.md.getVoxelSize() <= 0)):
        raise Exception("Non-positive voxel size (%s) found in input, affine_transform requires positive voxel size to be set." % (input.md.getVoxelSize(),))
    if (fill is None):
        fill = 0
        if (hasattr(input,"mtype") and (input.mtype != None)):
            fill = input.mtype.maskValue()
    
    
    if (offset is None):
        # Set the default offset to be the centre of the image.
        offset = sp.array(input.shape, dtype="float64")*0.5

    # Convert from relative offset value to absolute global coordinate.
    centre = sp.array(offset, dtype="float64") + input.origin
    mangoFilt = _mango_reg_core._TransformApplier(matrix, centre, shift, interptype, fill)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    ###trnsDds = filt(input, mode=mode, cval=cval)
    trnsDds = filt(input, mode="constant", cval=fill)

    return trnsDds



def rotate(input, angle, axis=0, offset=None, interptype=InterpolationType.CATMULL_ROM_CUBIC_SPLINE, fill=None):
    """
    Applies a rotation transformation to an image. This is the forward transform
    so that (conceptually)::
       
       idx  = scipy.array((i,j,k), dtype="int32")
       tidx = matrix.dot((idx-offset).T) + offset
       out[tuple(tidx)] = input[tuple(idx)]
    
    This differs from the :func:`scipy.ndimage.interpolation.rotate` function
    which does (conceptually, ignoring shift)::
    
       idx  = scipy.array((i,j,k), dtype="int32")
       tidx = matrix.dot((idx-offset).T) + offset
       out[tuple(idx)] = input[tuple(tidx)]

    
    :type input: :obj:`mango.Dds`
    :param input: Image to be transformed.
    :type angle: :obj:`float`
    :param angle: Rotation angle in degrees.
    :type axis: :obj:`int`
    :param axis: The axis of rotation.
    :type offset: 3-sequence
    :param offset: The centre-point of the rotation transformation (relative to input.origin).
       If :samp:`None`, the centre of the image is used as the centre of rotation.
       Elements can be :obj:`float`.
    :type interptype: :obj:`mango.image.InterpolationType`
    :param interptype: Interpolation type.
    :type fill: numeric
    :param fill: The value used for elements outside the image-domain.
       If :samp:`None` uses the :samp:`input.mtype.maskValue()` or :samp:`0`
       if there is no :samp:`input.mtype` attribute.
    :rtype: :obj:`mango.Dds`
    :return: Rotated :obj:`mango.Dds` image.
    
    """
    dim = len(input.shape)
    rMatrix = rotation_matrix(angle, axis, dim)

    return affine_transform(input, rMatrix, offset=offset, interptype=interptype, fill=fill)

__all__ = [s for s in dir() if not s.startswith('_')]    
