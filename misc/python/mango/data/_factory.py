import mango
import mango.core

import sys
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_open_data_generator as _mango_open_data_generator_so
    sys.setdlopenflags(_flags)
else:
    from . import _mango_open_data_generator as _mango_open_data_generator_so

from ._mango_open_data_generator import *


import numpy as np
import numpy.random
import scipy as sp
import mango.mpi

def createCheckerDds(shape, checkShape, checkOrigin=(0,0,0), black=0, white=1, dtype="uint8", mtype=None, halo=(0,0,0), mpidims=(0,0,0), origin=(0,0,0)):
    """
    Creates a 3D checker-board image.
    
    :type shape: 3-sequence
    :param shape: The global shape :samp:`(zSz,ySz,xSz)` of the returned :obj:`mango.Dds` image.
    :type checkShape: 3-sequence
    :param checkShape: The shape of the checks.
    :type checkOrigin: 3-sequence
    :param checkOrigin: The origin (relative to the :samp:`origin` parameter) of where the checks
       begin to be stacked.
    :type black: value
    :param black: The value for the *black* checks.
    :type white: value
    :param white: The value for the *white* checks.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: The element type of the returned array.
    :type mtype: :obj:`mango.mtype`
    :param mtype: The mango data type for the checker-board image.
    :type halo: 3-sequence
    :param halo: The halo size for the returned :obj:`mango.Dds` object.
    :type mpidims: 3-sequence
    :param mpidims: Cartesian MPI domain decomposition.
    :type origin: 3-sequence
    :param origin: The global origin index for the returned :obj:`mango.Dds` object.
    :rtype: :obj:`mango.Dds`
    :return: Checker-board :obj:`mango.Dds`

    """
    
    while (len(shape) <= 2):
        shape = (1,) + tuple(shape)

    cbDds = mango.empty(shape, dtype=dtype, mtype=mtype, halo=halo, mpidims=mpidims, origin=origin)
    checkOrigin = sp.array(checkOrigin) + cbDds.origin
    cbArr = cbDds.subd.asarray()
    
    sbeg = sp.array(cbDds.subd.origin)
    send = sbeg + cbDds.subd.shape
    
    coords = np.ogrid[sbeg[0]:send[0],sbeg[1]:send[1],sbeg[2]:send[2]]

    vals = np.array([black,white],dtype=dtype)
    cbArr[...] = \
        vals[
            (
                (coords[0]-checkOrigin[0])//checkShape[0]
                +
                (coords[1]-checkOrigin[1])//checkShape[1]
                +
                (coords[2]-checkOrigin[2])//checkShape[2]
            )
            %
            2
        ]

    return cbDds

def gaussian_noise(
    shape,
    mean=0.0,
    stdd=1.0,
    dtype=None,
    mtype=None,
    halo=(0,0,0),
    mpidims=(0,0,0),
    origin=(0,0,0),
    subdshape=None,
    subdorigin=None
):
    """
    Generates image of Gaussian (Normal) distributed noise.
    
    :type shape: 3-sequence
    :param shape: The global shape :samp:`(zSz,ySz,xSz)` of the returned :obj:`mango.Dds` image.
    :type mean: float
    :param mean: The mean parameter of the normally distributed noise.
    :type stdd: float
    :param stdd: The standard-deviation parameter of the normally distributed noise.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: The element type of the returned array.
    :type mtype: :obj:`mango.mtype`
    :param mtype: The mango data type for the return noise image.
    :type halo: 3-sequence
    :param halo: The halo size for the returned :obj:`mango.Dds` object.
    :type mpidims: 3-sequence
    :param mpidims: Cartesian MPI domain decomposition.
    :type origin: 3-sequence
    :param origin: The global origin index for the returned :obj:`mango.Dds` object.
    :type subdshape: 3-sequence
    :param subdshape: Explicitly specify the sub-domain shape for the current MPI process.
    :type subdorigin: 3-sequence
    :param subdorigin: Explicitly specify the sub-domain origin for the current MPI process.

    :rtype: :obj:`mango.Dds`
    :return: Noise :obj:`mango.Dds` image.

    """
    dds = \
        mango.empty(
            shape=shape,
            dtype=dtype,
            mtype=mtype,
            halo=halo,
            mpidims=mpidims,
            origin=origin,
            subdshape=subdshape,
            subdorigin=subdorigin
        )
    dds.subd.asarray()[...] = np.random.normal(loc=mean, scale=stdd, size=dds.subd.asarray().shape)
    
    return dds;

def gaussian_noise_like(
    input,
    mean=0.0,
    stdd=1.0,
    shape=None,
    dtype=None,
    mtype=None,
    halo=None,
    mpidims=None,
    origin=None,
    subdshape=None,
    subdorigin=None
):
    """
    Generates image of Gaussian (Normal) distributed noise.
    
    :type input: :obj:`Dds`
    :param input:  The type/shape/MPI-layout determine the same attributes of the returned :obj:`Dds`.
    :type shape: 3-sequence
    :param shape:  Overrides :samp:`dds.shape`.
    :type dtype: scipy.dtype
    :param dtype:  Overrides :samp:`dds.dtype`.
    :type mtype: mango.mtype
    :param mtype:  Overrides :samp:`dds.mtype`.
    :type halo: 3-sequence
    :param halo:  Overrides :samp:`dds.halo`.
    :type mpidims: 3-sequence
    :param mpidims:  Overrides :samp:`dds.mpi.shape`.
    :type origin: 3-sequence
    :param origin:  Overrides :samp:`dds.origin`.
    :type subdshape: 3-sequence
    :param subdshape: Overrides :samp:`dds.subd.shape`.
    :type subdorigin: 3-sequence
    :param subdorigin: Overrides :samp:`dds.subd.origin`.

    :rtype: :obj:`Dds`
    :return: :obj:`Dds` array of gaussian noise initialised elements with the same type,
       shape and MPI-layout as :samp:`input`.
    """
    out = \
        mango.empty_like(
            input,
            shape=shape,
            dtype=dtype,
            mtype=mtype,
            halo=halo,
            mpidims=mpidims,
            origin=origin,
            subdshape=subdshape,
            subdorigin=subdorigin
        )
    out.subd.asarray()[...] = np.random.normal(loc=mean, scale=stdd, size=out.subd.asarray().shape)

    return out

def chi_squared_noise(
    shape,
    dof=1.0,
    dtype=None,
    mtype=None,
    halo=(0,0,0),
    mpidims=(0,0,0),
    origin=(0,0,0),
    subdshape=None,
    subdorigin=None
):
    """
    Generates image of Chi-Squared distributed noise.
    
    :type shape: 3-sequence
    :param shape: The global shape :samp:`(zSz,ySz,xSz)` of the returned :obj:`mango.Dds` image.
    :type dof: float
    :param dof: The degrees-of-freedom parameter of the central-Chi-Squared distributed noise.
    :type dtype: :obj:`numpy.dtype`
    :param dtype: The element type of the returned array.
    :type mtype: :obj:`mango.mtype`
    :param mtype: The mango data type for the return noise image.
    :type halo: 3-sequence
    :param halo: The halo size for the returned :obj:`mango.Dds` object.
    :type mpidims: 3-sequence
    :param mpidims: Cartesian MPI domain decomposition.
    :type origin: 3-sequence
    :param origin: The global origin index for the returned :obj:`mango.Dds` object.
    :type subdshape: 3-sequence
    :param subdshape: Explicitly specify the sub-domain shape for the current MPI process.
    :type subdorigin: 3-sequence
    :param subdorigin: Explicitly specify the sub-domain origin for the current MPI process.

    :rtype: :obj:`mango.Dds`
    :return: Noise :obj:`mango.Dds` image.


    """
    dds = \
        mango.empty(
            shape=shape,
            dtype=dtype,
            mtype=mtype,
            halo=halo,
            mpidims=mpidims,
            origin=origin,
            subdshape=subdshape,
            subdorigin=subdorigin
        )
    dds.subd.asarray()[...] = np.random.chisquare(df=dof, size=dds.subd.asarray().shape)
    
    return dds;
