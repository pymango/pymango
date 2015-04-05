import scipy as sp
import numpy as np

import mango.core  as core
import mango.mpi as mpi

import sys
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_open_filters
    sys.setdlopenflags(_flags)
else:
    from . import _mango_open_filters


from ._mango_open_filters import InterpolationType, NeighbourhoodFilterType, _NeighbourhoodFilter
from ._mango_open_filters import centre_of_mass, moment_of_inertia

from mango.utils import ModuleObjectFactory as _ModuleObjectFactory

logger, rootLogger = mpi.getLoggers(__name__)

_moduleObjectFactory = _ModuleObjectFactory(_mango_open_filters)

# Sphinx doco class only
class StructuringElement (_mango_open_filters._StructuringElement_3):
    pass
StructuringElement.__doc__ = _mango_open_filters._StructuringElement_3.__doc__

# Sphinx doco class only
class SphereStructuringElement (_mango_open_filters._SphereStructuringElement_3):
    pass
SphereStructuringElement.__doc__ = _mango_open_filters._SphereStructuringElement_3.__doc__

# Sphinx doco class only
class OrderStructuringElement (_mango_open_filters._OrderStructuringElement_3):
    pass
OrderStructuringElement.__doc__ = _mango_open_filters._OrderStructuringElement_3.__doc__

class BoxStructuringElement (_mango_open_filters._BoxStructuringElement_3):
    pass
BoxStructuringElement.__doc__ = _mango_open_filters._BoxStructuringElement_3.__doc__

def se(structure=None, dim=3):
    """
    Returns a :obj:`StructuringElement` object for the specified :samp:`structure`
    array.
    
    :type structure: :obj:`numpy.array`
    :param structure: Can be a :samp:`(n,dim)` shaped array
       where :samp:`structure[n]` is an offset index indicating
       an members of the structuring element. Can also be a
       :samp:`(nz,ny,nx)` shaped array (for :samp:`dim==3`) of
       :samp:`numpy.dtype("bool")` elements, where the :samp:`True`
       elements indicate members of the structuring element.
    :type dim: :obj:`int`
    :param dim: Dimension of the structuring element.
    :rtype: :obj:`StructuringElement`
    :return: Structuring element object for use with neighbourhood-filters.
    """
    seObj = _moduleObjectFactory.create("_StructuringElement_", dtype=("%s" % dim))
    seArray = structure
    if (not(seArray is None)):
        if (not isinstance(seArray, np.ndarray)):
            seArray = sp.array(seArray)
        indices = seArray
        if (isinstance(seArray, np.ndarray) and (seArray.dtype == np.dtype("bool"))):
            indices = sp.where(seArray)
        seObj.setIndices(indices)
    return seObj

def sphere_se(radius=1.0, dim=3):
    """
    Returns a :obj:`SphereStructuringElement` object for the specified :samp:`radius`.
    
    :type radius: :obj:`float`
    :param radius: The radius of the spherical structuring element.
    :type dim: :obj:`int`
    :param dim: Dimension of the structuring element.
    :rtype: :obj:`SphereStructuringElement`
    :return: Spherical structuring element object for use with neighbourhood-filters.
    """
    return _moduleObjectFactory.create("_SphereStructuringElement_", dtype=str(dim), radius=radius)

def order_se(order=1, dim=3):
    """
    Returns a :obj:`OrderStructuringElement` object for the specified neighbourhood :samp:`order`.
    
    :type order: :obj:`int`
    :param order: The neighbourhood-order (radius-squared) of the spherical structuring element.
    :type dim: :obj:`int`
    :param dim: Dimension of the structuring element.
    :rtype: :obj:`OrderStructuringElement`
    :return: Neighbourhood order (spherical) structuring element object for use with
       neighbourhood-filters.
    """
    return _moduleObjectFactory.create("_OrderStructuringElement_", dtype=str(dim), order=order)

def box_se(shape=(3,3,3), dim=3):
    """
    Returns a :obj:`BoxStructuringElement` object for the specified box :samp:`shape`.
    
    :type shape: :obj:`int` sequence of length :samp:`dim`
    :param shape: The shape of the rectangular/box shaped structuring element.
    :type dim: :obj:`int`
    :param dim: Dimension of the structuring element.
    :rtype: :obj:`BoxStructuringElement`
    :return: Rectangular/box shaped structuring element object for use with neighbourhood-filters.
    """
    return _moduleObjectFactory.create("_BoxStructuringElement_", dtype=str(dim), boxShape=shape)

class _DdsMangoFilterApplier (object):
    """
    Helper class for applying an *in-place* filter to a :obj:`mango.Dds` image.
    Takes care of resizing halo as appropriate and copying result back
    to same halo-size as the input :obj:`mango.Dds`. 
    """
    def __init__(self, ddsMangoFilter):
        self.ddsMangoFilter = ddsMangoFilter
        self.outputNeedsResize = False

    def upsizeHalo(self, origInput, haloSz):
        self.outputNeedsResize = False
        input = origInput
        if (sp.any(origInput.halo < haloSz)):
            logger.debug("_DdsMangoFilterApplier.upsizeHalo: Up-sizing halo to size=%s..." % (haloSz,))
            self.outputNeedsResize = True
            input = \
                core.empty_like(origInput, halo=haloSz)
            input.fill(origInput)
            logger.debug("_DdsMangoFilterApplier.upsizeHalo: Done up-sizing halo to size=%s." % (haloSz,))

        return input
    
    def initialiseHalo(self, input, mode, cval):
        input.updateOverlapRegions()
        if (mode == None):
            mode = "mirror"
        mode = mode.lower()
        if (mode == "mirror"):
            input.mirrorOuterLayersToBorder(False)
        elif (mode == "reflect"):
            input.mirrorOuterLayersToBorder(True)
        elif (mode == "constant"):
            if (cval == None):
                cval = 0
                if (input.mtype != None):
                    cval = input.mtype.maskValue()
            input.setBorderToValue(cval)

    def createInput(self, input, halo):
        # Change the halo/overlap size (if required) as per convolution
        # filter requirements.
        self.origInput = input
        self.input = self.upsizeHalo(input, halo)
        return self.input

    def createSingleDdsOutput(self, output):
        if (self.outputNeedsResize):
            # We changed the halo size, want to return
            # output which has same size halo as input.
            self.input = core.empty_like(output, halo=self.origInput.halo)
            self.input.fill(output)
            output = self.input

        return output

    def isDds(self, obj):
        return core.isDds(obj)

    def createOutput(self, output):
        if (self.isDds(output)):
            output = self.createSingleDdsOutput(output)
        else:
            for outDdsIdx in range(len(output)):
                outDds = output[outDdsIdx]
                output[outDdsIdx] = self.createSingleDdsOutput(outDds)
                del outDds
        del self.input
        del self.origInput

        return output

    def __call__(self, input, output=None, mode="mirror", cval=None):
        if (output != None):
            raise ValueError("output != None, output argument is not yet supported.")

        # Change the halo/overlap size (if required) as per convolution
        # filter requirements.
        input = self.createInput(input, self.ddsMangoFilter.getRequiredHaloSize())
        # Initialise the halo region as per mode argument.
        self.initialiseHalo(input, mode, cval)
        
        # Do the filtering.
        logger.debug("_DdsMangoFilterApplier.__call__: Initialised halo input.halo=%s, filtering..." % (input.halo,))
        filtDds = self.ddsMangoFilter(input)
        logger.debug("_DdsMangoFilterApplier.__call__: Done filtering.")
        
        del input
        logger.debug("_DdsMangoFilterApplier.__call__: Reshaping output...")
        filtDds = self.createOutput(filtDds)
        logger.debug("_DdsMangoFilterApplier.__call__: Done reshaping output.")

        return filtDds

class _DdsConvolver:
    def __init__(self, kernel, kernelOrigin=None):
        self.kernel = kernel
        self.kernelOrigin = kernelOrigin
        if (self.kernelOrigin == None):
            self.kernelOrigin = sp.array(kernel.shape, dtype="int32")//2
    
    def getRequiredHaloSize(self):
        return np.max(self.kernel.shape)//2
    
    def __call__(self, input):
        spConvArr = sp.ndimage.convolve(input.asarray(), self.kernel, mode="mirror")
        output = core.empty_like(input)
        output.asarray()[...] = spConvArr
        if (input.mtype != None):
            output.asarray()[...] = sp.where(input.asarray() == input.mtype.maskValue(), input.mtype.maskValue(), output.asarray())
        
        return output

def spconvolve(input, kernel, kernelOrigin=None, mode="mirror", cval=None):
    """
    Convolves an input :obj:`Dds` image with the specified :samp:`kernel` weights.
    Uses the :func:`scipy.ndimage.convolve` function to perform MPI sub-domain
    convolutions.
    
    :type input: :obj:`mango.Dds`
    :param input: :obj:`mango.Dds` image which is to be convolved with kernel weights.
    :type kernel: :obj:`numpy.array`
    :param kernel: A 3D :obj:`numpy.array` convolution kernel.
    :type mode: str
    :param mode: String indicating how to handle borders, one of
        "mirror", "constant", "reflect".
    :type cval: scalar
    :param cval: When :samp:`mode=='constant'` use this value
        to initialise borders.
    :rtype: :obj:`mango.Dds`
    :return: Convolved image.
    """
    convolver = _DdsConvolver(kernel, kernelOrigin)
    filt = _DdsMangoFilterApplier(convolver)
    convDds = filt(input, mode=mode, cval=cval)

    return convDds

def convolve(input, weights=sp.ones((1,1,1)), mode="mirror", cval=None, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0), normWeights=False):
    """
    3D convolution filter.
    
    :type input: :obj:`mango.Dds`
    :param input: Image to be filtered.
    :type weights: :obj:`numpy.array`.
    :param weights: 3D convolution kernel/weights.
    :type mode: str
    :param mode: String indicating how to handle borders, one of
        "mirror", "constant", "reflect".
    :type cval: scalar
    :param cval: When :samp:`mode=='constant'` use this value
        to initialise borders.
    :type stride: 3-sequence
    :param stride: The sub-sampling step-size for each dimension.
    :type boffset: 3-sequence
    :param boffset: The offset (:samp:`(bz,by,bx)` relative to :samp:`input.origin`)
       at which the filtering starts.
    :type eoffset: 3-sequence
    :param eoffset: The offset (:samp:`(ez,ey,ex)` relative to :samp:`(input.origin + input.shape)`)
       at which the filtering stops.
    :type normWeights: :obj:`bool`
    :param normWeights: If :samp:`True`, weights are normalised before performing the
       convolution. Also, any partially masked-neighbourhoods are normalised by the sum
       of the non-masked weights. 
    
    :rtype: :obj:`mango.Dds`
    :return: Convolved image.
    """
    se = box_se(weights.shape)
    mangoFilt = \
        _NeighbourhoodFilter(
            NeighbourhoodFilterType.CONVOLUTION,
            se,
            stride,
            boffset,
            eoffset,
            mode,
            cval,
            {
                "weights" : weights,
                "do_weight_normalisation" : normWeights
            }
        )
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    convDds = filt(input, mode=mode, cval=cval)
    return convDds

def sobel(input, output=None, mode="mirror", cval=None):
    """
    Sobel gradient-magnitude filter.
    
    :type input: :obj:`mango.Dds`
    :param input: :obj:`mango.Dds` image which is to be Sobel filtered.
    :type mode: str
    :param mode: String indicating how to handle borders, one of
        "mirror", "constant", "reflect".
    :type cval: scalar
    :param cval: When :samp:`mode=='constant'` use this value
        to initialise borders.
    :rtype: :obj:`mango.Dds`
    :return: Sobel filtered gradient-magnitude image.
    """
    mangoFilt = _mango_open_filters._Sobel()
    filt = _DdsMangoFilterApplier(mangoFilt)
    sobelDds = filt(input, mode=mode, cval=cval)

    return sobelDds

def subset(input, offset=None, shape=None):
    """
    *Crops* an input image to a new origin and shape.

    :type input: :obj:`mango.Dds`
    :param input: :obj:`mango.Dds` image which is to be cropped.
    :type offset: 3-sequence of int
    :param offset: Global origin index (relative to :samp:`input.origin`)
       of the returned subset.
    :type shape: 3-sequence of int
    :param shape: Global shape of the returned subset.
    :rtype: :obj:`mango.Dds`
    :return: Cropped :obj:`mango.Dds`.
    """
    if (offset is None):
        offset = sp.zeros((0,0,0))
    else:
        offset = sp.array(offset)
    
    if (shape is None):
        shape = input.shape - offset
    
    if (sp.any(shape < input.halo)):
        raise \
            ValueError(
                "halo=%s is greater than requested crop-shape=%s"
                %
                (input.halo, shape)
            )

    if (sp.any(input.mpi.shape > shape)):
        raise \
            ValueError(
                "MPI domain decomposition mpidims=%s is greater than requested crop-shape=%s"
                %
                (input.mpi.shape, shape)
            )
    
    origin = input.origin + offset
    
    if (input.mtype != None):
        ssDds = core.empty_like(input, shape=shape, origin=origin)
        ssDds.setAllToValue(ssDds.mtype.maskValue())
    else:
        ssDds = core.zeros_like(input, shape=shape, origin=origin)
    
    ssDds.fill(input)
    
    return ssDds

def crop(input, offset=None, shape=None):
    return subset(input=input, offset=offset, shape=shape)
crop.__doc__ = subset.__doc__

def auto_crop(input, mask=None):
    """
    Crops an image to smallest possible bounding box containing all non-masked voxels.
    
    :type input: :obj:`mango.Dds`
    :param input: Image to be cropped.
    :type mask: numeric
    :param mask: Find smallest bounding box of all voxels not equal to this value. 
    :rtype: :obj:`mango.Dds`
    :return: Cropped version of :samp:`{input}` image.
    """
    return _mango_open_filters._auto_crop(input, mask)

def subsample(
    input,
    step=None,
    start=None,
    stop=None
):
    """
    Sub-sample/slice a :obj:`mango.Dds` object. This is akin to :obj:`numpy.array`
    `basic slicing <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing>`_ ,
    however a copy of the data is generated
    (not a `view <http://docs.scipy.org/doc/numpy/glossary.html#term-view>`_ ).
    
    :type input: :obj:`mango.Dds`
    :param input: :obj:`mango.Dds` image which is to be sub-sampled.
    :type step: 3-sequence of int
    :param step: The sub-sampling step size, :samp:`(zStep, yStep, xStep)`.
    :type start: 3-sequence of int
    :param start: The global index (relative to :samp:`input.origin`) of the first
       sub-sampled voxel, :samp:`(zStart, yStart, xStart)`.
    :type stop: 3-sequence of int
    :param stop: The global index (relative to :samp:`input.origin`) of the next-to-last
       sub-sampled voxel, :samp:`(zStop, yStop, xStop)`.

    :rtype: :obj:`mango.Dds`
    :return: Sub-sampled image.
    
    For example::
       >>> import mango;
       >>> dds = mango.zeros(origin=(16,-32,64), shape=(32,64,128))
       >>> ssDds = mango.image.subsample(dds, start=(8,8,16), stop=(24,56,116), step=(2,4,8))
       >>> print("ssDds.origin = %s" % (ssDds.origin,))
       ssDds.origin = [ 24 -24  80]
       >>> print("ssDds.shape  = %s" % (ssDds.shape,))
       ssDds.shape  = [ 8 12 13]

    """

    if (step is None):
        step = (1,1,1)
    if (start is None):
        start = (0,0,0)
    if (stop is None):
        stop = input.shape

    mangoFilt = _mango_open_filters._SubSample(start, stop, step)
    ssDds = mangoFilt(input)

    return ssDds

def gather_slice(
    input,
    axis=0,
    index=0,
    rank=None
):
    """
    Gathers a 2D slice of a 3D :obj:`mango.Dds` array to a single process. 
    
    :type input: :obj:`mango.Dds`
    :param input: :obj:`mango.Dds` image from which the slice is copied.
    :type axis: int
    :param axis: The indexing axis from which the slice is copied.
       For example, :samp:`axis=0` specifies an xy slice, :samp:`axis=1`
       specifies an xz slice and :samp:`axis=2` specifies a yz slice.
    :type index: int
    :param index: The global index (relative to :samp:`input.origin`) of the slice.
    :type rank: int or None
    :param rank: The MPI rank of the process where the slice is copied. If :samp:`rank=None`
       the slice is gathered to all processes.

    :rtype: :obj:`mango.Dds`
    :return: Slice image.
    
    For example::
       >>> import mango;
       >>> dds = mango.zeros(origin=(16,-32,64), shape=(32,64,128))
       >>> slcDds = mango.image.gather_slice(dds, axis=0, index=dds.shape[0]//2, rank=None)
       >>> print("slcDds.origin = %s" % (slcDds.origin,))
       slcDds.origin = [ 32 -32  63]
       >>> print("slcDds.shape  = %s" % (slcDds.shape,))
       slcDds.shape  = [ 1 64 128]

    """
    
    mpidims = sp.zeros_like(input.mpi.shape)
    origin = sp.array(input.origin, dtype="int32")
    shape = sp.array(input.shape, dtype="int32")
    subdorigin = sp.array(input.origin, dtype="int32")
    if ((rank == None) or (not mpi.haveMpi4py) or (rank == input.mpi.comm.Get_rank())):
        subdshape = sp.array(input.shape, dtype="int32")
        subdshape[axis] = 1
    else:
        subdshape = sp.array([0,0,0], dtype="int32")
        subdorigin = sp.array(input.origin, dtype="int32")

    subdorigin[axis] += index
    origin[axis] += index
    shape[axis] = 1
    mpidims[axis] = 1
        
    slcDds = core.empty_like(input, shape=shape, origin=origin, subdshape=subdshape, subdorigin=subdorigin, mpidims=mpidims)
    slcDds.fill(input)
    
    return slcDds

def resample(
    input,
    interptype=InterpolationType.CATMULL_ROM_CUBIC_SPLINE,
    factor=None,
    voxsz=None,
    voxunit=None
):
    """
    Re-sample :obj:`mango.Dds` object on new point-grid.
    
    :type input: :obj:`mango.Dds`
    :param input: :obj:`mango.Dds` image which is to be re-sampled.
    :type interptype: :obj:`InterpolationType`
    :param interptype: Type of interpolation for re-sampling, see :obj:`InterpolationType`.
    :type factor: 3-sequence of float
    :param factor: Sampling rate, :samp:`factor=(2.0,2.0,2.0)` returns
       an image which is 8 times the size (number of voxels) of the input.
       If :samp:`factor=(0.5,0.5,0.5)` returns an image which is 1/8
       times the size (number of voxels) of the input image.
    :type voxsz: 3-sequence of float
    :param voxsz: New voxel size, the re-sampling rate/factor is
       calculated as :samp:`factor=input.md.getVoxelSize()/voxsz`
       (with appropriate size-unit conversion).
    :type voxunit: :obj:`str`
    :param voxunit: String indicating the length unit of the :samp:`voxsz`
       argument.

    :rtype: :obj:`mango.Dds`
    :return: Re-sampled image.
    """
    
    if ((factor is None) and (voxsz is None)):
        raise ValueError("Need to specify one of 'factor' or 'voxsz' arguments, got None for both.")
    elif ((not (factor is None)) and (not (voxsz is None))):
        raise ValueError("Need to specify one of 'factor' or 'voxsz' arguments, both were specified non-None.")

    if (sp.any(input.md.getVoxelSize() <= 0)):
        raise ValueError("Non-positive voxel size, input.md.getVoxelSize()=%s." % (input.md.getVoxelSize(),))
    
    if (voxunit is None):
        voxunit = input.md.getVoxelSizeUnit() 

    mangoFilt = _mango_open_filters._ResampleGrid(interptype, factor, voxsz, voxunit)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    return filt(input)

class _DiscreteGaussianBlur(_mango_open_filters._DiscreteGaussian):
    def __call__(self, input):
        return _mango_open_filters._DiscreteGaussian.blur(self, input)

class _DiscreteGaussianGradientMagnitude(_mango_open_filters._DiscreteGaussian):
    def __call__(self, input):
        return _mango_open_filters._DiscreteGaussian.gradientMagnitude(self, input)

class _DiscreteGaussianMeanAndStdd(_mango_open_filters._DiscreteGaussian):
    def __call__(self, input):
        return _mango_open_filters._DiscreteGaussian.meanAndStdd(self, input)

class _DiscreteGaussianStdd(_mango_open_filters._DiscreteGaussian):
    def __init__(self, sigma=1.0, errtol=0.01, mean=None):
        _mango_open_filters._DiscreteGaussian.__init__(self, sigma, errtol)
        self.mean = mean
        
    def __call__(self, input):
        return _mango_open_filters._DiscreteGaussian.stdd(self, input, self.mean)

def discrete_gaussian_kernel(
    sigma = 1.0,
    errtol=0.01,
    dim=3
):
    """
    Discrete-Gaussian convolution kernel.
    
    :type sigma: float
    :param sigma: The Discrete-Gaussian standard deviation.
    :type errtol: float
    :param errtol: 1D Error tolerance which determines the size of the kernel window,
       smaller value gives larger kernel window.
    :type dim: int
    :param dim: Spatial dimension of kernel.
    :rtype: :obj:`numpy.array`
    :return: A :samp:`dim` dimensional Discrete Gaussian kernel.
    """
    return _DiscreteGaussianBlur(sigma, errtol).getKernel(dim)

def discrete_gaussian_laplacian_kernel(
    sigma = 1.0,
    errtol=0.01,
    dim=3
):
    """
    Discrete-Gaussian Laplacian kernel.
    
    :type sigma: float
    :param sigma: The Discrete-Gaussian standard deviation.
    :type errtol: float
    :param errtol: 1D Error tolerance which determines the size of the kernel window,
       smaller value gives larger kernel window.
    :type dim: int
    :param dim: Spatial dimension of kernel.
    :rtype: :obj:`numpy.array`
    :return: A :samp:`dim` dimensional Discrete Gaussian Laplacian kernel.
    """
    return _DiscreteGaussianBlur(sigma, errtol).getLaplacianKernel(dim)

def discrete_gaussian_gradient_kernel(
    axis = 0,
    sigma = 1.0,
    errtol=0.01,
    dim=3
):
    """
    Discrete-Gaussian convolution kernel.
    
    :type sigma: float
    :param sigma: The Discrete-Gaussian standard deviation.
    :type errtol: float
    :param errtol: 1D Error tolerance which determines the size of the kernel window,
       smaller value gives larger kernel window.
    :type dim: int
    :param dim: Spatial dimension of kernel.
    :rtype: :obj:`numpy.array`
    :return: A :samp:`dim` dimensional Discrete Gaussian kernel.
    """
    return _DiscreteGaussianGradientMagnitude(sigma, errtol).getGradientKernel(axis, dim)

def discrete_gaussian(
    input,
    sigma = 1.0,
    errtol=0.01
):
    """
    Discrete-Gaussian image convolution.
    
    :type input: :obj:`mango.Dds`
    :param input: Image for convolution.
    :type sigma: float
    :param sigma: The Discrete-Gaussian standard deviation.
    :type errtol: float
    :param errtol: Error tolerance which determines the size of the kernel window.
    :rtype: :obj:`mango.Dds`
    :return: Convolved image.
    """
    mangoFilt = _DiscreteGaussianBlur(sigma, errtol)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    blurDds = filt(input)
    return blurDds

def discrete_gaussian_gradient_magnitude(
    input,
    sigma = 1.0,
    errtol=0.01
):
    """
    Discrete-Gaussian gradient magnitude (equavalent to
    Discrete Gaussian blur followed by gradient magnitude).
    
    :type input: :obj:`mango.Dds`
    :param input: Image for gradient magnitude.
    :type sigma: float
    :param sigma: The Discrete-Gaussian standard deviation.
    :type errtol: float
    :param errtol: Error tolerance which determines the size of the kernel window.
    :rtype: :obj:`mango.Dds`
    :return: Gradient magnitude image.
    """
    mangoFilt = _DiscreteGaussianGradientMagnitude(sigma, errtol)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    grdDds = filt(input)
    return grdDds

def discrete_gaussian_mean_stdd(
    input,
    sigma = 1.0,
    errtol=0.01
):
    """
    Discrete-Gaussian weighted mean and weighted standard-deviation
    image calculation.
    
    :type input: :obj:`mango.Dds`
    :param input: Image for Discrete-Gaussian weighted mean and standard-deviation calculation.
    :type sigma: float
    :param sigma: The Discrete-Gaussian standard deviation.
    :type errtol: float
    :param errtol: Error tolerance which determines the size of the kernel window.
    :rtype: pair of :obj:`mango.Dds`
    :return: :samp:`[meanDds, stddDds]` image pair.
    """
    mangoFilt = _DiscreteGaussianMeanAndStdd(sigma, errtol)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    meanAndStdd = filt(input)
    return meanAndStdd

def discrete_gaussian_stdd(
    input,
    sigma = 1.0,
    errtol=0.01,
    mean=None
):
    """
    Discrete-Gaussian weighted standard-deviation
    image calculation.
    
    :type input: :obj:`mango.Dds`
    :param input: Image for Discrete-Gaussian weighted standard-deviation calculation.
    :type sigma: float
    :param sigma: The Discrete-Gaussian standard deviation.
    :type errtol: float
    :param errtol: Error tolerance which determines the size of the kernel window.
    :type mean: float
    :param mean: Mean value used in standard deviation calculation. If :samp:`None` uses
       local neighbourhood mean estimate.
    :rtype: pair of :obj:`mango.Dds`
    :return: :samp:`[meanDds, stddDds]` image pair.
    """
    mangoFilt = _DiscreteGaussianStdd(sigma, errtol, mean)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    stdd = filt(input)
    return stdd

def gaussian_downsample(
    input,
    sigma = None,
    interptype=InterpolationType.CATMULL_ROM_CUBIC_SPLINE,
    factor=None,
    voxsz=None,
    voxunit=None
):
    """
    Downsample an image (:obj:`mango.Dds`) by blurring with a Discrete Gaussian
    kernel and resampling on a sparse grid.
    
    :type input: :obj:`mango.Dds`
    :param input: :obj:`mango.Dds` image which is to be downsampled.
    :type sigma: :obj:`float`
    :param sigma: The scale (standard deviation) of the Discrete Gaussian filter.
       If :samp:`None`, default is :samp:`sigma=2.0*(1.0/factor)/6.0`.
    :type interptype: :obj:`InterpolationType`
    :param interptype: Type of interpolation for re-sampling, see :obj:`InterpolationType`.
    :type factor: 3-sequence of float
    :param factor: Sampling rate, :samp:`factor=(2.0,2.0,2.0)` returns
       an image which is 8 times the size (number of voxels) of the input.
       If :samp:`factor=(0.5,0.5,0.5)` returns an image which is 1/8
       times the size (number of voxels) of the input image.
    :type voxsz: 3-sequence of float
    :param voxsz: New voxel size, the re-sampling rate/factor is
       calculated as :samp:`factor=input.md.getVoxelSize()/voxsz`
       (with appropriate size-unit conversion).
    :type voxunit: :obj:`str`
    :param voxunit: String indicating the length unit of the :samp:`voxsz`
       argument.

    :rtype: :obj:`mango.Dds`
    :return: Re-sampled image.
    """
    
    if ((factor is None) and (voxsz is None)):
        raise ValueError("Need to specify one of 'factor' or 'voxsz' arguments, got None for both.")
    elif ((not (factor is None)) and (not (voxsz is None))):
        raise ValueError("Need to specify one of 'factor' or 'voxsz' arguments, both were specified non-None.")

    if (sp.any(input.md.getVoxelSize() <= 0)):
        raise ValueError("Non-positive voxel size, input.md.getVoxelSize()=%s." % (input.md.getVoxelSize(),))
    
    if (voxunit is None):
        voxunit = input.md.getVoxelSizeUnit()
    
    if (factor is None):
        factor = input.md.getVoxelSize(voxunit)/voxsz
    
    if (min(factor) != max(factor)):
        raise ValueError("Non-symmetric Gaussian blur not yet supported, got non-symmetric factor=%s." % (factor,))

    if (sigma is None):
        sigma = max(2.0*(1.0/sp.array(factor))/6.0)
    
    #
    # Separate the downsampling into an integer part
    # and a floating-point part. The integer downsampling
    # can be achieved with a stride (which means we don't
    # have to convolve for every voxel in the image, just
    # the strided voxels).
    # The floating point part is evaluated with resampling.
    #
    fltStride = 1.0/sp.array(factor, dtype="float64")
    stride = sp.array(sp.floor(fltStride), dtype="int32")

    intFactor = 1.0/sp.array(stride, dtype="float64")
    fltFactor = factor - intFactor
    gaussianKernel = discrete_gaussian_kernel(sigma)
    blurDds = convolve(input, gaussianKernel, stride=stride)

    remainderFactor = fltFactor/intFactor + 1
    if (sp.any(remainderFactor != 1)):
        rspDds = resample(input=blurDds, interptype=interptype, factor=remainderFactor)
    else:
        rspDds = blurDds

    return rspDds

def mean_filter(input, se=box_se(shape=(3,3,3)), mode="mirror", cval=None, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0)):
    """
    Neighbourhood mean filter.
    
    :type input: :obj:`mango.Dds`
    :param input: Image to be filtered.
    :type se: :obj:`StructuringElement`.
    :param se: Structuring element which defines the neighbourhood over which the means are calculated.
    :type mode: str
    :param mode: String indicating how to handle borders, one of
        "mirror", "constant", "reflect".
    :type cval: scalar
    :param cval: When :samp:`mode=='constant'` use this value
        to initialise borders.
    :type stride: 3-sequence
    :param stride: The sub-sampling step-size for each dimension.
    :type boffset: 3-sequence
    :param boffset: The offset (:samp:`(bz,by,bx)` relative to :samp:`input.origin`)
       at which the filtering starts.
    :type eoffset: 3-sequence
    :param eoffset: The offset (:samp:`(ez,ey,ex)` relative to :samp:`(input.origin + input.shape)`)
       at which the filtering stops.
    
    :rtype: :obj:`mango.Dds`
    :return: Mean-filtered image.
    """
    mangoFilt = _NeighbourhoodFilter(NeighbourhoodFilterType.MEAN, se, stride, boffset, eoffset, mode, cval)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    meanDds = filt(input, mode=mode, cval=cval)
    return meanDds

def median_filter(input, se=box_se(shape=(3,3,3)), mode="mirror", cval=None, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0)):
    """
    Neighbourhood median filter.
    
    :type input: :obj:`mango.Dds`
    :param input: Image to be filtered.
    :type se: :obj:`StructuringElement`.
    :param se: Structuring element which defines the neighbourhood over which the medians are calculated.
    :type mode: str
    :param mode: String indicating how to handle borders, one of
        "mirror", "constant", "reflect".
    :type cval: scalar
    :param cval: When :samp:`mode=='constant'` use this value
        to initialise borders.
    :type stride: 3-sequence
    :param stride: The sub-sampling step-size for each dimension.
    :type boffset: 3-sequence
    :param boffset: The offset (:samp:`(bz,by,bx)` relative to :samp:`input.origin`)
       at which the filtering starts.
    :type eoffset: 3-sequence
    :param eoffset: The offset (:samp:`(ez,ey,ex)` relative to :samp:`(input.origin + input.shape)`)
       at which the filtering stops.
    
    :rtype: :obj:`mango.Dds`
    :return: Median-filtered image.
    """
    mangoFilt = _NeighbourhoodFilter(NeighbourhoodFilterType.MEDIAN, se, stride, boffset, eoffset, mode, cval)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    medianDds = filt(input, mode=mode, cval=cval)
    return medianDds

def stdd_filter(input, se=box_se(shape=(3,3,3)), mode="mirror", cval=None, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0)):
    """
    Neighbourhood standard deviation filter.
    
    :type input: :obj:`mango.Dds`
    :param input: Image to be filtered.
    :type se: :obj:`StructuringElement`.
    :param se: Structuring element which defines the neighbourhood over which the standard deviations
       are calculated.
    :type mode: str
    :param mode: String indicating how to handle borders, one of
        "mirror", "constant", "reflect".
    :type cval: scalar
    :param cval: When :samp:`mode=='constant'` use this value
        to initialise borders.
    :type stride: 3-sequence
    :param stride: The sub-sampling step-size for each dimension.
    :type boffset: 3-sequence
    :param boffset: The offset (:samp:`(bz,by,bx)` relative to :samp:`input.origin`)
       at which the filtering starts.
    :type eoffset: 3-sequence
    :param eoffset: The offset (:samp:`(ez,ey,ex)` relative to :samp:`(input.origin + input.shape)`)
       at which the filtering stops.
    
    :rtype: :obj:`mango.Dds`
    :return: Standard-deviation-filtered image.
    """
    mangoFilt = _NeighbourhoodFilter(NeighbourhoodFilterType.STANDARD_DEVIATION, se, stride, boffset, eoffset, mode, cval)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    stddDds = filt(input, mode=mode, cval=cval)
    return stddDds

def mad_filter(input, se=box_se(shape=(3,3,3)), mode="mirror", cval=None, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0)):
    """
    Neighbourhood median absolute deviation filter.
    
    :type input: :obj:`mango.Dds`
    :param input: Image to be filtered.
    :type se: :obj:`StructuringElement`.
    :param se: Structuring element which defines the neighbourhood over which the
        median absolute deviations are calculated.
    :type mode: str
    :param mode: String indicating how to handle borders, one of
        "mirror", "constant", "reflect".
    :type cval: scalar
    :param cval: When :samp:`mode=='constant'` use this value
        to initialise borders.
    :type stride: 3-sequence
    :param stride: The sub-sampling step-size for each dimension.
    :type boffset: 3-sequence
    :param boffset: The offset (:samp:`(bz,by,bx)` relative to :samp:`input.origin`)
       at which the filtering starts.
    :type eoffset: 3-sequence
    :param eoffset: The offset (:samp:`(ez,ey,ex)` relative to :samp:`(input.origin + input.shape)`)
       at which the filtering stops.
    
    :rtype: :obj:`mango.Dds`
    :return: Median-absolute-deviation-filtered image.
    """
    mangoFilt = _NeighbourhoodFilter(NeighbourhoodFilterType.MEDIAN_ABSOLUTE_DEVIATION, se, stride, boffset, eoffset, mode, cval)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    madDds = filt(input, mode=mode, cval=cval)
    return madDds

def bilateral_filter(input, intensity_sigma, spatial_sigma=0.4, se = None, mode="mirror", cval=None, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0), normWeights=False):
    """
    3D bilateral filter.
    
    :type input: :obj:`mango.Dds`
    :param input: Image to be filtered.
    :type intensity_sigma: :obj:`float`.
    :param intensity_sigma: The Gaussian sigma value for intensity weighting.
    :type spatial_sigma: :obj:`float`.
    :param spatial_sigma: The Gaussian sigma value for spatial neighbourhood weighting.
    :type se: :obj:`StructuringElement`.
    :param se: Structuring element which defines the neighbourhood over which the
        bilateral filtering is calculated. If :samp:`None`, auto-calculated from
        the :samp:`spatial_sigma` value. 
    :type mode: str
    :param mode: String indicating how to handle borders, one of
        "mirror", "constant", "reflect".
    :type cval: scalar
    :param cval: When :samp:`mode=='constant'` use this value
        to initialise borders.
    :type stride: 3-sequence
    :param stride: The sub-sampling step-size for each dimension.
    :type boffset: 3-sequence
    :param boffset: The offset (:samp:`(bz,by,bx)` relative to :samp:`input.origin`)
       at which the filtering starts.
    :type eoffset: 3-sequence
    :param eoffset: The offset (:samp:`(ez,ey,ex)` relative to :samp:`(input.origin + input.shape)`)
       at which the filtering stops.
    
    :rtype: :obj:`mango.Dds`
    :return: Bilateral filtered image.
    """
    mangoFilt = \
        _NeighbourhoodFilter(
            NeighbourhoodFilterType.BILATERAL,
            se,
            stride,
            boffset,
            eoffset,
            mode,
            cval,
            {
                "intensity_sigma" : intensity_sigma,
                "spatial_sigma" : spatial_sigma
            }
        )
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    convDds = filt(input, mode=mode, cval=cval)
    return convDds

def histogramdd(inputList, bins, rootRank = None):
    """
    Returns tuple pair :samp:`(H,edges)` of multi-dimensional histogram counts
    and bin edges. Not mask-aware.
    
    :type inputList: sequence of :obj:`mango.Dds`
    :param inputList: A :samp:`n` length sequence of :obj:`mango.Dds` objects.
    :type bins: sequence or :obj:`int`
    :param bins: The bins edge points, or number of bins.
    :rtype: :obj:`tuple`
    :return: :samp:`(H,edges)` histogram and bin-edges pair.
    """
    
    rootLogger.debug("Calculating histogram range...")
    range = [list(core.minmax(input)) for input in inputList]
    rootLogger.debug("Done calculating histogram range.")
    
    rootLogger.debug("Calculating sub-domain histograms...")
    arrList = [input.subd.asarray() for input in inputList]
    arrList = [arr.reshape((arr.size,)) for arr in arrList]
    H, edges = np.histogramdd(np.column_stack(arrList), bins=bins, range=range)
    
    rootLogger.debug("Done calculating sub-domain histograms.")
    logger.debug("H=\n%s\nedges=\n%s" % (H, edges))
    if (inputList[0].mpi.comm != None):
        rootLogger.debug("Reducing histogram counts...")
        comm = inputList[0].mpi.comm
        hSrc = sp.array(H.flatten(), copy=True, dtype=H.dtype)
        hDst = sp.array(H.flatten(), copy=True, dtype=H.dtype)
        if (rootRank == None):
            comm.Allreduce(hSrc, hDst, op=mpi.SUM)
        else:
            comm.Reduce(hSrc, hDst, op=mpi.SUM, root=rootRank)
            edges = None
        H = hDst.reshape(H.shape)

        rootLogger.debug("Done reducing histogram counts.")

    return H, edges

# Sphinx documentation class only
class SphericalHistogram (_mango_open_filters._SphericalHistogram_float64):
    __doc__ = _mango_open_filters._SphericalHistogram_float64.__doc__
    pass

def spherical_histogram(sphere_c=(0,0,0), sphere_r=1.0, tri_rad_bnd=0.1, tri_cdst_bnd=0.1, tri_angl_bound=30.0):
    """
    Factory function for creating :obj:`SphericalHistogram` object.
    
    :type sphere_c: 3 sequence
    :param sphere_c: The centre coordinate of the sphere.
    :type sphere_r: :obj:`float`
    :param sphere_r: The centre coordinate of the sphere.
    :type tri_rad_bnd: :obj:`float`
    :param tri_rad_bnd: Bound on the sphere radius for Delauny triangulation of sphere.
    :type tri_cdst_bnd: :obj:`float`
    :param tri_cdst_bnd: Bound on the sphere-centre-point distance Delauny triangulation of sphere.
    :type tri_angl_bnd: :obj:`float`
    :param tri_angl_bnd: Degrees, lower bound on the triangle angles, greater than 30 degrees may be impossible.
    :rtype: :obj:`SphericalHistogram`
    :return: Zero count initialised :obj:`SphericalHistogram` object.
    """
    
    return \
        _mango_open_filters._SphericalHistogram_float64(
            sphere_c=sphere_c,
            sphere_r=sphere_r,
            tri_rad_bnd=tri_rad_bnd,
            tri_cdst_bnd=tri_cdst_bnd,
            tri_angl_bnd=tri_angl_bound
        )

def intensity_spherical_histogram(input, sphhist, centreoffset=None, usevoxsz=False, voxunit="mm"):
    """
    Populates bins of the specified spherical histogram with *intensity counts* from
    the :samp:`input` :obj:`mango.Dds` image.
    
    :type input: :obj:`mango.Dds`
    :param input: Intensity *counts* from this image/array are added to the
       bins of the :samp:`sphhist` spherical histogram.
    :type sphhist: :obj:`SphericalHistogram`
    :param sphhist: Bins of this histogram are populated using data from
       the :samp:`input` image.
       The sphere centre (:attr:`SphericalHistogram.sphere_c`) is assumed to be a"
       global-index relative to the :samp:`input.origin`."
    :type centreoffset: 3 sequence
    :param centreoffset: If not :samp:`None` the centre of the spherical histogram
       is moved to this index (i.e. moved to global index :samp:`input.origin + centreoffset`)
       before being populated with image *count* data. If :samp:`None`,
       the :attr:`SphericalHistogram.sphere_c` is used as the :samp:`centreoffset`.
    :type usevoxsz: :obj:`bool`
    :param usevoxsz: If :samp:`True`, indices are converted to spatial coordinates
       by multiplying by the voxel size.
    :type voxunit: :obj:`str`
    :param voxunit: Length-unit used for spatial coordinate conversion
       (e.g. :samp:`'nm','um','mm','cm',...`)

    """
    
    sphHistPopulator = _mango_open_filters._ImageSphericalHistogram_float64()
    sphHistPopulator.addIntensitySamples(
        input,
        sphhist,
        centreoffset,
        usevoxsz,
        voxunit
    )

def distance_spherical_histogram(input, sphhist, centreoffset=None, usevoxsz=False, voxunit="mm"):
    """
    Populates bins of the specified spherical histogram with *distance counts* from
    the :samp:`input` :obj:`mango.Dds` image. Distance is the distance from a voxel
    coordinate to the :attr:`SphericalHistogram.sphere_c` coordinate. 
    
    :type input: :obj:`mango.Dds`
    :param input: Intensity *counts* from this image/array are added to the
       bins of the :samp:`sphhist` spherical histogram.
    :type sphhist: :obj:`SphericalHistogram`
    :param sphhist: Bins of this histogram are populated using data from
       the :samp:`input` image.
       The sphere centre (:attr:`SphericalHistogram.sphere_c`) is assumed to be a"
       global-index relative to the :samp:`input.origin`."
    :type centreoffset: 3 sequence
    :param centreoffset: If not :samp:`None` the centre of the spherical histogram
       is moved to this index (i.e. moved to global index :samp:`input.origin + centreoffset`)
       before being populated with image *count* data. If :samp:`None`,
       the :attr:`SphericalHistogram.sphere_c` is used as the :samp:`centreoffset`.
    :type usevoxsz: :obj:`bool`
    :param usevoxsz: If :samp:`True`, indices are converted to spatial coordinates
       by multiplying by the voxel size.
    :type voxunit: :obj:`str`
    :param voxunit: Length-unit used for spatial coordinate conversion
       (e.g. :samp:`'nm','um','mm','cm',...`)

    """
    
    sphHistPopulator = _mango_open_filters._ImageSphericalHistogram_float64()
    sphHistPopulator.addDistanceSamples(
        input,
        sphhist,
        centreoffset,
        usevoxsz,
        voxunit
    )

def intensity_mult_distance_spherical_histogram(input, sphhist, centreoffset=None, usevoxsz=False, voxunit="mm"):
    """
    Populates bins of the specified spherical histogram with *intensity times distance counts* from
    the :samp:`input` :obj:`mango.Dds` image. Distance is the distance from a voxel
    coordinate to the :attr:`SphericalHistogram.sphere_c` coordinate. 
    
    :type input: :obj:`mango.Dds`
    :param input: Intensity *counts* from this image/array are added to the
       bins of the :samp:`sphhist` spherical histogram.
    :type sphhist: :obj:`SphericalHistogram`
    :param sphhist: Bins of this histogram are populated using data from
       the :samp:`input` image.
    :type centreoffset: 3 sequence
    :param centreoffset: If not :samp:`None` the centre of the spherical histogram
       is moved to this index (i.e. moved to global-index :samp:`input.origin + centreoffset`)
       before being populated with image *count* data. If :samp:`None`,
       the :attr:`SphericalHistogram.sphere_c` is used as the :samp:`centreoffset`.
    :type usevoxsz: :obj:`bool`
    :param usevoxsz: If :samp:`True`, indices are converted to spatial coordinates
       by multiplying by the voxel size.
    :type voxunit: :obj:`str`
    :param voxunit: Length-unit used for spatial coordinate conversion
       (e.g. :samp:`'nm','um','mm','cm',...`)

    """
    
    sphHistPopulator = _mango_open_filters._ImageSphericalHistogram_float64()
    sphHistPopulator.addIntensityTimesDistanceSamples(
        input,
        sphhist,
        centreoffset,
        usevoxsz,
        voxunit
    )

class _DdsLabeler:
    """
    Label connected components of segmented image.
    """
    def __init__(self, val, connectivity, dosort):
        self.val = val
        self.connectivity = connectivity
        self.dosort = dosort
    
    def getRequiredHaloSize(self):
        return 1
    
    def __call__(self, input):
        return \
            _mango_open_filters._doMakeClusterLabels(
                input,
                val=self.val,
                connectivity=self.connectivity,
                dosort=self.dosort
            )
def label(input, val, connectivity=6, dosort=True):
    """
    Generates image of labeled connected components.
    
    :type input: :obj:`mango.Dds`
    :param input: Image to label (must have :samp:`input.mtype == mtype("segmented")`).
    :type val: :obj:`int`
    :param val: The value in the :samp:`input` image for which connected component labels
        are generated.
    :type connectivity: :obj:`int`
    :param connectivity: Connectivity structuring elements type, only :samp:`6`-neighbour
       or :samp:`26` neighbour connectivity.
    :type dosort: :obj:`bool`
    :param dosort: Whether to *sort* the component labels. Sorting generates a label
        sequence which is related to the global voxel index of the labeled components.
        Sorting produces consistant labeling no matter the MPI layout of the input data.
    :rtype: :obj:`mango.Dds`
    :return: An image with connected components each having a unique label.
    """
    mangoFilt = _DdsLabeler(val=val, connectivity=connectivity, dosort=dosort)
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    lblDds = filt(input, mode="reflect")
    return lblDds

def eliminate_labels_by_size(input, minsz=0, maxsz=None, val=0, labels_are_connected=True):
    """
    Eliminates labeled objects (clusters) according to size (number of voxels).
    Any labeled object which has size :samp:`sz in range(minsz, maxsz+1)` will
    be *eliminated* from the image by replacing the label with :samp:`val`.
    
    :type input: :obj:`mango.Dds`
    :param input: A label image (must have :samp:`input.mtype == mtype("labels")`).
    :type minsz: :obj:`int`
    :param minsz: The minimum size of the clusters which are to be eliminated.
    :type maxsz: :obj:`int`
    :param maxsz: The maximum size of the clusters which are to be eliminated.
    :type val: :obj:`int`
    :param val: The label-value assigned to voxels which are members of eliminated labeled-objects. 
    :type labels_are_connected: :obj:`bool`
    :param labels_are_connected: If :samp:`True`, assumes labeled-objects are (26-neighbour) connected,
        and uses a less-memory-hungary MPI parallel algorithm for determining the
        size of the labeled objects/clusters.
    :rtype: :obj:`mango.Dds`
    :return: A label image with specifically sized labeled-objects eliminated.
    """
    if (maxsz == None):
        maxsz = int(sp.product(input.shape)+1)

    mangoFilt = \
        _mango_open_filters._RemoveClustersBySize(
            minsz=minsz,
            maxsz=maxsz,
            val=val,
            labels_are_connected=labels_are_connected
        )
    filt = _DdsMangoFilterApplier(mangoFilt)
    
    lblDds = filt(input, mode="constant", cval=None)
    return lblDds

def convex_hull_2d(input, axis=0, inputmsk=None, outhull=None, inhull=None, copy=False, copyoutside=True):
    """
    Calculates 2D convex hull of non-masked voxels in each slice of an image.
    By default, returns an image where inside-hull-voxels have value :samp:`inhull`
    and outside-hull-voxels have value :samp:`outhull`.

    :type input: :obj:`mango.Dds`
    :param input: Finds the convex hull of voxels which are not
       masked (masked voxels have value :samp:`input.mtype.maskValue()` or value :samp:`inputmsk`).
    :type axis: :obj:`int`
    :param axis: Axis for the per-slice 2D convex hulls (:samp:`0` is z-axis, :samp:`1` is y-axis
       and :samp:`2` is x-axis).
    :type inputmsk: :samp:`input.dtype` or :samp:`None`
    :param inputmsk: Additional mask value to exclude voxels from convex-hull calculation.
       If :samp:`None`, :samp:`inputmsk=input.mtype.maskValue()` or if :samp:`input` doesn't
       have an :samp:`mtype` attribute :samp:`inputmsk=0`.
    :type outhull: :samp:`input.dtype` or :samp:`None`
    :param outhull: Value for voxels which lie *outside* the convex-hull.
       If :samp:`None`, :samp:`outhull=input.mtype.maskValue()` or if :samp:`input` doesn't
       have an :samp:`mtype` attribute :samp:`outhull=0`.
    :type inhull: :samp:`input.dtype` or :samp:`None`
    :param inhull: Value for voxels which lie *inside* the convex-hull.
       If :samp:`None`, :samp:`inhull=1`.
    :type copy: :obj:`bool`
    :param copy: If :samp:`True`, returns a copy of the :samp:`input` image with
        the outside-hull voxels (:samp:`copyoutside=True`) set to :samp:`outhull`.
    :type copyoutside: :obj:`bool`
    :param copyoutside: If :samp:`True` (and :samp:`copy=True`), the returned image
        is a copy of the :samp:`input` image with the *outside*-hull voxels
        set to :samp:`outhull`.
        If :samp:`False` (and :samp:`copy=True`), the returned image
        is a copy of the :samp:`input` image with the *inside*-hull voxels
        set to :samp:`inhull`. 
    """
    if (inputmsk == None):
        inputmsk = 0
        if (hasattr(input, "mtype") and input.mtype != None):
            inputmsk = input.mtype.maskValue()
    if (outhull == None):
        outhull = 0
        if (hasattr(input, "mtype") and input.mtype != None):
            outhull = input.mtype.maskValue()
    if (inhull == None):
        inhull = 1
    
    if (inhull == outhull):
        raise ValueError("inhull=%s is equal to outhull=%s" % (inhull, outhull))
    
    return _mango_open_filters._convex_hull_2d(
        input=input,
        axis=axis,
        inputmsk=inputmsk,
        inhull=inhull,
        outhull=outhull,
        copy=copy,
        copyoutside=copyoutside
    )

def convex_hull_3d(input, inputmsk=None, outhull=None, inhull=None, copy=False, copyoutside=True):
    """
    Calculates 3D convex hull of non-masked voxels of an image.
    By default, returns an image where inside-hull-voxels have value :samp:`inhull`
    and outside-hull-voxels have value :samp:`outhull`.

    :type input: :obj:`mango.Dds`
    :param input: Finds the convex hull of voxels which are not
       masked (masked voxels have value :samp:`input.mtype.maskValue()` or value :samp:`inputmsk`).
    :type inputmsk: :samp:`input.dtype` or :samp:`None`
    :param inputmsk: Additional mask value to exclude voxels from convex-hull calculation.
       If :samp:`None`, :samp:`inputmsk=input.mtype.maskValue()` or if :samp:`input` doesn't
       have an :samp:`mtype` attribute :samp:`inputmsk=0`.
    :type outhull: :samp:`input.dtype` or :samp:`None`
    :param outhull: Value for voxels which lie *outside* the convex-hull.
       If :samp:`None`, :samp:`outhull=input.mtype.maskValue()` or if :samp:`input` doesn't
       have an :samp:`mtype` attribute :samp:`outhull=0`.
    :type inhull: :samp:`input.dtype` or :samp:`None`
    :param inhull: Value for voxels which lie *inside* the convex-hull.
       If :samp:`None`, :samp:`inhull=1`.
    :type copy: :obj:`bool`
    :param copy: If :samp:`True`, returns a copy of the :samp:`input` image with
        the outside-hull voxels (:samp:`copyoutside=True`) set to :samp:`outhull`.
    :type copyoutside: :obj:`bool`
    :param copyoutside: If :samp:`True` (and :samp:`copy=True`), the returned image
        is a copy of the :samp:`input` image with the *outside*-hull voxels
        set to :samp:`outhull`.
        If :samp:`False` (and :samp:`copy=True`), the returned image
        is a copy of the :samp:`input` image with the *inside*-hull voxels
        set to :samp:`inhull`. 
    """
    if (inputmsk == None):
        inputmsk = 0
        if (hasattr(input, "mtype") and input.mtype != None):
            inputmsk = input.mtype.maskValue()
    if (outhull == None):
        outhull = 0
        if (hasattr(input, "mtype") and input.mtype != None):
            outhull = input.mtype.maskValue()
    if (inhull == None):
        inhull = 1
    
    if (inhull == outhull):
        raise ValueError("inhull=%s is equal to outhull=%s" % (inhull, outhull))
    
    return _mango_open_filters._convex_hull_3d(
        input=input,
        inputmsk=inputmsk,
        inhull=inhull,
        outhull=outhull,
        copy=copy,
        copyoutside=copyoutside
    )

def distance_transform_edt(input, val=0):
    """
    Returns Euclidean distance transform image. Calculates for *non-obstacle*
    voxels (i.e. voxels with value :samp:`val`) the distance to the nearest *obstacle* voxel.
    
    :type dds: :obj:`mango.Dds`
    :param dds: The :obj:`mango.Dds` from which distances are
       created (must have :samp:`input.mtype == mango.mtype('segmented')`.
    :type val: :obj:`int`
    :param val: Value for non-obstacle :samp:`input` voxels, for all non-obstacle voxels,
       calculates distance to nearest obstacle voxel.
    :rtype: :obj:`mango.Dds`
    :return: Euclidean distance image :samp:`output.mtype == mango.mtype('distance_map')`.,

    """
    return _mango_open_filters._calcEuclideanDistance(input, val)


def max_covering_radius(input, maxdist=200.0, filecache=False):
    """
    Calculates the maximal covering radius transform from the input
    Euclidean distance transform image.
    
    :type input: :obj:`mango.Dds`
    :param input: Euclidean distance transform image (:func:`distance_transform_edt`)
       for which the maximal covering sphere radii are calculated.
    :type maxdist: :obj:`float`
    :param maxdist: Upper bound on the covering sphere radii.
    :type filecache: :obj:`bool`
    :param filecache: If :samp:`True`, uses cache files to store some of
       the temporary computation data.
    """
    return _mango_open_filters._calcMaxCoveringSphere2(input, maxdist, filecache)
