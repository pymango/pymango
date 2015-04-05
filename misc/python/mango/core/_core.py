# The _mango_open_core module is a shared-object library
# which exports python functions and classes.
import mango.mpi
import sys
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_open_ndarray_converters as _mango_open_ndarray_converters_so
    from . import _mango_open_core as _mango_open_core_so
    sys.setdlopenflags(_flags)
else:
    from . import _mango_open_ndarray_converters as _mango_open_ndarray_converters_so
    from . import _mango_open_core as _mango_open_core_so

from ._mango_open_core import *

import scipy as sp
import numpy as np
_mango_open_core_so._setNumpyDTypeType(type(sp.dtype("int32")))
import re
import sys
from mango.utils import ModuleObjectFactory as _ModuleObjectFactory

logger,rootLogger = mango.mpi.getLoggers(__name__)

if (mango.mpi.haveMpi4py):
    _mango_open_core_so._setMpi4pyPyMPICommTypeObject(type(mango.mpi.Comm()))
    _mango_open_core_so._setMpi4pyPyMPICartcommTypeObject(type(mango.mpi.Cartcomm()))

# Class definition for (sphinx) documentation only.
class mtype(_mango_open_core_so.mtype):
    """
    Image type for mango :obj:`Dds` distributed array objects.
    """
    def __init__(self, name):
        _mango_open_core_so.mtype.__init__(self, name)
    
    def __str__(self):
        return self.name()
    
    def __eq__(self, other):
        return (
            issubclass(other.__class__, _mango_open_core_so.mtype)
            and
            (self.name() == other.name())
        )

# Class definition for (sphinx) documentation only.
class gtype(_mango_open_core_so.gtype):
    """
    Type for mango non-array data types.
    """
    def __init__(self, name):
        _mango_open_core_so.gtype.__init__(self, name)
    
    def __str__(self):
        return self.name()
    
    def __eq__(self, other):
        return (
            issubclass(other.__class__, _mango_open_core_so.gtype)
            and
            (self.name() == other.name())
        )

# To distinguish betweem 'mtype' parameter name and 'mtype' class.
_mtypeCls = _mango_open_core_so.mtype

def _ddsSetItem(self, *args):
    """
    Set value at specified voxel index. May be called as dds[zIdx,yIdx,xIdx]=v
    or as dds[(zIdx,yIdx,xIdx)]=v or as dds[scalarIdx]=v.
    """
    sfx = self.__class__.__name__
    if (len(args) == 4):
        return getattr(_mango_open_core_so, "_setitemzyxidx" + sfx)(self, *args)
    elif (len(args) == 2):
        idx = args[0]
        if (hasattr(idx, "__getitem__")):
            if (len(idx) == 3):
                return getattr(_mango_open_core_so, "_setitemseqidx" + sfx)(self, *args)
            else:
                raise ValueError("Expected 3D index, got len(idx)=%s" % len(idx))
        else:
            return getattr(_mango_open_core_so, "_setitemscalaridx" + sfx)(self, *args)
    raise RuntimeError("Expected 4 or 2 arguments, got %s arguments: %s" % (len(args), str(args)))

def _ddsGetItem(self, *args):
    """
    Return value at specified voxel index. May be called as dds[zIdx,yIdx,xIdx]
    or as dds[(zIdx,yIdx,xIdx)] or as dds[scalarIdx].
    """
    sfx = self.__class__.__name__
    if (len(args) == 3):
        return getattr(_mango_open_core_so, "_getitemzyxidx" + sfx)(self, *args)
    elif (len(args) == 1):
        idx = args[0]
        if (hasattr(idx, "__getitem__")):
            if (len(idx) == 3):
                return getattr(_mango_open_core_so, "_getitemseqidx" + sfx)(self, *args)
            else:
                raise ValueError("Expected 3D index, got len(idx)=%s" % len(idx))
        else:
            return getattr(_mango_open_core_so, "_getitemscalaridx" + sfx)(self, *args)
    raise RuntimeError("Expected 3 or 1 arguments, got %s arguments: %s" % (len(args), str(args)))

def _ddsCopy(self, *args, **kwargs):
    """
    Creates a (deep) copy of this :obj:`Dds` object (halo elements not copied or updated).

    :type self: :obj:`Dds`
    :param self: Copies data elements from dds into newly created :obj:`Dds` object.
    :type dtype: :samp:`scipy.dtype`
    :param dtype: Specifies the type of elements in the newly created :obj:`Dds` object,
        e.g. "uint16", "float32", etc.
    :type mtype: :obj:`mango.mtype`
    :param mtype: The mango image mtype of the Dds, one of:
        "Tomographic_Data", "Segmented_Data", etc.
    :type mpidims: sequence
    :param mpidims: The cartesian MPI process layout for the new created :obj:`Dds`.

    :rtype:  :obj:`Dds`
    :return: New :obj:`Dds` instance of same dimensions and layout as :samp:`self`
       with non-halo elements copied from :samp:`self`.
    """

    return mango.copy(*args, **kwargs)

_ddsMtypeList = [_mtypeCls(n) for n in _mango_open_core_so._getMtypeNameList()]
_ddsMtypeNameList = []
for mt in _ddsMtypeList:
    _ddsMtypeNameList.append(mt.name())
    _ddsMtypeNameList.append(mt.shortName())
    if (mt.fileNameBase() != mt.shortName()):
        _ddsMtypeNameList.append(mt.fileNameBase())

_ddsDtypeList = []
_ddsTypeList = []

def getDdsMtypeList():
    """
    Returns the list of possible :obj:`mango.mtype` objects for
    the :obj:`mango.Dds` :samp:`mtype` attribute.

    :rtype: :obj:`list` of :obj:`mango.mtype`
    :return: List of possible :obj:`mango.mtype` objects.
    """
    return _ddsMtypeList

def __addClassAttributes():
    """
    Adds some attributes to the _mango_open_core_so._Dds_(.*) classes, in particular
    __getitem__ method, __setitem__ method, and the dtype attribute.
    """
    thingStrList = dir(_mango_open_core_so)
    ddsClsRegEx = re.compile("_Dds_(.*)")
    for thingStr in thingStrList:
        mtch = ddsClsRegEx.match(thingStr)
        if (mtch != None):
            thing = getattr(_mango_open_core_so, thingStr)
            thing.dtype = sp.dtype(mtch.group(1))
            _ddsDtypeList.append(thing.dtype)
            thing.__getitem__ = _ddsGetItem
            thing.__setitem__ = _ddsSetItem
            thing.copy = _ddsCopy
            _ddsTypeList.append(thing)

__addClassAttributes()

def isDds(obj):
    """
    Returns whether specified :samp:`obj` is an instance of a :obj:`mango.Dds`.
    
    :type obj: :obj:`object`
    :param obj: Object to be tested.
    :rtype: :obj:`bool`
    :return: :samp:`True` is an instance of a :obj:`mango.Dds` distributed array.
    """
    for ddsType in _ddsTypeList:
        if (isinstance(obj, ddsType)):
            return True
    return False

def empty(shape, dtype=None, mtype=None, halo=(0,0,0), mpidims=(0,0,0), origin=(0,0,0), subdshape=None, subdorigin=None):
    if (isinstance(halo, int) or ((sys.version_info.major < 3) and isinstance(halo, long))):
        # Make halo a 3-sequence
        halo = [halo,]*3

    if (mtype != None):
        mtype = _mtypeCls(str(mtype))
        if ((dtype != None) and (mtype.dtype != sp.dtype(dtype))):
            raise \
                ValueError(
                    "dtype=%s argument does not agree with mtype.dtype=%s argument (mtype=%s)"
                    %
                    (dtype, mtype.dtype, mtype)
                )
        dtype = mtype.dtype
    if ((mtype is None) and (dtype is None)):
        dtype = "float64"

    if (len(shape) != 3):
        raise \
            ValueError(
                (
                    "Can only create 3D shaped Dds: %dD Dds shape=%s requested."
                )
                %
                (len(shape), shape)
            )
    if (len(origin) != 3):
        raise \
            ValueError(
                (
                    "Need 3D origin index, got origin=%s."
                )
                %
                (origin, )
            )
    if (len(mpidims) != 3):
        raise \
            ValueError(
                (
                    "Need 3D mpidims layout, got mpidims=%s."
                )
                %
                (mpidims, )
            )

    createFunc = getattr(_mango_open_core_so, "_create" + "_Dds_" + str(sp.dtype(dtype)))
    try:
        newDds = \
            createFunc(
                gshape=shape,
                gorigin=origin,
                haloshape=halo,
                mpidims=mpidims,
                lshape=subdshape,
                lorigin=subdorigin
            )
    except RuntimeError as e:
        logger.error(
            "Error creating Dds: %s(gshape=%s, gorigin=%s, haloshape=%s, mpidims=%s, lshape=%s, lorigin=%s)"
            %
            (createFunc, shape, origin, halo, mpidims, lshape, lorigin)
        )
        raise
    newDds.mtype = mtype
    
    return newDds

empty.__doc__=\
    """
    Creates an uninitialised :obj:`Dds` object of specified global size.
    
    :type shape: 3-sequence
    :param shape: Three element sequence indicating the global shape
        of the array e.g. :samp:`(zSz, ySz, xSz)`.
    :type dtype: :samp:`scipy.dtype`
    :param dtype: The data-type of the :obj:`Dds` elements, one of:
        %s.
    :type mtype: :obj:`mango.mtype`
    :param mtype: The mango image mtype of the :obj:`Dds`, one of:
        %s.
    :type halo: 3-sequence or int
    :param halo: The number of voxels added as a layer around each MPI-subdomain.
    :type mpidims: 3-sequence
    :param mpidims: The shape of the cartesian MPI process layout.
    :type origin: 3-sequence
    :param origin: The global indexing origin for the :obj:`Dds` array.
    :type subdshape: 3-sequence
    :param subdshape: The sub-domain shape for the :obj:`Dds` on this MPI process.
    :type subdorigin: 3-sequence
    :param subdorigin: The global indexing origin for the :obj:`Dds` sub-domain on this MPI process.
    
    :rtype: :obj:`Dds`
    :return: :obj:`Dds` with uninitialised (arbitrary) array elements. 

    """ % \
    (
         ", ".join(["'" + e + "'" for e in map(str,_ddsDtypeList)]),
         ", ".join(["'" + e + "'" for e in map(str,_ddsMtypeNameList)])
    )

def empty_like(dds, shape=None, dtype=None, mtype=None, halo=None, mpidims=None, origin=None, subdshape=None, subdorigin=None):
    """
    Create a new :obj:`Dds` object of same type/shape/MPI-layout as a specified :obj:`Dds` object.
    
    :type dds: :obj:`Dds`
    :param dds:  The type/shape/MPI-layout determine the same attributes of the returned :obj:`Dds`.
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
    :return: :obj:`Dds` of uninitialised (arbitrary) data with the same type,
       shape and MPI-layout as :samp:`dds`.
    """
    if ((shape is None) and (origin is None) and (subdorigin is None) and (mpidims is None)):
        subdorigin = dds.subd.origin
    if ((shape is None) and (origin is None) and (subdshape is None) and (mpidims is None)):
        subdshape = dds.subd.shape
    if (shape is None):
        shape = dds.shape
    if (mtype is None and dtype is None):
        if (hasattr(dds, "mtype")):
            mtype = dds.mtype
        dtype = dds.dtype
    if (halo is None):
        halo = dds.halo
    if (mpidims is None):
        mpidims = dds.mpi.shape
    if (origin is None):
        origin = dds.origin

    newDds = \
        empty(
            shape=shape,
            dtype=dtype,
            mtype=mtype,
            halo=halo,
            mpidims=mpidims,
            origin=origin,
            subdshape=subdshape,
            subdorigin=subdorigin
        )
    newDds.copyMetaData(dds)

    return newDds

def zeros(shape, dtype=None, mtype=None, halo=(0,0,0), mpidims=(0,0,0), origin=(0,0,0)):
    dds = empty(shape=shape, dtype=dtype, mtype=mtype, halo=halo, mpidims=mpidims, origin=origin)
    dds.setAllToValue(dds.dtype.type(0))
    return dds

zeros.__doc__=\
    """
    Creates an zero-initialised :obj:`Dds` object of specified global size.
    
    :type shape: 3-sequence
    :param shape: Three element sequence indicating the global shape
        of the array e.g. :samp:`(zSz, ySz, xSz)`.
    :type dtype: :samp:`scipy.dtype`
    :param dtype: The data-type of the :obj:`Dds` elements, one of:
        %s.
    :type mtype: :obj:`mango.mtype`
    :param mtype: The mango image mtype of the :obj:`Dds`, one of:
        %s.
    :type halo: 3-sequence or int
    :param halo: The number of voxels added as a layer around each MPI-subdomain.
    :type mpidims: 3-sequence
    :param mpidims: The shape of the cartesian MPI process layout.
    :type origin: 3-sequence
    :param origin: The global indexing origin for the :obj:`Dds` array.
    
    :rtype: :obj:`Dds`
    :return: :obj:`Dds` with array elements initialised to zero.
    """ % \
    (
         ", ".join(["'" + e + "'" for e in map(str,_ddsDtypeList)]),
         ", ".join(["'" + e + "'" for e in map(str,_ddsMtypeNameList)])
    )

def zeros_like(dds, shape=None, dtype=None, mtype=None, halo=None, mpidims=None, origin=None):
    """
    Create a new :obj:`Dds` object of same type/shape/MPI-layout as a specified :obj:`Dds` object.
    
    :type dds: :obj:`Dds`
    :param dds:  The type/shape/MPI-layout determine the same attributes of the returned :obj:`Dds`.
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

    :rtype: :obj:`Dds`
    :return: :obj:`Dds` of initialised (to zero) data with the same type,
        shape and MPI-layout as :samp:`dds`.
    """
    dds = empty_like(dds=dds, shape=shape, dtype=dtype, mtype=mtype, halo=halo, mpidims=mpidims, origin=origin)
    dds.setAllToValue(dds.dtype.type(0))
    return dds

def ones(shape, dtype=None, mtype=None, halo=(0,0,0), mpidims=(0,0,0), origin=(0,0,0)):
    dds = empty(shape=shape, dtype=dtype, mtype=mtype, halo=halo, mpidims=mpidims, origin=origin)
    dds.setAllToValue(dds.dtype.type(1))
    return dds

ones.__doc__=\
    """
    Creates an one-initialised :obj:`Dds` object of specified global size.
    
    :type shape: 3-sequence
    :param shape: Three element sequence indicating the global shape
        of the array e.g. :samp:`(zSz, ySz, xSz)`.
    :type dtype: :samp:`scipy.dtype`
    :param dtype: The data-type of the :obj:`Dds` elements, one of:
        %s.
    :type mtype: :obj:`mango.mtype`
    :param mtype: The mango image mtype of the :obj:`Dds`, one of:
        %s.
    :type halo: 3-sequence or int
    :param halo: The number of voxels added as a layer around each MPI-subdomain.
    :type mpidims: 3-sequence
    :param mpidims: The shape of the cartesian MPI process layout.
    :type origin: 3-sequence
    :param origin: The global indexing origin for the :obj:`Dds` array.
    
    :rtype: :obj:`Dds`
    :return: :obj:`Dds` with array elements initialised to one. 
    """ % \
    (
         ", ".join(["'" + e + "'" for e in map(str,_ddsDtypeList)]),
         ", ".join(["'" + e + "'" for e in map(str,_ddsMtypeNameList)])
    )

def ones_like(dds, shape=None, dtype=None, mtype=None, halo=None, mpidims=None, origin=None):
    """
    Create a new :obj:`Dds` object of same type/shape/MPI-layout as a specified :obj:`Dds` object.
    
    :type dds: :obj:`Dds`
    :param dds:  The type/shape/MPI-layout determine the same attributes of the returned :obj:`Dds`.
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

    :rtype: :obj:`Dds`
    :return: :obj:`Dds` of initialised (to one) data with the same type,
       shape and MPI-layout as :samp:`dds`.
    """
    dds = empty_like(dds=dds, shape=shape, dtype=dtype, mtype=mtype, halo=halo, mpidims=mpidims, origin=origin)
    dds.setAllToValue(dds.dtype.type(1))
    return dds

def copy(dds, shape=None, dtype=None, mtype=None, halo=None, mpidims=None, origin=None):
    """
    Copy-create a :obj:`Dds` object from an existing :obj:`Dds` object.

    :type dds: :obj:`Dds`
    :param dds: Copies data elements from dds into newly created :obj:`Dds` object.
    :type dtype: :samp:`scipy.dtype`
    :param dtype: Specifies the type of elements in the newly created :obj:`Dds` object,
        e.g. "uint16", "float32", etc.
    :type mtype: :obj:`mango.mtype`
    :param mtype: The mango image mtype of the Dds, one of:
        "Tomographic_Data", "Segmented_Data", etc.
    :type mpidims: sequence
    :param mpidims: The cartesian MPI process layout for the new created :obj:`Dds`.

    :rtype:  :obj:`Dds`
    :return: New :obj:`Dds` instance of same dimensions and layout as :samp:`dds`
       with non-halo elements copied from :samp:`dds`.
    """

    newDds = empty_like(dds, shape=shape, dtype=dtype, mtype=mtype, halo=halo, mpidims=mpidims, origin=origin)
    if (hasattr(newDds, "mtype") and (newDds.mtype != None)):
        newDds.setAllToValue(newDds.mtype.maskValue())
    newDds.fill(dds)

    return newDds

def minmax(input):
    """
    Returns :obj:`Dds` element-wise :samp:`(min, max)` pair.
    
    :type input: :obj:`Dds`
    :param input: Input elements for which minimum and maximum are calculated.
    
    :rtype: scalar
    :return: :samp:`(min,max)` pair.
    
    """
    if (hasattr(input, "mtype") and (input.mtype != None)):
        (mn, mx) = input.minmax(input.mtype.maskValue())
    else:
        (mn, mx) = input.minmax()
    
    return mn, mx

def sum(input, dtype=None):
    """
    Returns sum of all non-masked :obj:`Dds` elements.
    
    :type input: :obj:`Dds`
    :param input: Input elements for summation is calculated.
    :type dtype: :obj:`numpy.dtype` or dtype :obj:`str`
    :param dtype: Type used for summation of elements.
    
    :rtype: scalar
    :return: Sum of all elements.
    """
    mskVal = None
    if (hasattr(input, "mtype") and (input.mtype != None)):
        mskVal = input.mtype.maskValue()
    
    mpiComm = None
    if (hasattr(input, "mpi") and hasattr(input.mpi, "comm") and (input.mpi.comm != None)):
        mpiComm = input.mpi.comm

    inArr = input.subd.asarray()
    if (mskVal != None):
        s = sp.sum(sp.where(inArr != mskVal, inArr, 0), dtype=dtype)
    else:
        s = sp.sum(inArr, dtype=dtype)
    
    if (mpiComm != None):
        s = mpiComm.allreduce(s, mango.mpi.SUM)

    return s

def sum2(input, dtype=None):
    """
    Returns sum of all non-masked :obj:`Dds` elements-squared.
    
    :type input: :obj:`Dds`
    :param input: Input elements for which sum of squared-elements is calculated.
    :type dtype: :obj:`numpy.dtype` or dtype :obj:`str`
    :param dtype: Type used for summation of elements.
    
    :rtype: scalar
    :return: Sum of the squared-elements (i.e. :samp:`scipy.sum((input.asarray())**2, dtype)`).
    """
    mskVal = None
    if (hasattr(input, "mtype") and (input.mtype != None)):
        mskVal = input.mtype.maskValue()
    
    mpiComm = None
    if (hasattr(input, "mpi") and hasattr(input.mpi, "comm") and (input.mpi.comm != None)):
        mpiComm = input.mpi.comm

    inArr = input.subd.asarray()
    if (mskVal != None):
        s = sp.sum(sp.where(inArr != mskVal, inArr**2, 0), dtype=dtype)
    else:
        s = sp.sum(inArr**2, dtype=dtype)
    
    if (mpiComm != None):
        s = mpiComm.allreduce(s, mango.mpi.SUM)

    return s



def unique(input):
    """
    Returns sorted list of unique values in the specified :samp:`input` :obj:`Dds` array.
    
    :type input: :obj:`Dds`
    :param input: Input elements for uniqueness set calculation.
    
    :rtype: sequence of elements
    :return: Sorted sequence of unique elements of entire :samp:`input` array.
    """
    mskVal = None
    if (hasattr(input, "mtype") and (input.mtype != None)):
        mskVal = input.mtype.maskValue()
    
    mpiComm = None
    if (hasattr(input, "mpi") and hasattr(input.mpi, "comm") and (input.mpi.comm != None)):
        mpiComm = input.mpi.comm

    inArr = input.subd.asarray()
    u = np.unique(inArr)

    if (mskVal != None):
        u = np.delete(u, sp.where(u == mskVal))
    
    if (mpiComm != None):
        u = input.mpi.comm.allreduce(u.tolist(), op=mango.mpi.SUM)
        u = np.unique(u)

    return u

def have_same_subd_decomp(dds0, dds1):
    """
    Returns :samp:`True` if pairs of non-halo-sub-domains on all processes have the
    same (global) non-halo-sub-domain origin index and
    same non-halo-sub-domain shape. Note: performs an MPI *allreduce* operation.
    
    :type dds0: :obj:`mango.Dds`
    :param dds0: Array.
    :type dds1: :obj:`mango.Dds`
    :param dds1: Array.
    :rtype: :obj:`bool`
    :return: :samp:`True` if MPI non-halo-subdomain decomposition is the
       same for :samp:`dds0` and :samp:`dds1`.  

    """
    numDiff = 0
    if (sp.any(sp.logical_or(dds0.subd.origin != dds1.subd.origin, (dds0.subd.shape != dds1.subd.shape)))):
        numDiff = 1
    
    mpiComm = None
    if (hasattr(dds0, "mpi") and hasattr(dds0.mpi, "comm") and (dds0.mpi.comm != None)):
        mpiComm = dds0.mpi.comm
        numDiff = mpiComm.allreduce(numDiff, op=mango.mpi.SUM)

    return (numDiff == 0)

def copy_masked(src, dst, srcval=None, srcmask=None, dstmask=None):
    """
    Copies masked elements from the :samp:`src` array into the :samp:`dst` array.
    If the arrays are different layouts/shapes then the :samp:`src` array will
    be copied to an array of the same layout/shape as the :samp:`dst` array.
    By default, elements of the copied :samp:`src` array which are not in the
    original input :samp:`src` global domain are set to the mask value
    (i.e. elements outside the input :samp:`src` array are assumed to be masked).
    
    Only non-halo elements are copied.

    :type src: :obj:`Dds`
    :param src: Array from which masked voxels are copied.
    :type dst: :obj:`Dds`
    :param dst: Array to which masked voxels are copied.
    :type srcval: numeric
    :param srcval: Value for elements outside the :samp:`src` global domain.
       If :samp:`None` set to :samp:`src.mtype.maskValue()`.
    """
    srcMskVal = srcmask
    dstMskVal = dstmask

    srcMtype = None
    srcDtype = src.dtype

    if (hasattr(src, "mtype") and (src.mtype != None)):
        srcMtype = src.mtype
        srcDtype = src.dtype
        if (srcMskVal is None):
            srcMskVal = src.mtype.maskValue()
    
    if (srcMskVal is None):
        raise Exception("Source Dds object does not have a non-None mtype attribute required to deterime mask value.")

    if ((dstMskVal is None) and hasattr(dst, "mtype") and (dst.mtype != None)):
        dstMskVal = dst.mtype.maskValue()

    if (dstMskVal is None):
        raise Exception("Destination Dds object does not have a non-None mtype attribute required to deterime mask value.")

    if (srcval is None):
        srcval = srcMskVal

    if (not have_same_subd_decomp(src, dst)):
        newSrc = mango.empty_like(dst, mtype=srcMtype, dtype=srcDtype)
        newSrc.setAllToValue(newSrc.dtype.type(srcval))
        newSrc.fill(src)
        src = newSrc
    
    _mango_open_core_so._copy_masked_voxels(src, dst, srcMskVal, dstMskVal)

def copy_non_masked(src, dst, srcval=None, srcmask=None):
    """
    Copies non-masked elements from the :samp:`src` array into the :samp:`dst` array.
    If the arrays are different layouts/shapes then the :samp:`src` array will
    be copied to an array of the same layout/shape as the :samp:`dst` array.
    By default, elements of the copied :samp:`src` array which are not in the
    original input :samp:`src` global domain are set to the mask value
    (i.e. elements outside the input :samp:`src` array are assumed to be masked).
    
    Only non-halo elements are copied.

    :type src: :obj:`Dds`
    :param src: Array from which non-masked voxels are copied.
    :type dst: :obj:`Dds`
    :param dst: Array to which non-masked voxels are copied.
    :type srcval: numeric
    :param srcval: Value for elements outside the :samp:`src` global domain.
       If :samp:`None` set to :samp:`src.mtype.maskValue()`.
    """

    srcMtype = None
    srcDtype = src.dtype

    if (hasattr(src, "mtype") and (src.mtype != None)):
        srcMtype = src.mtype
        srcDtype = src.dtype
        if (srcMskVal is None):
            srcMskVal = src.mtype.maskValue()
    
    if (srcMskVal is None):
        raise Exception("Source Dds object does not have a non-None mtype attribute required to deterime mask value.")

    if (srcval is None):
        srcval = srcMskVal

    if (not have_same_subd_decomp(src, dst)):
        newSrc = mango.empty_like(dst, mtype=src.mtype)
        newSrc.setAllToValue(newSrc.dtype.type(srcval))
        newSrc.fill(src)
        src = newSrc

    _mango_open_core_so._copy_non_masked_voxels(src, dst, srcMskVal)

def map_element_values(input, ufunc, mtype=None, dtype=None, use1to1cache=True):
    """
    Maps each element value if :samp:`input` array to new value calculated by :samp:`ufunc`.
    
    :type input: :obj:`mango.Dds`
    :param input: Elements of this array are mapped to new values via the :samp:`ufunc` argument.
    :type ufunc: callable :obj:`object`
    :param ufunc: Unary function which accepts a :samp:`input.dtype` argument and returns
       a :samp:`dtype` (or :samp:`mtype.dtype`) value.
    :type mtype: :obj:`mango.mtype` or :obj:`str`
    :param mtype: The :obj:`mango.mtype` of the returned :obj:`mango.Dds` object.
    :type dtype: :obj:`numpy.dtype` or :obj:`str`
    :param dtype: The :obj:`numpy.dtype` of the returned :obj:`mango.Dds` array.
    :type use1to1cache: :obj:`bool`
    :param use1to1cache: If :samp:`True`, an internal lookup table is kept which
       saves multiple calls of the :samp:`ufunc` function for the same argument.
       Can excessively use memory (for float data types, say) and is not correct if
       the :samp:`ufunc` function is not a 1-to-1 mapping. 
    :rtype: :obj:`mango.Dds`
    :return: Array with elements mapped as per :samp:`ufunc` mapping.
    
    Example::
    
       import mango
       import mango.data
       import math
    
       def cubed_root_func(x):
          return math.pow(x,1.0/3.0)
       
       srcDds  = mango.data.gaussian_noise(shape=(64,128,128), mean=2000, stdd=10, dtype="int32")
       sqrdDds = mango.map_element_values(srcDds, lambda x: x*x)
       sqrtDds = mango.map_element_values(srcDds, lambda x: math.sqrt(x), dtype="float32")
       cubdDds = mango.map_element_values(srcDds, cubed_root_func, mtype="tomo_float")
   
    
    """
    if (mtype != None):
        # Convert to object in case mtype argument is a string
        mtype = mango.mtype(mtype)
    if (dtype != None):
        # Convert to object in case dtype argument is a string
        dtype = sp.dtype(dtype)
    
    return \
        _mango_open_core_so._map_element_values(
            input=input,
            ufunc=ufunc,
            mtype=mtype,
            dtype=dtype,
            use1to1cache=use1to1cache
        )

def _arrayitemfreq(a):
    """
    Version 0.13 implementation of :func:`scipy.stats.itemfreq`
    """
    items, inv = np.unique(a, return_inverse=True)
    freq = np.bincount(inv)
    return np.array([items, freq]).T

def itemfreq(input):
    """
    Like :func:`scipy.stats.itemfreq`, returns a frequency count for
    the *unique* element values of the :samp:`input` array.
    
    :type input: :obj:`mango.Dds`
    :param input: Frequency counts calculated for the elements of this array.
    :rtype: :obj:`tuple`
    :returns: The :samp:`(elems, counts)` pair.
    """
    itemFreq = _arrayitemfreq(input.subd.asarray())
    # get rid of the maskValue bin
    if (
        hasattr(input, "mtype")
        and
        (input.mtype != None)
    ):
        itemFreq = itemFreq[sp.where(itemFreq[:,0] != input.mtype.maskValue())]
    
    dtype = itemFreq.dtype
    # merge counts from the different MPI processes.
    if (input.mpi.comm != None):
        allItemFreq = input.mpi.comm.allgather(itemFreq)
        elems  = []
        counts = []
        for itemFreq in allItemFreq:
            elems  += itemFreq[:,0].tolist()
            counts += itemFreq[:,1].tolist()
        elems  = sp.array(elems, dtype=dtype)
        counts = sp.array(counts, dtype=dtype)
        uElems  = sp.unique(elems)
        uCounts = sp.zeros_like(uElems)
        for i in range(0, uElems.size):
            uCounts[i] = sp.sum(counts[sp.where(elems == uElems[i])])
    else:
        uElems  = itemFreq[:,0]
        uCounts = itemFreq[:,1]
    return uElems, uCounts

# This class is for documentation purposes, never instantiate one of these.
class Dds(_mango_open_core_so._Dds_uint16):
    """
    Mango distributed array (*DistributedDataStruct*) python API.
    """

# This class is for (sphinx) documentation purposes, never instantiate one of these.
class DdsMpiInfo(_mango_open_core_so._MpiInfo_Dds_uint16):
    """
    MPI data associated with sub-domain of a :obj:`Dds` object.
    """

# This class is for (sphinx) documentation purposes, never instantiate one of these.
class DdsNonHaloSubDomain(_mango_open_core_so._NonHaloSubDomain_Dds_uint16):
    """
    Non-halo-sub-region data associated with MPI domain decomposition of a :obj:`Dds` object.
    """

# This class is for (sphinx) documentation purposes, never instantiate one of these.
class DdsHaloSubDomain(_mango_open_core_so._HaloSubDomain_Dds_uint16):
    """
    Halo-sub-region data associated with MPI domain decomposition of a :obj:`Dds` object.
    """

# This class is for (sphinx) documentation purposes, never instantiate one of these.
class DdsMetaData(_mango_open_core_so._MetaData_Dds_uint16):
    """
    Meta-data associated with a :obj:`Dds` object.
    """
