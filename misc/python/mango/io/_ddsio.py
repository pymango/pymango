from mango import mpi
from mango.core import mtype
import mango.core

import sys
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_open_io as _mango_open_io_so
    sys.setdlopenflags(_flags)
else:
    from . import _mango_open_io as _mango_open_io_so

import os
import os.path
import re

logger, rootLogger = mpi.getLoggers(__name__)

def splitext(ddsDataPath):
    """
    Splits the specified path into a (*base*, *extension*) tuple.
    
    :type ddsDataPath: :obj:`str`
    :param ddsDataPath: '.nc' file or '_nc' directory.
    
    :rtype: :obj:`tuple`
    :return: A :samp:`(baseStr, extStr)` pair.
    """
    base,ext = os.path.splitext(ddsDataPath)
    if (ext == ""):
        extIdx = ddsDataPath.rfind("_")
        base, ext = (ddsDataPath[0:extIdx], ddsDataPath[extIdx:])
    
    return base, ext

def splitpath(ddsDataPath):
    """
    Splits the specified path into a (*dir*, *prefix*, *suffix*, *extension*) tuple.
    
    :type ddsDataPath: :obj:`str`
    :param ddsDataPath: '.nc' file or '_nc' path.
    
    :rtype: :obj:`tuple`
    :return: A :samp:`(dirStr, prefixStr, suffixStr, extStr)` tuple.
    """
    dir,leaf = os.path.split(ddsDataPath)
    mtypePrefixList = [mtyp.fileNameBase() for mtyp in mango.core.getDdsMtypeList()]
    mtypePrefixList.sort()
    mtypePrefixList.reverse()
    mtypePrefixesStr = "|".join(mtypePrefixList)
    regEx = re.compile("(%s)(.*)([_\\.]nc)" % (mtypePrefixesStr,))
    m = regEx.match(leaf)
    if (m != None):
        prefix = m.group(1)
        suffix = m.group(2)
        ext = m.group(3)
    else:
        raise Exception("Could not match path %s to regular expression %s." % (ddsDataPath, regEx))
    
    return [dir, prefix, suffix, ext]

def readDds(fileName, halo=(0,0,0), mpidims=(0,0,0)):
    """
    Reads mango netCDF data from specified file/directory.
    
    :type fileName: str
    :param fileName: The netCDF file/directory from which 3D array data is read.
    :type halo: 3-sequence
    :param halo: Extra number of halo/overlap voxels (nZ,nY,nX) on the read array.
    :type mpidims: 3-sequence
    :param mpidims: The MPI cartesian processor layout (nZ,nY,nX) for the returned
        :obj:`mango.Dds` object. 
    :rtype: :obj:`mango.Dds`
    :return: :obj:`mango.Dds` object initialised with data from file.
    """
    rootLogger.info("Reading Dds from file '%s'..." % fileName)
    dds = _mango_open_io_so._readDds(fileName, halo, mpidims)
    dds.mtype = mtype(str(dds.mtype))
    rootLogger.info("Done reading Dds from file '%s'." % fileName)
    return dds

def writeDds(fileName, dds, melemperfile=256, writehistogram=True):
    """
    Writes Dds array data to a specified file/directory.
    
    :type fileName: str
    :param fileName: The netCDF file/directory to which 3D array data is written.
    :type dds: :obj:`mango.Dds`
    :param dds: The data which is to be written to file.
    :type melemperfile: :obj:`int`
    :param melemperfile: Number (approximate) of array elements (mega-elements) per file.
        Large arrays are split into multiple files. This parameter specifies the number of
        mega-elements per file. A value of :samp:`melemperfile=256` implies
        approximately 256-mega-elements (:samp:`256*1024*1024` elements)
        per block-file.
    :type writehistogram: :obj:`bool`
    :param writehistogram: If :samp:`True`, write image histogram data to file as netCDF variable.
    :rtype: :obj:`str`
    :return: Path where the data was written, this is parent directory of
       of the block files, or just the single netCDF file containing the entire
       array.
    
    Example::

       import mango
       import mango.data
       import mango.io
       import os
       import os.path
       
       melemperfile=4
       segDds = mango.data.gaussian_noise(shape=(150,400,400), mtype="segmented", mean=128, stdd=16)
       segOutFile = mango.io.writeDds("./segmentedGaussianNoise.nc", segDds, melemperfile=melemperfile)
       tomDds = mango.data.gaussian_noise(shape=(150,400,400), mtype="tomo", mean=32000, stdd=4000)
       tomOutFile = mango.io.writeDds("./tomoGaussianNoise.nc", tomDds, melemperfile=melemperfile)
       lblDds = mango.data.gaussian_noise(shape=(150,400,400), mtype="labels", mean=128000, stdd=8000)
       lblOutFile = mango.io.writeDds("./labelsGaussianNoise.nc", lblDds, melemperfile=melemperfile)
       
       def getFileAndSizeListString(dirName):
           strList = []
           fileNameList = os.listdir(dirName)
           fileNameList.sort()
           for fileName in fileNameList:
              fileName = os.path.join(dirName, fileName)
              strList.append("%10d %50s" % (os.path.getsize(fileName), fileName))
           return "\\n".join(strList)
       
       
       print("segOutFile=%s:\\n%s" % (segOutFile, getFileAndSizeListString(segOutFile)))
       print("\\ntomOutFile=%s:\\n%s" % (tomOutFile, getFileAndSizeListString(tomOutFile)))
       print("\\nlblOutFile=%s:\\n%s" % (lblOutFile, getFileAndSizeListString(lblOutFile)))
    
    
    And the output is::
    
       segOutFile=./segmentedGaussianNoise_nc:
          4480736       ./segmentedGaussianNoise_nc/block00000000.nc
          4480588       ./segmentedGaussianNoise_nc/block00000001.nc
          4480588       ./segmentedGaussianNoise_nc/block00000002.nc
          4480588       ./segmentedGaussianNoise_nc/block00000003.nc
          4480588       ./segmentedGaussianNoise_nc/block00000004.nc
          1600588       ./segmentedGaussianNoise_nc/block00000005.nc
       
       tomOutFile=./tomoGaussianNoise_nc:
          8960704            ./tomoGaussianNoise_nc/block00000000.nc
          8960568            ./tomoGaussianNoise_nc/block00000001.nc
          8960568            ./tomoGaussianNoise_nc/block00000002.nc
          8960568            ./tomoGaussianNoise_nc/block00000003.nc
          8960568            ./tomoGaussianNoise_nc/block00000004.nc
          3200568            ./tomoGaussianNoise_nc/block00000005.nc
       
       lblOutFile=./labelsGaussianNoise_nc:
         17920708          ./labelsGaussianNoise_nc/block00000000.nc
         17920572          ./labelsGaussianNoise_nc/block00000001.nc
         17920572          ./labelsGaussianNoise_nc/block00000002.nc
         17920572          ./labelsGaussianNoise_nc/block00000003.nc
         17920572          ./labelsGaussianNoise_nc/block00000004.nc
          6400572          ./labelsGaussianNoise_nc/block00000005.nc
    
    
    """
    rootLogger.info("Writing Dds to file '%s'..." % fileName)
    destPathStr = _mango_open_io_so._writeDds(fileName, dds, melemperfile, writehistogram)
    rootLogger.info("Done writing Dds to file '%s'." % destPathStr)
    
    return destPathStr

__all__ = [s for s in dir() if not s.startswith('_')]
