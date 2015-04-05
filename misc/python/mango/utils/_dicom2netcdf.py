
import scipy as sp
import numpy as np
import scipy.io
import scipy.misc
import scipy.ndimage

import os
import os.path
import copy
import re

from mango import mpi

logger, rootLogger = mpi.getLoggers(__name__)

havePyDicom = False
try:
    import dicom
    havePyDicom = True
except ImportError as e:
    dicom = e
    # print "Error importing pydicom package: %s" % str(e)
    havePyDicom = False

class VolumeMetaData:
    def __init__(self):
        self.sampleId       = None
        self.voxelSize      = None
        self.voxelUnit      = None
        self.volumeSize     = None
        self.volumeOrigin   = None
        self.numFiles       = None
        self.attribDict     = None
    
    def log(self, logfunc=rootLogger.debug):
        logfunc("sampleId     = %s" % self.sampleId)
        logfunc("voxelSize    = %s" % self.voxelSize)
        logfunc("voxelUnit    = %s" % self.voxelUnit)
        logfunc("volumeSize   = %s" % self.volumeSize)
        logfunc("volumeOrigin = %s" % self.volumeOrigin)
        logfunc("numFiles     = %s" % self.numFiles)
        logfunc("attribDict   = %s" % self.attribDict)

class Dicom2NetcdfJobMetaData(VolumeMetaData):
    def __init__(self):
        VolumeMetaData.__init__(self)
        self.subVolumeSize     = None
        self.subVolumeOrigin   = None

class Dicom2NetcdfConvertJob:
    """
    Convert a single DICOM data file to a corresponding netCDF file.
    """
    def __init__(self, inDicomFileName, outNetcdfFileName, metaData):
        self.inDicomFileName = inDicomFileName
        self.outNetcdfFileName = outNetcdfFileName
        self.metaData = metaData

    def __call__(self):
        import mango.io

        logger.debug("Converting %s to %s..." % (self.inDicomFileName, self.outNetcdfFileName))
        try:
            dcm = dicom.read_file(self.inDicomFileName)
            if (not hasattr(dcm, "SamplesPerPixel")):
                dcm.SamplesPerPixel = 1
            img = dcm.pixel_array
            logger.debug("Read %s shaped pixel_array from DICOM file." % (img.shape,))
        except Exception as e:
            logger.error("Error encountered reading DICOM file, dicom state:\n %s" % str(dcm))
            raise

        (isSigned,dtype) = mango.io.convertToNetcdfDtype[str(img.dtype).lower()]
        if (len(img.shape) <= 3):
            varName = mango.io.dtypeToNetcdfVariable[str(img.dtype).lower()]
        else:
            varName = "rgba8"
        dims = [varName + "_zdim", varName + "_ydim", varName + "_xdim"]
        ncdf = scipy.io.netcdf.netcdf_file(self.outNetcdfFileName, mode='w')
        if (len(img.shape) < len(dims)):
            outImg = scipy.zeros((1,)*(len(dims)-len(img.shape)) + img.shape, dtype=img.dtype)
            slicesTuple = ((0,)*(len(dims)-len(img.shape)) + (slice(None),)*len(img.shape))
            outImg[slicesTuple] = img
            img = outImg
        if (len(img.shape) <= 3):
            for i in range(len(dims)):
                ncdf.createDimension(dims[i], img.shape[i])
        else:
            raise RuntimeError("Converting to multi-channel 3D data not supported.")

        logger.debug("NetCDF img shape=%s." % (img.shape,))
        ncdf.createVariable(varName, dtype, dims)
        ncdf.variables[varName].value_signed=str(isSigned).lower()
        ncdf.variables[varName].data.flat[:] = scipy.array(img, copy=True, dtype=dtype).flat

        ncdf.zdim_total            = self.metaData.volumeSize[2]
        ncdf.number_of_files       = self.metaData.numFiles
        ncdf.zdim_range            = \
            sp.array(
                [
                    self.metaData.subVolumeOrigin[2],
                    self.metaData.subVolumeOrigin[2] + self.metaData.subVolumeSize[2]-1
                ],
                dtype="int32"
            )
        ncdf.data_description      = "DICOM converted to netCDF" ;
        ncdf.voxel_size_xyz        = self.metaData.voxelSize
        ncdf.voxel_unit            = self.metaData.voxelUnit
        ncdf.coord_transform       = "",
        ncdf.total_grid_size_xyz   = self.metaData.volumeSize
        ncdf.coordinate_origin_xyz = self.metaData.volumeOrigin

        ncdf.flush()


    def log(self, logfunc=rootLogger.debug):
        logfunc("inDicomFileName   = %s" % self.inDicomFileName)
        logfunc("outNetcdfFileName = %s" % self.outNetcdfFileName)
        logfunc("meta-data:")
        self.metaData.log(logfunc)

def readXtekMetaData(inputDir, metaData = VolumeMetaData()):
    if (inputDir != None):
        if (os.path.exists(inputDir)):
            dirListing = os.listdir(inputDir)
            dirListing.sort()
            fileRegex = re.compile('.*xtekct')
            d = dict()
            for fName in dirListing:
                fileMatch = fileRegex.match(fName)
                if (fileMatch != None):
                    if (hasattr(metaData, "infoFileName")):
                        metaData.infoFileName = copy.deepcopy(fName)
                    fName = os.path.join(inputDir, fName)
                    rootLogger.info("Reading meta-data from file %s..." % fName)
                    fLines = file(fName, 'r').readlines()
                    rex = re.compile('(.*)=(.*)')
                    for line in fLines:
                        m = rex.match(line)
                        if (m != None):
                            d[m.group(1).strip()] = m.group(2).strip()
                    voxMmSz  = None
                    voxSzUnit= None
                    voxCount = None
                    sampleId = None
                    if (("VoxelsX" in d.keys()) and ("VoxelsY" in d.keys()) and ("VoxelsZ" in d.keys())):
                        voxCount = [int(d["VoxelsX"]), int(d["VoxelsY"]), int(d["VoxelsZ"])]
                    if (("VoxelSizeX" in d.keys()) and ("VoxelSizeY" in d.keys()) and ("VoxelSizeZ" in d.keys())):
                        voxMmSz = [float(d["VoxelSizeX"]), float(d["VoxelSizeY"]), float(d["VoxelSizeZ"])]
                        voxSzUnit = "mm"
                    if ("Name" in d.keys()):
                        sampleId = d["Name"]
                    metaData.sampleId       = sampleId
                    metaData.voxelSize      = sp.array(voxMmSz, dtype="float64")
                    metaData.voxelUnit      = voxSzUnit
                    metaData.volumeSize     = sp.array(voxCount, dtype="int32")
                    metaData.volumeOrigin   = sp.array([0,0,0], dtype="int32")
                    metaData.attribDict     = d

                    break
        else:
            raise RuntimeError("Path %s does not exist." % inputDir)
    return metaData

class Dicom2Netcdf:
    """
    Converts DICOM file format to mango NetCDF format.
    """
    
    def __init__(self):
        self.netcdfExt      = "nc"
        self.dicomExt       = "dcm"
        self.dicomDir       = None
        self.netcdfDir      = None
        self.xtekInfoDir    = None

    def readMetaData(self):
        return readXtekMetaData(self.xtekInfoDir)

    def getDicomFileNameList(self):
        rex = re.compile(".*%s" % self.dicomExt)
        dirListing = os.listdir(self.dicomDir)
        dirListing.sort()
        dicomFileNameList = []
        for fileName in dirListing:
            m = rex.match(fileName)
            if (m != None):
                dicomFileNameList.append(os.path.join(self.dicomDir, fileName))
        return dicomFileNameList
    
    def createJobList(self):
        metaData = self.readMetaData()
        dicomFileNameList = self.getDicomFileNameList()
        metaData.numFiles = len(dicomFileNameList)
        jobList = []
        rex = re.compile("[^0-9]*([0-9]+)([^0-9]*)%s" % self.dicomExt)
        for fileName in dicomFileNameList:
            m = rex.match(os.path.split(fileName)[1])
            if (m != None):
                idxStr = m.group(1).strip()
                idxStrStripped = idxStr.lstrip('0')
                if (len(idxStrStripped) > 0):
                    idx = int(idxStrStripped)
                else:
                    idx = 0
                jobMetaData                 = Dicom2NetcdfJobMetaData()
                jobMetaData.sampleId        = copy.deepcopy(metaData.sampleId)
                jobMetaData.voxelSize       = copy.deepcopy(metaData.voxelSize)
                jobMetaData.voxelUnit       = copy.deepcopy(metaData.voxelUnit)
                jobMetaData.volumeSize      = copy.deepcopy(metaData.volumeSize)
                jobMetaData.volumeOrigin    = copy.deepcopy(metaData.volumeOrigin)
                jobMetaData.numFiles        = copy.deepcopy(metaData.numFiles)
                jobMetaData.attribDict      = None
                if (
                    (jobMetaData.volumeOrigin != None)
                    and
                    (jobMetaData.volumeSize != None)
                ):
                    jobMetaData.subVolumeSize   = copy.deepcopy(metaData.volumeSize)
                    jobMetaData.subVolumeOrigin = copy.deepcopy(metaData.volumeOrigin)
                    jobMetaData.subVolumeOrigin[2] = idx
                    jobMetaData.subVolumeSize[2] = 1

                job = \
                    Dicom2NetcdfConvertJob(
                        inDicomFileName = fileName,
                        outNetcdfFileName = os.path.join(self.netcdfDir, ("block%s" % idxStr) + "." + self.netcdfExt),
                        metaData = jobMetaData
                    )
                jobList.append(job)
            else:
                raise RuntimeError("Could not parse index string from file name %s" % fileName)
        return jobList

    def prepareNetcdfDir(self):
        if (os.path.exists(self.netcdfDir)):
            if (os.path.isdir(self.netcdfDir)):
                for f in os.listdir(self.netcdfDir):
                    ff = os.path.join(self.netcdfDir, f)
                    ffExt = os.path.splitext(ff)[1]
                    if (os.path.isfile(ff) and (ffExt.find(self.netcdfExt) < len(ffExt))):
                        rootLogger.debug("Removing file %s" % ff)
                        os.remove(ff)
            else:
                os.remove(self.netcdfDir)
                os.makedirs(self.netcdfDir)
        else:
            os.makedirs(self.netcdfDir)

    def runJob(self, job):
        try:
            job()
        except Exception as e:
            logger.error("Exception encountered converting %s to %s:" % (job.inDicomFileName, job.outNetcdfFileName))
            logger.error(str(e))

    def executeJobListUsingMpi(self, jobList):
        myRank = mpi.world.Get_rank()
        if (myRank == 0):
            self.prepareNetcdfDir()
        mpi.world.barrier()
        startIdx = myRank
        for jobIdx in range(startIdx, len(jobList), mpi.world.Get_size()):
            job = jobList[jobIdx]
            self.runJob(job)


    def executeJobList(self, jobList):
        if (mpi.haveMpi4py):
            self.executeJobListUsingMpi(jobList)
        else:
            self.prepareNetcdfDir()
            for job in jobList:
                self.runJob(job)

    def convert(self, dicomDir, netcdfDir, xtekInfoDir=None):
        self.dicomDir       = dicomDir
        self.netcdfDir      = netcdfDir
        self.xtekInfoDir    = xtekInfoDir
        
        jobList = self.createJobList()
        self.executeJobList(jobList)
