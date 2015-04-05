#!/usr/bin/env python

import mango
import mango.io
import mango.image
import mango.application.mdss as mdss
import mango.mpi as mpi

import scipy as sp
import logging
import re
import os
import sys

haveArgParse = False
try:
    import argparse
    haveArgParse = True
except:
    import optparse


logger, rootLogger = mpi.getLoggers(__name__)

def ddsFileMetaDataIsConsistent(ddsPath, mpiComm=mpi.world):
    rootRank = 0
    if (mpiComm != None):
        rank = mpiComm.Get_rank()
        size = mpiComm.Get_size()
    else:
        rank = rootRank
        size = 1

    fileList = None
    totalNumZ = None
    if (rank == rootRank):
        if (os.path.isdir(ddsPath)):
            fileList = [os.path.join(ddsPath, fileName) for fileName in os.listdir(ddsPath)]
        else:
            fileList = [ddsPath,]
        fileList.sort()
        ncdfFile = sp.io.netcdf_file(fileList[0], 'r')
        totalNumZ = ncdfFile.zdim_total
    
    if (mpiComm != None):
        fileList,totalNumZ = mpiComm.bcast((fileList, totalNumZ), rootRank)
    
    numZ = 0
    for fIdx in range(rank, len(fileList), size):
        ncdfFile = sp.io.netcdf_file(fileList[fIdx], 'r')
        ncdfZRange = ncdfFile.zdim_range
        numZ += ncdfZRange[1] - ncdfZRange[0] + 1
    
    if (mpiComm != None):
        numZ = mpiComm.allreduce(numZ, op=mpi.SUM)
    
    message = None
    if (numZ != totalNumZ):
        message = ("%s NetCDF inconsistency, expected %s zdim_total, got %s from files" % (ddsPath, totalNumZ, numZ))
    
    return message
 
class DownsampleJob(object):
    def __init__(
        self,
        mpiComm = None,
        mdssSrcImagePath = None,
        mdssDstDirPath = None,
        mdssPrj = None,
        loclSrcImagePath = None,
        loclDstDirPath = None,
        voxelSizeTuples = None
    ):
        self.srcDds            = None
        self.mpiComm           = mpiComm
        if (self.mpiComm == None) and (mpi.haveMpi4py):
            self.mpiComm = mpi.world

        self.rootRank          = 0
        self.mdssSrcImagePath  = mdssSrcImagePath
        self.mdssDstDirPath    = mdssDstDirPath
        self.mdssDstImagePaths = []
        self.mdssPrj           = mdssPrj
        self.loclSrcImagePath  = loclSrcImagePath
        self.loclSrcDirPath    = os.path.split(self.loclSrcImagePath)[0]
        self.loclDstDirPath    = loclDstDirPath
        self.loclDstImagePaths = []
        self.voxelSizeTuples   =  voxelSizeTuples
    
    def mdssStage(self):
        if (self.mpiComm != None):
            self.mpiComm.barrier()
        if ((self.mpiComm == None) or (self.mpiComm.Get_rank() == self.rootRank)):
            if (self.mdssSrcImagePath != None):
                rootLogger.info("MDSS staging %s" % self.mdssSrcImagePath)
                mdss.stage(self.mdssSrcImagePath, recursive=True, project=self.mdssPrj)
        if (self.mpiComm != None):
            self.mpiComm.barrier()

    def mdssGet(self):
        if (self.mpiComm != None):
            self.mpiComm.barrier()

        if ((self.mpiComm == None) or (self.mpiComm.Get_rank() == self.rootRank)):
            if (not os.path.exists(self.loclSrcDirPath)):
                rootLogger.info("Creating directory %s" % self.loclSrcDirPath)
                os.makedirs(self.loclSrcDirPath, mode="u+rwX")
            if (self.mdssSrcImagePath != None):
                rootLogger.info("MDSS getting %s" % self.mdssSrcImagePath)
                mdss.get(self.mdssSrcImagePath, self.loclSrcDirPath, recursive=True, project=self.mdssPrj)
        if (self.mpiComm != None):
            self.mpiComm.barrier()
    
    def loclSrcImagePathUncompress(self):
        uncompressedFiles = \
            mango.io.uncompressDdsData(
                self.loclSrcImagePath,
                preserve=False,
                mpiComm=self.mpiComm
            )
        if (not os.path.isdir(self.loclSrcImagePath)):
            self.loclSrcImagePath = uncompressedFiles[0]

    def loclSrcImagePathReadData(self):
        self.srcDds = mango.io.readDds(self.loclSrcImagePath)

    def createLoclDstDirPath(self):
        if (self.mpiComm != None):
            self.mpiComm.barrier()

        if ((self.mpiComm == None) or (self.mpiComm.Get_rank() == self.rootRank)):
            if (not os.path.exists(self.loclDstDirPath)):
                rootLogger.info("Creating directory %s" % self.loclDstDirPath)
                os.makedirs(self.loclDstDirPath, mode="u+rwX")

        if (self.mpiComm != None):
            self.mpiComm.barrier()
        
    def downsampleAndWriteData(self):
        self.createLoclDstDirPath()
        
        dstImageBasename, dstImageExt = mango.io.splitext(os.path.split(self.loclSrcImagePath)[1])
        dstImageBasename = os.path.join(self.loclDstDirPath, dstImageBasename + "_GDS")
        dstImageExt = ".nc"
        for voxSzTuple in self.voxelSizeTuples:
            rootLogger.info(
                "Downsampling %s image (voxel size=%s%s) to voxel size = %s"
                %
                (
                     os.path.split(self.loclSrcImagePath)[1],
                     tuple(self.srcDds.md.getVoxelSize()),
                     self.srcDds.md.getVoxelSizeUnit(),
                     voxSzTuple[2]
                )
            )
            dspDds = \
                mango.image.gaussian_downsample(
                    self.srcDds,
                    voxsz=[voxSzTuple[0],]*len(self.srcDds.shape),
                    voxunit=voxSzTuple[1]
                )
            loclDstImagePath = dstImageBasename + "x" + voxSzTuple[2] + dstImageExt
            rootLogger.info("Writing downsampled image to %s" % loclDstImagePath)
            mango.io.writeDds(loclDstImagePath, dspDds)
            self.loclDstImagePaths.append(loclDstImagePath)
        
        self.srcDds = None

    def loclDstImagePathsCompress(self):
        
        compLoclDstImagePaths = []
        for loclDstImagePath in self.loclDstImagePaths:
            compressFileList = mango.io.compressDdsData(loclDstImagePath, preserve=False, mpiComm=self.mpiComm)
            if (os.path.isdir(loclDstImagePath)):
                compLoclDstImagePaths.append(loclDstImagePath)
            else:
                compLoclDstImagePaths += compressFileList
        
        self.loclDstImagePaths = compLoclDstImagePaths

    def mdssPut(self):
        if (self.mpiComm != None):
            self.mpiComm.barrier()
        
        if (self.mdssDstDirPath != None):
            self.loclDstImagePathsCompress()

        if ((self.mpiComm == None) or (self.mpiComm.Get_rank() == self.rootRank)):
            if (self.mdssDstDirPath != None):
                if (not mdss.exists(self.mdssDstDirPath, project=self.mdssPrj)):
                    rootLogger.info("MDSS creating directory %s" % self.mdssDstDirPath)
                    mdss.makedirs(self.mdssDstDirPath, mode="u+rwX", project=self.mdssPrj)
                elif (not mdss.isdir(self.mdssDstDirPath, project=self.mdssPrj)):
                    raise Exception("MDSS destination %s exists but is not a directory." % self.mdssDstDirPath)

                rootLogger.info("MDSS putting %s" % self.loclDstImagePaths)
                mdss.put(self.loclDstImagePaths, self.mdssDstDirPath, recursive=True, project=self.mdssPrj)
                mdssDstImagePaths = \
                    [
                        os.path.join(self.mdssDstDirPath, os.path.split(loclDstImagePath)[1])
                        for loclDstImagePath in self.loclDstImagePaths
                    ]
                self.mdssDstImagePaths = [p for p in mdssDstImagePaths if mdss.exists(p, project=self.mdssPrj)]
        if (self.mpiComm != None):
            self.mpiComm.barrier()
        
    def __call__(self):
        self.mdssGet()
        self.loclSrcImagePathUncompress()
        rootLogger.info("Checking consistency of image data %s..." % self.loclSrcImagePath)
        msg = ddsFileMetaDataIsConsistent(self.loclSrcImagePath, self.mpiComm)
        if (msg == None):
            self.loclSrcImagePathReadData()
            self.downsampleAndWriteData()
            self.mdssPut()
        else:
            rootLogger.error(msg)
            self.mdssDstImagePaths, self.loclDstImagePaths = [],[]
        
        return self.mdssDstImagePaths, self.loclDstImagePaths
    
    def __str__(self):
        return \
            (
                (
                    "mdssSrcImagePath = %s\n        mdssDstDirPath   = %s"
                    +
                    "\n        mdssPrj          = %s"
                    +
                    "\n        loclSrcImagePath = %s\n        loclDstDirPath   = %s"
                    +
                    "\n        voxelSizeTuples  = %s"
                )
                %
                (
                    self.mdssSrcImagePath,
                    self.mdssDstDirPath,
                    self.mdssPrj,
                    self.loclSrcImagePath,
                    self.loclDstDirPath,
                    self.voxelSizeTuples
                )
            )

def parseVoxelSizes(voxelSizes):
    regEx = re.compile("\\s*([+-]?[0-9]+\\.?[0-9]*(?:[eE][+-]?[0-9]+)?)\\s*([a-z]*)\\s*")
    voxelSizeTuples = []
    for s in voxelSizes:
        s = s.lower()
        m = regEx.match(s)
        if (m != None):
            voxelSizeTuples.append((float(m.group(1)), m.group(2), s.strip()))
        else:
            raise Exception("Could not parse float and unit-string from '%s' in %s list." % (s, voxelSizes))
    
    return voxelSizeTuples


class BatchDownsampler:
    """
    """
    def __init__(
        self,
        mdssSrcRootDir,
        mdssDstRootDir,
        mdssPrj,
        loclSrcRootDir,
        loclDstRootDir,
        voxelSizeTuples
    ):
        self.mdssSrcRootDir = mdssSrcRootDir
        self.mdssDstRootDir = mdssDstRootDir
        self.loclSrcRootDir = loclSrcRootDir
        self.loclDstRootDir = loclDstRootDir
        self.mdssPrj = mdssPrj
        self.voxelSizeTuples = voxelSizeTuples
        rootLogger.debug("Parsed voxels sizes: %s" % (self.voxelSizeTuples,))
    
    def getRelativeImagePaths(self, sampleDirNames):
        regEx = re.compile("SrSkull2009_([0-9]*)_[0-9]")
        return [os.path.join(dirName, "tomo%s_nc" % dirName) for dirName in sampleDirNames if (regEx.match(dirName))]
    
    def createJobList(self):
        if (mpi.haveMpi4py):
            mpi.world.barrier()
            
        if (self.mdssSrcRootDir != None):
            sampleDirNames = mdss.listdir(self.mdssSrcRootDir, project=self.mdssPrj)
        else:
            sampleDirNames = os.listdir(self.loclSrcRootDir)
            
        sampleDirNames.sort()
        
        relImagePaths = self.getRelativeImagePaths(sampleDirNames)
        sampleDirNames = [os.path.split(relImagePath)[0] for relImagePath in relImagePaths]
        rootLogger.debug("relImagePaths = %s" % (relImagePaths, ))
        
        if (self.mdssSrcRootDir != None):
            mdssSrcImagePaths = [os.path.join(self.mdssSrcRootDir, relImagePath) for relImagePath in relImagePaths]
        else:
            mdssSrcImagePaths = [None,]*len(relImagePaths)
        if (self.mdssDstRootDir == None):
            mdssDstDirPaths = [None,]*len(relImagePaths)
        else:
            mdssDstDirPaths = [os.path.join(self.mdssDstRootDir, dirName) for dirName in sampleDirNames]

        loclSrcImagePaths = [os.path.join(self.loclSrcRootDir, relImagePath) for relImagePath in relImagePaths]
        if (self.loclDstRootDir == None):
            loclDstDirPaths = [os.path.split(imagePath)[0] for imagePath in loclSrcImagePaths]
        else:
            loclDstDirPaths = [os.path.join(self.loclDstRootDir, dirName) for dirName in sampleDirNames]

        jobList = []
        for i in range(0, len(sampleDirNames)):
            jobList.append(
                DownsampleJob(
                    mdssSrcImagePath = mdssSrcImagePaths[i],
                    mdssDstDirPath = mdssDstDirPaths[i],
                    mdssPrj = self.mdssPrj,
                    loclSrcImagePath = loclSrcImagePaths[i],
                    loclDstDirPath = loclDstDirPaths[i],
                    voxelSizeTuples = self.voxelSizeTuples
                )
            )
        if (mpi.haveMpi4py):
            mpi.world.barrier()
        
        return jobList
    
    def executeJobs(self, jobList):
        rootLogger.debug("\n\n".join(map(str, jobList)))
        
        for job in jobList:
            job.mdssStage()

        mdssCreatedImagePathList = []
        loclCreatedImagePathList = []
        for job in jobList:
            createdPathPair = job()
            mdssCreatedImagePathList += createdPathPair[0]
            loclCreatedImagePathList += createdPathPair[1]
        
        return mdssCreatedImagePathList, loclCreatedImagePathList

    def __call__(self):
        mdss.setDefaultProject(self.mdssPrj)
        jobList = self.createJobList()
        return self.executeJobs(jobList)



def getArgumentParser():
    """
    Returns object for parsing command line options.
    :rtype: argparse.ArgumentParser
    :return: Object to parse command line options.
    """
    descStr = \
        (
            "Creates downsampled images from netCDF files on MDSS."
        )
    
    argList = []
    argList.append(
        {
             'cmdLine':['-P', '--mdss-project',],
             'dest':'mdssPrj',
             'type':str,
             'metavar':'P',
             'default':None,
             'action':'store',
             'help': "MDSS project ID string."
        }
    )
    argList.append(
        {
             'cmdLine':['--mdss-dst',],
             'dest':'mdssDst',
             'type':str,
             'metavar':'D',
             'default':None,
             'action':'store',
             'help': "MDSS top level destination directory."
        }
    )
    argList.append(
        {
             'cmdLine':['--mdss-src',],
             'dest':'mdssSrc',
             'type':str,
             'metavar':'D',
             'default':None,
             'action':'store',
             'help': "MDSS top level source directory."
        }
    )

    argList.append(
        {
             'cmdLine':['--local-dst',],
             'dest':'loclDst',
             'type':str,
             'metavar':'D',
             'default':None,
             'action':'store',
             'help': "Local filesystem top level destination directory."
        }
    )
    argList.append(
        {
             'cmdLine':['--local-src',],
             'dest':'loclSrc',
             'type':str,
             'metavar':'D',
             'default':None,
             'action':'store',
             'help': "Local filesystem top level source directory."
        }
    )

    argList.append(
        {
             'cmdLine':['--voxel-sizes',],
             'dest':'voxelSizes',
             'type':str,
             'metavar':'F',
             'default':"150um,200um",
             'action':'store',
             'help': "Voxel size for downsampled images."
        }
    )

    argList.append(
        {
             'cmdLine':['-l','--logging-level'],
             'dest':'loggingLevel',
             'type':str,
             'metavar':'LVL',
             'default':"INFO",
             'action':'store',
             'help':"Level of logging output (one of 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')."
        }
    )

    if (haveArgParse):
    
        parser = argparse.ArgumentParser(description=descStr)
        for arg in argList:
            addArgumentDict = dict(arg)
            del addArgumentDict['cmdLine']
            parser.add_argument(*arg['cmdLine'], **addArgumentDict)
    else:
        parser = optparse.OptionParser(description=descStr)
        for arg in argList:
            addOptionDict = dict(arg)
            del addOptionDict['cmdLine']
            parser.add_option(*arg['cmdLine'], **addOptionDict)

    return parser


if (__name__ == "__main__"):
    argParser = getArgumentParser()
    if haveArgParse:
        args = argParser.parse_args()
    else:
        (args, argv) = argParser.parse_args()

    mpi.initialiseLoggers(
        [__name__, "mango.io", "mango.application"],
        logLevel=getattr(logging, args.loggingLevel)
    )

    bdsp = \
        BatchDownsampler(
            mdssSrcRootDir=args.mdssSrc,
            mdssDstRootDir=args.mdssDst,
            mdssPrj=args.mdssPrj,
            loclSrcRootDir=args.loclSrc,
            loclDstRootDir=args.loclDst,
            voxelSizeTuples=(parseVoxelSizes(args.voxelSizes.split(",")))
        )
    bdsp()
