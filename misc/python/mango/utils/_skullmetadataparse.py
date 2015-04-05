
import scipy as sp
import numpy as np
import scipy.io
import scipy.misc
import scipy.ndimage

import os
import os.path
import copy
import re
from datetime import datetime
import shutil

from mango import mpi
from mango.utils._dicom2netcdf import *

logger, rootLogger = mpi.getLoggers(__name__)

class SkullDcmMetaData(VolumeMetaData):
    def __init__(self):
        VolumeMetaData.__init__(self)
        self.duplicateOf    = None
        self.sampleImageId  = None
        self.sampleDataDir  = None 
        self.dcmDir         = None
        self.infoDir        = None
        self.infoFileName   = None
        self.driveId        = None
        self.dateTime       = None
    
    def getSampleIdNumber(self):
        if (self.sampleId != None):
            rex = re.compile("[^0-9]*([0-9]*)[^0-9]*")
            m = rex.match(self.sampleId)
            if (m != None):
                return int(m.group(1).lstrip("0"))
        else:
            logger.error("self.sampleId is None, need string:")
            self.log(logger.error)
            raise RuntimeError("self.sampleId is None, need string.")
        
        raise RuntimeError("Could not match sample ID number in sample ID string '%s'" * self.sampleId)

    def log(self, logfunc=rootLogger.debug):
        VolumeMetaData.log(self, logfunc)
        logfunc("sampleImageId   = %s" % self.sampleImageId)
        logfunc("sampleDataDir   = %s" % self.sampleDataDir )
        logfunc("dcmDir          = %s" % self.dcmDir)
        logfunc("infoDir         = %s" % self.infoDir)
        logfunc("infoFileName    = %s" % self.infoFileName)
        logfunc("driveId         = %s" % self.driveId)
        logfunc("dateTime        = %s" % self.getDateTime())

    def getDateTime(self):
        if (self.dateTime == None):
            # (RLA 22-10-2009 12.54.53)
            dateTimeRegEx = re.compile(".*\\(\s*RLA\s*([0-9]+)-([0-9]+)-([0-9]+)\s*([0-9]+)\\.([0-9]+)\\.([0-9]+)\s*\\).*")
            mtch = dateTimeRegEx.match(self.sampleDataDir)
            if (mtch != None):
                self.dateTime = \
                    datetime(
                        year   = int(mtch.group(3)),
                        month  = int(mtch.group(2)),
                        day    = int(mtch.group(1)),
                        hour   = int(mtch.group(4)),
                        minute = int(mtch.group(5)),
                        second = int(mtch.group(6)),
                    )
            else:
                self.dateTime = datetime(year=2009, month=1, day=1)
        return self.dateTime

    def getYear(self):
        return self.getDateTime().year

def cmpVoxelSize(md0, md1):
    r = min(md0.voxelSize) - min(md1.voxelSize)
    if (r < 0):
        return -1
    if (r > 0):
        return 1
    return 0

class SkullDcmMetaDataParser:
    def __init__(self, searchDirList = ["."]):
        self.searchDirList = searchDirList
        self.dataDirRegEx  = re.compile("([Ss][Kk][Uu][Ll][Ll][^0-9]*|[Ss][Rr][^0-9]*)([0-9]*)(.*)")
        self.infoDirRegEx  = re.compile("[Ii][Nn][Ff][Oo]")
        self.dicomDirRegEx = re.compile(".*[Dd][Ii][Cc][Oo][Mm]")
        self.driveRegEx    = re.compile(".*Drive_(.).*")

    def findSampleDataDirectories(self):
        searchDirList = copy.deepcopy(self.searchDirList)
        dataDirList = []
        while (len(searchDirList) > 0):
            nextSearchDirList = []
            for searchDir in searchDirList:
                listing = os.listdir(searchDir)
                for f in listing:
                    fPath = os.path.join(searchDir, f)
                    if (os.path.isdir(fPath)):
                        m = self.dataDirRegEx.match(f)
                        if (m != None):
                            dataDirList.append(fPath)
                        else:
                            nextSearchDirList.append(fPath)
            searchDirList = nextSearchDirList
        dataDirList.sort()
        return dataDirList

    def parseInfoMetaDataFromDir(self, dir):
        listing = os.listdir(dir)
        md = SkullDcmMetaData()
        md.sampleDataDir = dir
        mtch = self.driveRegEx.match(md.sampleDataDir)
        if (mtch != None):
            md.driveId = mtch.group(1)

        foundInfoDirs = []
        foundDcmDirs = []
        for f in listing:
            mtch = self.infoDirRegEx.match(f)
            if ((mtch != None) and os.path.isdir(os.path.join(dir, f))):
                foundInfoDirs.append(f)
            else:
                mtch = self.dicomDirRegEx.match(f)
                if ((mtch != None) and os.path.isdir(os.path.join(dir, f))):
                    foundDcmDirs.append(f)
        
        if ((len(foundInfoDirs) == 1) and (len(foundDcmDirs) == 1)):
            md.infoDir = foundInfoDirs[0]
            md.dcmDir  = foundDcmDirs[0]
            readXtekMetaData(os.path.join(md.sampleDataDir, md.infoDir), md)
        else:
            raise RuntimeError(
                ("Did not find unique DICOM and INFO dirs in data dir %s" % dir)
                +
                ("\nINFO  dirs:%s" % (foundInfoDirs,))
                +
                ("\nDICOM dirs:%s" % (foundDcmDirs,))
            )

        return md

    def parseInfoMetaData(self, dataDirList):
        mdList = []
        for dir in dataDirList:
            md = self.parseInfoMetaDataFromDir(dir)
            if (md != None):
                mdList.append(md)
        return mdList
        
    def parseMetaData(self):
        mdList = None
        rootLogger.info("Searching top level dir for skull sample directories...")
        dataDirList = self.findSampleDataDirectories()
        rootLogger.info("Found %3d top-level data directories..." % len(dataDirList))
        rootLogger.debug("\n".join(dataDirList))
        rootLogger.info("Parsing info data for individual data sets...")
        mdList = self.parseInfoMetaData(dataDirList)
        
        return mdList

class SkullDcmRenamer:
    def __init__(self, metaDataList, outputDir = None):
        self.initialise(metaDataList, outputDir)


    def isDuplicateImage(self, md, mdOther):
        return \
            (
                sp.all((md.voxelSize - mdOther.voxelSize) < 1.0e-5)
                and
                sp.all(md.voxelUnit == mdOther.voxelUnit)
                and
                sp.all(md.volumeSize == mdOther.volumeSize)
            )

    def initialise(self, metaDataList, outputDir = None):
        self.outputDir = outputDir
        self.origMetaDataList = metaDataList
        mdDict = dict()
        for md in metaDataList:
            sampIdNum = md.getSampleIdNumber()
            if (not (sampIdNum in mdDict.keys())):
                mdDict[sampIdNum] = []
            mdDict[sampIdNum].append(md)
        sampIdNumList = mdDict.keys()
        sampIdNumList.sort()
        metaDataPairList = []
        renamedMetaDataList = []
        for sampIdNum in sampIdNumList:
            mdList = mdDict[sampIdNum]
            mdList.sort(cmp = cmpVoxelSize)
            imgIdx = 0
            mdImgIdx = 0
            isDuplicate = False
            for mdIdx in range(0, len(mdList)):
                md = mdList[mdIdx]
                if (mdIdx > 0):
                    isDuplicate = self.isDuplicateImage(mdList[mdImgIdx], md)
                    if (not isDuplicate):
                        imgIdx += 1
                        mdImgIdx = mdIdx

                rmd = copy.deepcopy(md)
                sampleImageId = ("SrSkull%04d_%04d_%d" % (md.getYear(), md.getSampleIdNumber(), imgIdx))
                if (isDuplicate):
                    rmd.duplicateOf = sampleImageId
                rmd.sampleImageId = sampleImageId
                outDir = self.outputDir
                if (self.outputDir == None):
                    outDir = os.path.split(md.sampleDataDir)[0]
                rmd.sampleDataDir = os.path.join(outDir, sampleImageId)
                rmd.dcmDir         = "DICOM"
                rmd.infoDir        = "INFO"
                metaDataPairList.append((md, rmd))
        self.metaDataPairList = metaDataPairList

    def writeCsvSummary(self, csvOutputSummaryFileName):
        strRowList = []
        headerStrRow =\
            [
                 "orig parent dir",
                 "rename parent dir",
                 "orig samp dir",
                 "rename samp dir",
                 "orig info dir",
                 "rename info dir",
                 "orig dcm dir",
                 "rename dcm dir",
                 "image shape",
                 "voxel size",
                 "drive id"
            ]
        maxColWidth = [0,]*len(headerStrRow)
        maxColWidth  = [max([maxColWidth[i], len(headerStrRow[i])]) for i in range(len(headerStrRow))]

        for pair in self.metaDataPairList:
            if (pair[1].duplicateOf == None):
                strRow = \
                    [
                        os.path.split(pair[0].sampleDataDir)[0],
                        os.path.split(pair[1].sampleDataDir)[0],
                        os.path.split(pair[0].sampleDataDir)[1],
                        os.path.split(pair[1].sampleDataDir)[1],
                        pair[0].infoDir,
                        pair[1].infoDir,
                        pair[0].dcmDir,
                        pair[1].dcmDir,
                        str(pair[1].volumeSize),
                        str(pair[1].voxelSize),
                        pair[1].driveId,
                    ]
            else:
                strRow = \
                    [
                        os.path.split(pair[0].sampleDataDir)[0],
                        os.path.split(pair[1].sampleDataDir)[0],
                        os.path.split(pair[0].sampleDataDir)[1],
                        "dup_of_" + os.path.split(pair[1].sampleDataDir)[1],
                        pair[0].infoDir,
                        pair[1].infoDir,
                        pair[0].dcmDir,
                        pair[1].dcmDir,
                        str(pair[1].volumeSize),
                        str(pair[1].voxelSize),
                        pair[1].driveId,
                    ]

            maxColWidth  = [max([maxColWidth[i], len(strRow[i])]) for i in range(len(strRow))]
            strRowList.append(strRow)
        if (mpi.haveMpi4py and mpi.world.Get_rank() == 0):
            rootLogger.info("Writing to file %s..." % csvOutputSummaryFileName)
            f = file(csvOutputSummaryFileName, 'w')
            f.write(
                ", ".join(
                    [
                         ("%%%ds" %  maxColWidth[i]) % headerStrRow[i] for i in range(len(headerStrRow))
                    ]
                )
            )
            
            for strRow in strRowList:
                f.write("\n")
                f.write(
                    ", ".join(
                        [
                             ("%%%ds" %  maxColWidth[i]) % strRow[i] for i in range(len(strRow))
                        ]
                    )
                )

    def doImageRename(self, mdSrc, mdDst):
        srcInfoDir = os.path.join(mdSrc.sampleDataDir, mdSrc.infoDir)
        srcDicomDir = os.path.join(mdSrc.sampleDataDir, mdSrc.dcmDir)
        dstInfoDir = os.path.join(mdDst.sampleDataDir, mdDst.infoDir)
        dstDicomDir = os.path.join(mdDst.sampleDataDir, mdDst.dcmDir)
        
        logger.info("Copying %s to %s..." % (srcInfoDir, dstInfoDir))
        shutil.copytree(srcInfoDir, dstInfoDir)
        logger.info("Copying %s to %s..." % (srcDicomDir, dstDicomDir))
        shutil.copytree(srcDicomDir, dstDicomDir)

    def doDataRenameUsingMpi(self, mdPairList):
        rootRank = 0
        mdPairList = mpi.world.bcast(mdPairList, root=rootRank)
        startIdx = mpi.world.Get_rank()
        idxStep  = mpi.world.Get_size()
        for idx in range(startIdx, len(mdPairList), idxStep):
            pair = mdPairList[idx]
            self.doImageRename(pair[0], pair[1])
 
    def doDataRename(self, mdPairList):
        if (mpi.haveMpi4py and (mpi.world.Get_size() > 1)):
            self.doDataRenameUsingMpi(mdPairList)
        else:
            for pair in mdPairList:
                self.doImageRename(pair[0], pair[1])

    def doRename(self):
        nonDupPairList = []
        for pair in (self.metaDataPairList):
            if (pair[1].duplicateOf == None):
                nonDupPairList.append(pair)
        self.doDataRename(nonDupPairList)

