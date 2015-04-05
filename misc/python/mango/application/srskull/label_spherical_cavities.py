#!/usr/bin/env python

import mango
import mango.io
import mango.image
import mango.fmm
import mango.mpi as mpi

import scipy as sp
import numpy as np
import logging
import re
import os
import os.path
import sys

haveArgParse = False
try:
    import argparse
    haveArgParse = True
except:
    import optparse

logger, rootLogger = mpi.getLoggers(__name__)

def elemfreq(mskDds):
    return mango.itemfreq(mskDds)
   
_sphericalCavityLabelerCount = 0
class SphericalCavityLabeler:
    """
    Performs convex hull mask of skull, calculates max-covering-radius
    transform on the cavity voxels and subsequent labelling of the cavities. 
    """
    
    def __init__(self, outDir=None, outSuffix=None, outExt=None):
        global _sphericalCavityLabelerCount
        if ((outDir == None) or (outDir == "")):
            outDir = "."
        if (outSuffix == None):
            outSuffix = "%02d" % _sphericalCavityLabelerCount
        if (outExt == None):
            outExt = ".nc"

        self.outDir     = outDir
        self.outSuffix  = outSuffix
        self.outExt     = outExt
        
        self.se = mango.image.sphere_se(3)
        self.writeIntermediateImages = False
        
        _sphericalCavityLabelerCount += 1

    def writeIntermediateDds(self, intermediateSuffix, dds):
        if (self.writeIntermediateImages):
            outPath = \
                os.path.join(
                    self.outDir,
                    (
                        "%s%s%s%s"
                        %
                        (
                            dds.mtype.fileNameBase(),
                            self.outSuffix,
                            intermediateSuffix,
                            self.outExt
                        )
                     )
                )
            mango.io.writeDds(outPath, dds)

    def maskLowStddVoxels(self, dds, nMeanDds, nStddDds):
        unique = np.unique(sp.where(nStddDds.subd.asarray() <= 1.0/3, dds.subd.asarray(), dds.mtype.maskValue()))
        unique = unique[sp.where(unique != dds.mtype.maskValue())]
        if (dds.mpi.comm != None):
            unique = dds.mpi.comm.allreduce(unique.tolist(), op=mpi.SUM)
            unique = np.unique(unique)
        rootLogger.info("Unique constant stdd values = %s" % (unique,))
        rootLogger.info("Creating mask from unique constant values...")
        mskDds = mango.zeros_like(dds, mtype="segmented")
        for uVal in unique:
            mskDds.asarray()[...] = sp.where(dds.asarray() == uVal, 1, mskDds.asarray())
        rootLogger.info("Done creating mask from unique constant values.")
    
        rootLogger.info("Labeling connected constant zero-stdd regions...")
        mskDds.updateHaloRegions()
        mskDds.mirrorOuterLayersToBorder(False)
        self.writeIntermediateDds("_000ZeroStddForLabeling", mskDds)
        lblDds = mango.image.label(mskDds, 1)
        rootLogger.info("Done labeling connected constant stdd regions.")
        self.writeIntermediateDds("_000ZeroStdd", lblDds)
        
        countThresh = 0.01 * sp.product(lblDds.shape)
        rootLogger.info("Eliminating large clusters...")
        lblDds = mango.image.eliminate_labels_by_size(lblDds, minsz=int(countThresh), val=lblDds.mtype.maskValue())
        self.writeIntermediateDds("_000ZeroStddLargeEliminated", lblDds)
    
        rootLogger.info("Assigning mask values...")
        mskDds.subd.asarray()[...] = \
            sp.where(lblDds.subd.asarray() == lblDds.mtype.maskValue(), True, False)
        self.writeIntermediateDds("_000ZeroStddMskd", mskDds)
        del lblDds
        for tmpDds in [dds, nMeanDds, nStddDds]:
            tmpDds.subd.asarray()[...] = \
                sp.where(mskDds.subd.asarray(), tmpDds.mtype.maskValue(), tmpDds.subd.asarray())
    
    
    def eliminateSmallClusters(self, mskDds, clusterSzThreshFraction):
        """
        Performs a labelling of non-mask-value connected components of
        the :samp:`mskDds` image and eliminates clusters/objects
        which have number of voxels which is less
        than :samp:`clusterSzThreshFraction*mango.count_non_masked(mskDds)`.
        :type mskDds: :obj:`mango.Dds`
        :param mskDds: This image is modified by eliminating small clusters/objects
           (by setting small-cluster-voxels to value :samp:`mskDds.mtype.maskValue()`.
        :type clusterSzThreshFraction: :obj:`float`
        :param clusterSzThreshFraction: Value in interval :samp:`[0,1]`. Threshold fraction of
           total non-masked :samp:`mskDds` voxels for eliminating small clusters/objects.
        """
        elimClustersSmallerThan = int(sp.ceil(mango.count_non_masked(mskDds)*clusterSzThreshFraction))
        segDds = mango.ones_like(mskDds, mtype="segmented")
        mango.copy_masked(mskDds, segDds)
        rootLogger.info("eliminateSmallClusters: Labeling mskDds masked connected components...")
        lblDds = mango.image.label(segDds, 1)
        rootLogger.info("eliminateSmallClusters: Done labeling mskDds masked connected components...")
        self.writeIntermediateDds("_111MskDdsLabels", lblDds)
        rootLogger.info("eliminateSmallClusters: Eliminating clusters of size range [%s, %s]..." % (0, elimClustersSmallerThan))
        lblDds = mango.image.eliminate_labels_by_size(lblDds, val=lblDds.mtype.maskValue(), minsz=0, maxsz=elimClustersSmallerThan)
        rootLogger.info("eliminateSmallClusters: Done eliminating clusters in size range [%s, %s]." % (0, elimClustersSmallerThan))
        rootLogger.info("eliminateSmallClusters: Copying small-cluster mask to mskDds...")
        mango.copy_masked(lblDds, mskDds)
        rootLogger.info("eliminateSmallClusters: Done copying small-cluster mask to mskDds.")
    
    def eliminatePercentileTails(self, mskDds, loPercentile=10.0, hiPercentile=90.0):
        """
        Trims lower and/or upper image histogram tails by replacing :samp:`mskDds`
        voxel values with :samp:`mskDds.mtype.maskValue()`. 
        """
        rootLogger.info("Eliminating percentile tails...")
        rootLogger.info("Calculating element frequencies...")
        elems, counts = elemfreq(mskDds)
        rootLogger.info("elems:\n%s" % (elems,))
        rootLogger.info("counts:\n%s" % (counts,))
        cumSumCounts = sp.cumsum(counts, dtype="float64")
        percentiles = 100.0*(cumSumCounts/float(cumSumCounts[-1]))
        percentileElems = elems[sp.where(sp.logical_and(percentiles > loPercentile, percentiles < hiPercentile))]
        loThresh = percentileElems[0]
        hiThresh = percentileElems[-1]
        rootLogger.info("Masking percentiles range (%s,%s) = (%s,%s)" % (loPercentile, hiPercentile, loThresh, hiThresh))
        mskDds.asarray()[...] = \
            sp.where(
                sp.logical_and(
                    sp.logical_and(mskDds.asarray() >= loThresh, mskDds.asarray() <= hiThresh),
                    mskDds.asarray() != mskDds.mtype.maskValue()
                ),
                mskDds.asarray(),
                mskDds.mtype.maskValue()
            )
        rootLogger.info("Done eliminating percentile tails.")
    
    def calcSphericalCavityLabels(self, dds):
    
        mpidims = dds.mpi.shape
        if (dds.mpi.comm != None):
            mpidims = (dds.mpi.comm.Get_size(), 1, 1)
        dds = mango.copy(dds, mpidims=mpidims)
        rootLogger.info("Calculating neighbourhood mean image...")
        fltDds = mango.copy(dds, mtype="tomo_float", halo=self.se.getHaloSize())
        nMeanDds = mango.image.mean_filter(fltDds, self.se)
        nMeanDds = mango.copy(nMeanDds, halo=(0,0,0))
        rootLogger.info("Calculating neighbourhood stdd image...")
        nStddDds = mango.image.stdd_filter(fltDds, self.se)
        nStddDds = mango.copy(nStddDds, halo=(0,0,0))
        del fltDds
        self.maskLowStddVoxels(dds, nMeanDds, nStddDds)
        rootLogger.info("Calculating mean vs stdd histogram...")
        h2d, edges = mango.image.histogramdd([nMeanDds, nStddDds], bins=(1024, 8192))
        h2d = h2d[:, :-2]
        rootLogger.info("h2d.shape = %s, edges[0].shape=%s, edges[1].shape=%s" % (h2d.shape, edges[0].shape, edges[1].shape))
        rootLogger.info("Done calculating mean vs stdd histogram...")
        maxIdx = np.unravel_index(np.argmax(h2d), h2d.shape)
        rootLogger.info("np.argmax(h2d) = %s" % (maxIdx,))
        backgroundMean = 0.5 * (edges[0][maxIdx[0]] + edges[0][maxIdx[0] + 1])
        backgroundStdd = 0.5 * (edges[1][maxIdx[1]] + edges[1][maxIdx[1] + 1])

        mskDds = mango.copy(dds)
        rootLogger.info("Background (mean,stdd) = (%s, %s)." % (backgroundMean, backgroundStdd))
        mskDds.subd.asarray()[...] = \
            sp.where(
                sp.logical_or(
                    nStddDds.subd.asarray() < (8 * backgroundStdd),
                    nMeanDds.subd.asarray() < (backgroundMean + 3 * backgroundStdd),
                ),
                mskDds.mtype.maskValue(),
                mskDds.subd.asarray()
            )
        del nMeanDds, nStddDds

        self.writeIntermediateDds("_AaaPrePercentileTailMask", mskDds)
        self.eliminatePercentileTails(mskDds, 1.0, 92.5)
        rootLogger.info("Calculating neighbourhood stdd image...")
        nStddDds = mango.image.stdd_filter(mskDds, self.se)
        self.eliminatePercentileTails(nStddDds, 33.0, 100.0)
        rootLogger.info("Copying stdd percentile tail mask to mskDds...")
        mango.copy_masked(nStddDds, mskDds)
        rootLogger.info("Done copying stdd percentile tail mask to mskDds.")
        self.writeIntermediateDds("_AaaPstPercentileTailMask", mskDds)
        rootLogger.info("Eliminating small clusters from mskDds...")
        self.eliminateSmallClusters(mskDds, 0.1)
        rootLogger.info("Done eliminating small clusters from mskDds.")
        self.writeIntermediateDds("_AaaPstSmallClusterMask", mskDds)
        del nStddDds
        segEdtDds = mango.zeros_like(mskDds, mtype="segmented")
        segEdtDds.asarray()[...] = sp.where(mskDds.asarray() == mskDds.mtype.maskValue(), 0, 1)
        self.writeIntermediateDds("_AaaPreCvxHullMask", segEdtDds)
        rootLogger.info("Calculating convex hull...")
        cvxHullMsk = mango.image.convex_hull_3d(segEdtDds, inputmsk=0, outhull=segEdtDds.mtype.maskValue(), inhull=1)
        segEdtDds.asarray()[...] = sp.where(cvxHullMsk.asarray() == cvxHullMsk.mtype.maskValue(), 1, segEdtDds.asarray())
        rootLogger.info("Done calculating convex hull.")

        self.writeIntermediateDds("_AaaPstCvxHullMask", segEdtDds)
        segEdtDds.setFacesToValue(1)
        rootLogger.info("Calculating EDT image...")
        edtDds = mango.image.distance_transform_edt(segEdtDds, val=0)
        self.writeIntermediateDds("_AaaPstCvxHullMaskEdt", edtDds)
        rootLogger.info("Calculating MCR image...")
        mcrDds = mango.image.max_covering_radius(edtDds, maxdist=0.5*(np.min(edtDds.shape)), filecache=True)
        mango.copy_masked(cvxHullMsk, mcrDds)
        rootLogger.info("Calculating (min,max) MCR values...")
        mcrMin, mcrMax = mango.minmax(mcrDds)
        rootLogger.info("Masking small MCR values...")
        mcrDds.asarray()[...] = sp.where(mcrDds.asarray() >= 0.05*mcrMax, mcrDds.asarray(), mcrDds.mtype.maskValue())
        self.writeIntermediateDds("_AaaPstCvxHullMaskMcr", mcrDds)
        del cvxHullMsk, edtDds
        #
        # Normalise the intensities so that a relative-gradient is computed for the largest
        # MCR radii.
        rootLogger.info("Normalising MCR image...")
        mnmx = mango.minmax(mcrDds)
        tmpDds = mango.copy(mcrDds)
        tmpDds.asarray()[...] -= mnmx[1]
        tmpDds.asarray()[...] *= tmpDds.asarray()
        tmpDds.asarray()[...] = 1+mnmx[1]*np.exp(-(tmpDds.asarray())/(2*0.133*0.133*(mnmx[1]*mnmx[1])))
        mcrDds.asarray()[...] = sp.where(mcrDds.asarray() > 0, mcrDds.asarray()/tmpDds.asarray(), mcrDds.asarray())
        rootLogger.info("Calculating MCR image gradient...")
        grdMcrDds = mango.image.discrete_gaussian_gradient_magnitude(mcrDds, 0.65, errtol=0.01)
        grdMcrDds.asarray()[...] = sp.where(grdMcrDds.asarray() <= 3.0e-2, mcrDds.asarray(), mcrDds.mtype.maskValue())
        rootLogger.info("Calculating unique MCR low-gradient values...")
        u = mango.unique(grdMcrDds)
        rootLogger.info("Converting low gradient MCR image to binary segmentation...")
        segDds = mango.map_element_values(grdMcrDds, lambda x: x in u, mtype="segmented")
        rootLogger.info("Labeling low gradient MCR regions...")
        mango.copy_masked(mcrDds, segDds)
        lblDds = mango.image.label(segDds, val=1, connectivity=26, dosort=True)
        self.writeIntermediateDds("_AaaPstCvxHullMaskMcrGrdLbl", lblDds)
        del segDds, grdMcrDds

        rootLogger.info("Calculating Principal Moment of Inertia...")
        self.pmoi, self.pmoi_axes, self.com = mango.image.moment_of_inertia(mskDds)
        rootLogger.info("Done calculating Principal Moment of Inertia.")
        return lblDds, mcrDds, segEdtDds
    
    
def getArgumentParser():
    """
    Returns object for parsing command line options.
    :rtype: argparse.ArgumentParser
    :return: Object to parse command line options.
    """
    descStr = \
        (
            "Labels spherical cavity regions of skull image."
        )
    
    argList = []

    argList.append(
        {
             'cmdLine':['-l', '--logging-level'],
             'dest':'loggingLevel',
             'type':str,
             'metavar':'LVL',
             'default':"INFO",
             'action':'store',
             'help':"Level of logging output (one of 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')."
        }
    )

    argList.append(
        {
             'cmdLine':['-v', '--mango-verbosity'],
             'dest':'mangoVerbosity',
             'type':str,
             'metavar':'LVL',
             'default':"low",
             'action':'store',
             'help':"Level of mango C++ logging verbosity level (one of 'none', 'low', 'medium', 'high', 'extreme')."
        }
    )

    argList.append(
        {
             'cmdLine':['-s', '--save-intermediates'],
             'dest':'saveIntermediates',
             'type':str,
             'metavar':'LVL',
             'default':"False",
             'action':'store',
             'help':"If True, (a lot of) intermediate images are written to file."
        }
    )

    if (haveArgParse):
        parser = argparse.ArgumentParser(description=descStr)

        parser.add_argument(
            'fileNameList',
            metavar='F',
            type=str,
            nargs='+',
            help='Mango NetCDF image files/dirs.'
        )

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
        argv = args.fileNameList
    else:
        (args, argv) = argParser.parse_args()

    mpi.initialiseLoggers(
        [__name__, "mango.core", "mango.image", "mango.io", "mango.application"],
        logLevel=getattr(logging, args.loggingLevel)
    )
    mango.setLoggingVerbosityLevel(args.mangoVerbosity)
    ddsFileNameList = argv
    args.saveIntermediates = eval(args.saveIntermediates)
    
    for ddsFileName in ddsFileNameList:
        ucDdsFileName = mango.io.uncompressDdsData(ddsFileName, preserve=False)
        if (not os.path.isdir(ddsFileName)):
            ddsFileName = ucDdsFileName[0]
        imgDds = mango.io.readDds(ddsFileName)
        inDir, inPrefix, inSuffix, inExt = mango.io.splitpath(ddsFileName)
        outDir = inDir
        outSuffix = inSuffix
        outExt = inExt 
        scl = SphericalCavityLabeler(outDir, outSuffix, outExt)
        scl.writeIntermediateImages = args.saveIntermediates
        sclDds, mcrDds, segEdtDds = scl.calcSphericalCavityLabels(imgDds)
        
        for p in ((segEdtDds, "SCLsegedt"), (mcrDds, "SCLmcr"), (sclDds, "SCLlbl")):
            outFileName = os.path.join(outDir, "%s%s%s_%s%s" % (outDir, p[0].mtype.fileNameBase(), outSuffix, p[1], outExt))
            mango.io.writeDds(outFileName, p[0])
