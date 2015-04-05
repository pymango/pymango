__doc__ = \
"""
======================================================================
Multi-modal shale CT imaging analysis (:mod:`mango.application.shale`)
======================================================================

.. currentmodule:: mango.application.shale

Analysis of shale dry, dry-after, Iodine-stained and Diiodomethane-stained CT images. 

Functions
=========

.. autosummary::
   :toctree: generated/
   
   convertShaleHist2dToTernary - Converts 2D histogram data to ternary Histogram data.
   resolveHistogramDuplicateEntries - Resolves duplicate/close ternary coordinates for triangulation.
   generateShaleTernaryPlot - Generates ternary histogram plots from micro-porosity segmented data.
"""

from .io import readCsvHistData
from .plot import ternaryPlot
import numpy as np
import scipy as sp

import mango.mpi as mpi
logger, rootLogger = mpi.getLoggers(__name__)

class MicroPorosityBinToPercentMapper:
    """
    Maps micro-porosity segmentation class values to percentage value.
    """
    def __init__(self, bin0percent, bin100percent):
        self.bin0percent = bin0percent
        self.bin100percent = bin100percent
    
    def __call__(self, binIdx):
        
        if (binIdx <= self.bin0percent):
            percentVal = 0.0
        elif (binIdx >= self.bin100percent):
            percentVal = 100.0
        else:
            percentVal = 100.0*(binIdx - self.bin0percent)/float((self.bin100percent-self.bin0percent))
            
        return percentVal

INVALID_PROPORTION_RESCALE = 0
INVALID_PROPORTION_DISCARD = 1
def convertShaleHist2dToTernary(
    histData,
    invalidProportionMethod=INVALID_PROPORTION_RESCALE,
    cropRange=None,
    cropIndex=None
):
    """
    Returns a :samp:`(N,4)` shaped :obj:`numpy.array` of ternary
    *(mineral,pore,organic,frequency)* data. The input :samp:`histData`
    is 2D histogram data generated from a pair of *micro-porosity* segmented
    images. The x-axis data is assumed to be the CH2I2 differenced data.
    
    :type histData: :obj:`mango.application.io.HistData`  
    :param histData: 2D histogram data of micro-porosity segmentation image pair
       (micro-porosity segmentation of CH2I2-image minus dry-after-image image and
       micro-porosity segmentation of I2-image minus dry-image). Assumes that
       the CH2I2 data is the x-axis of the :samp:`histData`.
    :type invalidProportionMethod: int
    :param invalidProportionMethod: Method used to resolve data
       points where :samp:`pore_percent+organic_percent` exceeds 100%.
    :rtype: :obj:`numpy.ndarray`
    :rtype: A :samp:`(N,4)` shaped :obj:`numpy.ndarray`, where :samp:`N=num_x_bins*num_y_bins`.
       Each row of the returned array is :samp:`(mineral-percent, pore-percent, organic-percent, count)`.
    """
    numPorosityBins   = histData.hist1dData0.size
    numOrganicityBins = histData.hist1dData1.size
    
    pMapper = MicroPorosityBinToPercentMapper(1.0, numPorosityBins-2.0)
    oMapper = MicroPorosityBinToPercentMapper(1.0, numOrganicityBins-2.0)
    #pMapper = MicroPorosityBinToPercentMapper(0.0, numPorosityBins-1.0)
    #oMapper = MicroPorosityBinToPercentMapper(0.0, numOrganicityBins-1.0)
    
    ternList = []
    for pIdx in range(0, numPorosityBins):
        for oIdx in range(0, numOrganicityBins):
            porosity   = 100.0 - pMapper(pIdx)
            organicity = 100.0 - oMapper(oIdx)
            if (porosity + organicity > 100.0):
                if ((invalidProportionMethod == INVALID_PROPORTION_RESCALE)):
                    f = 100.0/(porosity + organicity + 1.0e-3)
                    porosity   *= f
                    organicity *= f
                elif (invalidProportionMethod == INVALID_PROPORTION_DISCARD):
                    continue

            minerality = 100.0-porosity-organicity
            valList = [minerality, porosity, organicity]
            if (
                ((cropRange == None) or (cropIndex == None))
                or
                ((valList[cropIndex] >= cropRange[0]) and (valList[cropIndex] <= cropRange[1]))
            ):
                if ((cropRange != None) and (cropIndex!=None)):
                    valList[cropIndex] = 100.0*(valList[cropIndex] - cropRange[0])/(cropRange[1]-cropRange[0])
                minerality, porosity, organicity = valList
                ternList.append([minerality, porosity, organicity, histData.hist2dData[oIdx, pIdx]])
    
    return sp.array(ternList, dtype="float64")

def resolveHistogramDuplicateEntries(ternaryArray, tol=1.0e-4):
    """
    Remove duplicate/close coordinates in the :samp:`(N,4)` shaped :samp:`ternaryArray`
    histogram.
    """
    numCoords = ternaryArray.shape[0]
    msk = sp.ones((numCoords,), dtype="bool")
    coordArray = ternaryArray[:, 0:3]
    nonDupList = []
    for i in range(0, numCoords):
        if (msk[i]):
            coord = coordArray[i,:]
            d = coordArray - coord
            d = sp.sqrt(sp.sum(d*d, axis=1))
            
            nearCoordIdxs = sp.where(sp.logical_and(d < tol, msk))
            nonDupList.append(coord.tolist() + [sp.sum(ternaryArray[:,-1][nearCoordIdxs]),])
            msk[nearCoordIdxs] = False
    
    return sp.array(nonDupList, dtype=ternaryArray.dtype)

def generateShaleTernaryPlot(
    histData,
    cropRange = None,
    cropIndex = None,
    invalidProportionMethodList=[INVALID_PROPORTION_RESCALE,INVALID_PROPORTION_DISCARD],
    doLogScale = True,
    cmap = None,
    contourNumLevels = 32,
    contourNumLines = None,
    doContourColourBar = False,
    shading='gouraud'
):
    """
    Returns a list of (:obj:`matplotlib.figure.Figure`, :obj:`str`) pairs with ternary
    *mineral-pore-organic* 2D histogram plots.
    
    :type histData: :obj:`mango.application.io.HistData`  
    :param histData: 2D histogram data of micro-porosity segmentation image pair
       (micro-porosity segmentation of CH2I2-image minus dry-after-image image and
       micro-porosity segmentation of I2-image minus dry-image). Assumes that
       the CH2I2 data is the x-axis of the :samp:`histData`.
    :rtype: :obj:`list` of pairs
    :return: List of (:obj:`matplotlib.figure.Figure`, :obj:`str`) pairs.
    """
    import matplotlib.pyplot as plt

    if (cmap == None):
        cmap = plt.cm.get_cmap("gray_r")
    figList = []
    origCountSum = sp.sum(histData.hist2dData)
    labels=("mineral", "pore", "organic")
    for invalidProportionMethod in invalidProportionMethodList:
        ternaryArray = convertShaleHist2dToTernary(histData, invalidProportionMethod, cropRange=cropRange, cropIndex=cropIndex)
        ternaryArray = resolveHistogramDuplicateEntries(ternaryArray, tol=0.9990)
        countSum = sp.sum(ternaryArray[:,-1])
        percentCountsDiscarded = 100.0*(origCountSum-countSum)/float(origCountSum)
        titleOffset = 1.08
        fontSize = "small"
        if (invalidProportionMethod == INVALID_PROPORTION_DISCARD):
            invalidProportionMethodStr = "discard"
            titleStr = "Percent Counts Discarded = %g%%" % percentCountsDiscarded
        else:
            invalidProportionMethodStr = "rescale"
            titleStr = "Rescaled Points"
        
        if (cropIndex != None) and (cropRange != None):
            titleStr += " (%s cropped to range [%s%%,%s%%])" % ((labels[cropIndex], ) + cropRange)
            fontSize = "x-small"
        
        logger.info(titleStr)

        logger.debug("ternaryArray.shape=%s", (ternaryArray.shape,))
        logger.debug("ternaryArray:\n")
        logger.debug(str(ternaryArray))
        logger.debug(
            "ternaryArray (min-x,min-y,min-z)=(%s,%s,%s)"
            %
            (np.min(ternaryArray[:,0]), np.min(ternaryArray[:,1]), np.min(ternaryArray[:,2]))
        )
        logger.debug(
            "ternaryArray (max-x,max-y,max-z)=(%s,%s,%s)"
            %
            (np.max(ternaryArray[:,0]), np.max(ternaryArray[:,1]), np.max(ternaryArray[:,2]))
        )
    
        ternaryPlotData, ternAxes = ternaryPlot(ternaryArray[:, 0:3], labels=labels)
        logger.debug("ternaryPlotData.shape=%s", (ternaryPlotData.shape,))
        logger.debug("ternaryPlotData:\n")
        logger.debug(str(ternaryPlotData))

        ax, fig = ternAxes.createAxes()
        ax.scatter(ternaryPlotData[:,0], ternaryPlotData[:,1])
        figList.append((fig, "coords_%s" %  invalidProportionMethodStr))

        if (doLogScale):
            ternaryArray[:,-1] = sp.log(1.0+ternaryArray[:,-1])
            pass

    
        ax, fig = ternAxes.createAxes()
        ax.triplot(ternaryPlotData[:,0], ternaryPlotData[:,1])
        figList.append((fig, "coords_triangulated_%s" %  invalidProportionMethodStr))
        ax, fig = ternAxes.createAxes()
        ax.tripcolor(ternaryPlotData[:,0], ternaryPlotData[:,1], ternaryArray[:,-1], shading=shading, cmap=cmap)
        t = plt.title(titleStr, fontsize=fontSize)
        t.set_y(titleOffset)
        figList.append((fig, "ternary_triangulated_%s" %  invalidProportionMethodStr))
        
        ax, fig = ternAxes.createAxes()
        if (contourNumLines == None):
            contourNumLines = contourNumLevels//2
        cs = ax.tricontourf(ternaryPlotData[:,0], ternaryPlotData[:,1], ternaryArray[:,-1],contourNumLevels, cmap=cmap)
        if (doContourColourBar):
            fig.colorbar(cs, shrink=0.9)
        contourPlt = ax.tricontour(ternaryPlotData[:,0], ternaryPlotData[:,1], ternaryArray[:,-1],contourNumLines, colors='k', linewidths=1)
        t = plt.title(titleStr, fontsize=fontSize)
        t.set_y(titleOffset)
        figList.append((fig, "ternary_contour_triangulated_%s" %  invalidProportionMethodStr))

    return figList



__all__ = [s for s in dir() if not s.startswith('_')]

