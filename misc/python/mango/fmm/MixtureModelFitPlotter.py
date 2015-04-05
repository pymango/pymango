#!/usr/bin/env python
import sys

import mango.utils
from .Gaussian import *
from .MixtureModel import *
import re
import os
import os.path
import scipy
import numpy
from mango import mpi

logger, rootLogger = mpi.getLoggers(__name__)

def importMatplotlib():
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt

classColourList = \
    [
        "black",
        "#00ff00", # "Qt::green",
        "red",
        "blue",
        "cyan",
        "magenta",
        "darkred",
        "darkblue",
        "lightpink",
        "darkcyan",
        "#808000", #"darkyellow",
        "darkmagenta",
        "deeppink",
        "yellow",
        "lightblue",
        "darkgreen",
        "lightgrey",
        "darkgray",
        "beige",
        "cadetblue",
        "chocolate",
        "darkturquoise",
        "gainsboro",
        "hotpink",
        "khaki",
        "lightcoral",
        "maroon",
        "mintcream",
        "midnightblue",
        "olive",
        "palevioletred",
        "peachpuff"
    ]

_classColourDict = None

def getClassColourDict():
    if (_classColourDict == None):
        importMatplotlib()
        _classColourDict = dict()
        for i in range(len(classColourList)):
            _classColourDict[i] = matplotlib.colors.colorConverter.to_rgba(classColourList[i])
    return _classColourDict

class Bin:
    def __init__(self, start, size):
        self.start = start
        self.size = size
    
    def getStart(self):
        return self.start
    
    def getEnd(self):
        return self.start + self.size
    
    def getSize(self):
        return self.size

class HistogramBin(Bin):
    def __init__(self, start, size, count=0):
        Bin.__init__(self, start, size)
        self.count = count
    
    def __str__(self):
        return ("%s, %s, %s" % (self.getStart(), self.getSize(), self.count))

class PlotHistogramData:
    def __init__(self):
        self.binList = []
    
    def rebin(self, newBinSize):
        newBinList = []
        bin = HistogramBin((self.binList[0].start/newBinSize)*newBinSize, newBinSize, 0)
        newBinList.append(bin)
        for bin in self.binList:
            while (bin.start >= (newBinList[-1].start + newBinList[-1].size)):
                newBinList.append(HistogramBin(newBinList[-1].start + newBinList[-1].size, newBinList[-1].size, 0))
            newBinList[-1].count += bin.count
        rebinnedHist = PlotHistogramData()
        rebinnedHist.binList = newBinList
        
        return rebinnedHist

    def empty(self):
        return ((len(self.binList) <= 0) or (self.getNumSamples() <= 0))
    
    def getNumSamples(self):
        return sum([bin.count for bin in self.binList])

    def getPercentileValue(self, percentile):
        numSamp = self.getNumSamples()
        idx = 0
        count = self.binList[idx].count
        while ((idx < (len(self.binList))) and ((count*100.0)/numSamp < percentile)):
            count += self.binList[idx].count
            idx += 1
        
        if (idx >= len(self.binList)):
            return self.binList[-1].getEnd()
        return self.binList[idx].getStart()
    
    def getMinSamp(self):
        return self.binList[0].start

    def getMaxSamp(self):
        return self.binList[-1].start + self.binList[-1].size
    
    def generateLinePlotData(self, binRangePair=None):
        if (binRangePair == None):
            logger.debug("Setting bin range pair...")
            binRangePair = (self.getMinSamp(), self.getMaxSamp())
        logger.debug("len(self.binList)=%s, binRangePair = %s" % (len(self.binList),binRangePair,))
        x = []
        y = []
        if (len(self.binList) > 0):
            numSamples = float(self.getNumSamples())
            for bin in self.binList:
                if (bin.getStart() >= binRangePair[0]) and (bin.getEnd() <= binRangePair[1]):
                    x.append(bin.getStart())
                    x.append(bin.getEnd())
                    y.append(bin.count/(numSamples*bin.size))
                    y.append(bin.count/(numSamples*bin.size))
            if (len(x) > 0):
                x = [x[0],] + x + [x[-1],]
            if (len(y) > 0):
                y = [0, ] + y + [0,]
        return x,y

    def __str__(self):
        return "\n".join(map(str, self.binList)) 

class MultiClassPlotHistogramData:
    def __init__(self):
        self.binList = []
        self.countArray = scipy.zeros((0,0), dtype="uint32")
    
    def rebin(self, newBinSize):
        newBinList = []
        startBin = Bin((self.binList[0].start/newBinSize)*newBinSize, newBinSize)
        newBinList = [Bin(startBin.start + i*newBinSize, newBinSize) for i in range(0, (self.binList[-1].getEnd())/newBinSize +1)]
        newCountArray = scipy.zeros((len(newBinList), self.countArray.shape[1]), dtype=self.countArray.dtype)
        newBinIdx = 0
        for oldBinIdx in range(0, self.countArray.shape[0]):
            while (self.binList[oldBinIdx].start >= (newBinList[newBinIdx].getEnd())):
                newBinIdx += 1
            newCountArray[newBinIdx,:] += self.countArray[oldBinIdx,:]
            newBinList[-1].count += bin.count
        rebinnedHist = PlotHistogramData()
        rebinnedHist.binList = newBinList
        rebinnedHist.countArray = newCountArray
        
        return rebinnedHist

    def getNumSamples(self):
        return scipy.sum(self.countArray)
    
    def getMinSamp(self):
        return self.binList[0].start

    def getMaxSamp(self):
        return self.binList[-1].getEnd()
    
    def empty(self):
        return ((len(self.binList) <= 0) or (scipy.sum(self.countArray) <= 0))
    
    def generateLinePlotData(self, classIdx):
        x = []
        y = []
        if (len(self.binList) > 0):
            x = [self.binList[0].start]
            y = [0.0]
            numSamples = float(self.getNumSamples())
            for binIdx in range(0, len(self.binList)):
                bin = self.binList[binIdx]
                x.append(bin.start)
                x.append(bin.start+bin.size)
                y.append(self.countArray[binIdx, classIdx]/(numSamples*bin.size))
                y.append(self.countArray[binIdx, classIdx]/(numSamples*bin.size))
        
        return x,y

def csvParseHistogram(csvString):
    lineList = csvString.split("\n")
    lineIdx = 0
    line = lineList[lineIdx].strip()
    headerLineRegEx = re.compile("(.*intensity.*|.*size.*),.*count.*")
    while ((lineIdx < len(lineList)) and (headerLineRegEx.match(line) == None)) :
        lineIdx += 1
        line = lineList[lineIdx].strip()
    if (lineIdx >= len(lineList)):
        raise RuntimeError("Could not find header line match for reg-ex %s " % headerLineRegEx)
    binList = []
    dataLineRegEx = re.compile("\s*([^\s,]*)\s*,\s*([^\s,]*).*")
    while ((lineIdx+1) < len(lineList)):
        lineIdx += 1
        line = lineList[lineIdx].strip()
        if (len(line) > 0):
            mtch = dataLineRegEx.match(line)
            if (mtch != None):
                binList.append(HistogramBin(start=float(mtch.group(1)), size=1, count=int(mtch.group(2))))
    
    histogram = PlotHistogramData()
    histogram.binList = binList

    return histogram

def csvParseMixtureModelGroupings(csvString):
    lineList = csvString.split("\n")
    lineIdx = 0
    line = lineList[lineIdx].strip()
    headerLineRegEx = re.compile(".*mix.*model.*index.*,.*class.*index.*,.*group.*index.*")
    while ((lineIdx < len(lineList)) and (headerLineRegEx.match(line) == None)) :
        lineIdx += 1
        line = lineList[lineIdx].strip()
    if (lineIdx >= len(lineList)):
        raise RuntimeError("Could not find header line match for reg-ex %s " % headerLineRegEx)
    mixModelGroupings = dict()
    dataLineRegEx = re.compile("\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*.*")
    while ((lineIdx+1) < len(lineList)):
        lineIdx += 1
        line = lineList[lineIdx].strip()
        if (len(line) > 0):
            mtch = dataLineRegEx.match(line)
            if (mtch != None):
                mixModelIdx = int(mtch.group(1))
                if (not mixModelGroupings.has_key(mixModelIdx)):
                    classToGroup = dict()
                    groupToClass = dict()
                    mixModelGroupings[mixModelIdx] = [classToGroup, groupToClass]
                
                classIdx = int(mtch.group(2))
                groupIdx = int(mtch.group(3))
                
                if (mixModelGroupings[mixModelIdx][0].has_key(classIdx)):
                    raise RuntimeError(
                        "Mix model idx=%s: class index %s is already mapped to group %s" %
                        (mixModelIdx, classIdx, mixModelGroupings[mixModelIdx][0][classIdx])
                    )
                mixModelGroupings[mixModelIdx][0][classIdx] = groupIdx
                if (not mixModelGroupings[mixModelIdx][1].has_key(groupIdx)):
                    mixModelGroupings[mixModelIdx][1][groupIdx] = []
                mixModelGroupings[mixModelIdx][1][groupIdx].append(classIdx)

    return mixModelGroupings

def csvParseResultantHistogram(csvString):
    lineList = csvString.split("\n")
    lineIdx = 0
    line = lineList[lineIdx].strip()
    headerLineRegEx = re.compile(".*intensity.*(,[^,]*mixture[^,]*)+")
    while ((lineIdx < len(lineList)) and (headerLineRegEx.match(line) == None)) :
        lineIdx += 1
        line = lineList[lineIdx].strip()
    if (lineIdx >= len(lineList)):
        raise RuntimeError("Could not find header line match for reg-ex %s " % headerLineRegEx)
    else:
        numMixtures = len(",".split(line))-1
    binList = []
    cntList = []
    dataLineRegEx = re.compile("\s*([^\s,]*)\s*(,([^,]*))+")
    while ((lineIdx+1) < len(lineList)):
        lineIdx += 1
        line = lineList[lineIdx].strip()
        if (len(line) > 0):
            mtch = dataLineRegEx.match(line)
            if (mtch != None):
                numList = line.split(',')
                binList.append(Bin(start=int(numList[0].strip()), size=1))
                cntList.append(map(int,numList[1:]))
       
    histogram = MultiClassPlotHistogramData()
    histogram.binList = binList
    histogram.countArray = scipy.array(cntList, dtype=histogram.countArray.dtype)

    return histogram

class MixtureModelFitDataReader:
    def __init__(self, inputDir):
        self.inputDir = inputDir
        self.runId = ""
        self.mixtureModelsCsvFileName        = None
        self.fitResultantHistCsvFileNameList = []
        self.imgResultantHistCsvFileNameList = []
        self.imgFitHistogramCsvFileName      = None
        self.imgHistogramCsvFileName         = None
        self.wsRegionImgSzHistCsvFileName    = None
        self.wsRegionFitSzHistCsvFileName    = None
        self.mixModelGroupingsCsvFileName    = None
        dirList = os.listdir(self.inputDir)
        mixModelCsvRe           = re.compile("mixtureModels(.*)\\.csv")
        imgFitHistCsvRe         = re.compile("(fit|imgWsSeedNeighb)Histogram(.*)\\.csv")
        imgHistCsvRe            = re.compile("imgHistogram(.*)\\.csv")
        wsRegionImgSzHistCsvRe  = re.compile("wsRegionSizeImageHistogram(.*)\\.csv")
        wsRegionFitSzHistCsvRe  = re.compile("wsRegionSizeFitHistogram(.*)\\.csv")
        mixModelGroupingsCsvRe  = re.compile("mixtureModelsGroupings(.*)\\.csv")
        for f in dirList:
            mtch = mixModelGroupingsCsvRe.match(f)
            if (self.mixModelGroupingsCsvFileName == None) and (mtch != None):
                self.mixModelGroupingsCsvFileName = os.path.join(self.inputDir, f)
            else:
                mtch = mixModelCsvRe.match(f)
                if (self.mixtureModelsCsvFileName == None) and (mtch != None):
                    self.mixtureModelsCsvFileName = os.path.join(self.inputDir, f)
                    self.runId = mtch.group(1)

            mtch = imgFitHistCsvRe.match(f)
            if (self.imgFitHistogramCsvFileName == None) and (mtch != None):
                self.imgFitHistogramCsvFileName = os.path.join(self.inputDir, f)
            mtch = imgHistCsvRe.match(f)
            if (self.imgHistogramCsvFileName == None) and (mtch != None):
                self.imgHistogramCsvFileName = os.path.join(self.inputDir, f)
            mtch = wsRegionImgSzHistCsvRe.match(f)
            if (self.wsRegionImgSzHistCsvFileName == None) and (mtch != None):
                self.wsRegionImgSzHistCsvFileName = os.path.join(self.inputDir, f)
            mtch = wsRegionFitSzHistCsvRe.match(f)
            if (self.wsRegionFitSzHistCsvFileName == None) and (mtch != None):
                self.wsRegionFitSzHistCsvFileName = os.path.join(self.inputDir, f)

        self.mixtureModelList = []
        self.imgFitHistogram = None
        self.imgHistogram = None
        self.imgWsRegionSzHistogram = None
        self.fitWsRegionSzHistogram = None
        self.imgResultantHistogramList = []
        self.fitResultantHistogramList = []
        self.mixtureModelIndexGroupings = None
        
    def readData(self):
        logger.info("Reading mixture models from file %s..." % self.mixtureModelsCsvFileName )
        self.mixtureModelList = csvParseMixtureModels(file(self.mixtureModelsCsvFileName, 'r').read())
        logger.info("Mixture models:\n%s" % "\n".join(map(str, self.mixtureModelList)))

        if ((self.imgFitHistogramCsvFileName != None) and (os.path.exists(self.imgFitHistogramCsvFileName))):
            logger.info("Reading mixture model fit histogram from file %s..." % self.imgFitHistogramCsvFileName )
            self.imgFitHistogram = csvParseHistogram(file(self.imgFitHistogramCsvFileName, 'r').read())
        else:
            self.imgFitHistogram = PlotHistogramData()

        if ((self.imgHistogramCsvFileName != None) and (os.path.exists(self.imgHistogramCsvFileName))):
            logger.info("Reading image histogram from file %s..." % self.imgHistogramCsvFileName )
            self.imgHistogram = csvParseHistogram(file(self.imgHistogramCsvFileName, 'r').read())
            logger.info("Done reading image histogram from file %s." % self.imgHistogramCsvFileName )
        else:
            logger.info("Could not find image histogram data file=%s..." % self.imgHistogramCsvFileName )
            self.imgHistogram = PlotHistogramData()

        if ((self.wsRegionImgSzHistCsvFileName != None) and (os.path.exists(self.wsRegionImgSzHistCsvFileName))):
            logger.info("Reading watershed region size histogram from file %s..." % self.wsRegionImgSzHistCsvFileName )
            self.imgWsRegionSzHistogram = csvParseHistogram(file(self.wsRegionImgSzHistCsvFileName, 'r').read())
        else:
            logger.info("Could not find watershed region size histogram from file %s...skipping." % self.wsRegionImgSzHistCsvFileName )
            self.imgWsRegionSzHistogram = PlotHistogramData()

        if ((self.wsRegionFitSzHistCsvFileName != None) and (os.path.exists(self.wsRegionFitSzHistCsvFileName))):
            logger.info("Reading watershed fit sub-region histogram from file %s..." % self.wsRegionFitSzHistCsvFileName )
            self.fitWsRegionSzHistogram = csvParseHistogram(file(self.wsRegionFitSzHistCsvFileName, 'r').read())
        else:
            logger.info("Could not find watershed fit sub-region histogram file %s...skipping." % self.wsRegionFitSzHistCsvFileName )
            self.fitWsRegionSzHistogram = PlotHistogramData()

        if ((self.mixModelGroupingsCsvFileName != None) and (os.path.exists(self.mixModelGroupingsCsvFileName))):
            logger.info("Reading mixture model groupings file %s..." % self.mixModelGroupingsCsvFileName )
            self.mixtureModelIndexGroupings = csvParseMixtureModelGroupings(file(self.mixModelGroupingsCsvFileName, 'r').read())
        else:
            logger.info("Could not find mixture model groupings file %s...skipping." % self.mixModelGroupingsCsvFileName )
            self.mixtureModelIndexGroupings = None

        for mixModel in self.mixtureModelList:
            fName = os.path.join(self.inputDir, "imgResultantHistogram%s_k%02d.csv" % (self.runId, mixModel.size()))
            if (not os.path.exists(fName)):
                fName = os.path.join(self.inputDir, "resultantHistogram%s_k%02d.csv" % (self.runId, mixModel.size()))
            if (os.path.exists(fName)):
                logger.info("Reading resultant histogram data from file %s..." % fName )
                self.imgResultantHistCsvFileNameList.append(fName)
                self.imgResultantHistogramList.append(csvParseResultantHistogram(file(fName, "r").read()))
                logger.info(
                    "Finished reading %sx%s image-classification resultant histogram data from file %s."
                    %
                    (tuple(self.imgResultantHistogramList[-1].countArray.shape) + (fName,))
                )
            else:
                logger.info("Could not find resultant histogram data file %s...skipping." % fName )
                self.imgResultantHistCsvFileNameList.append(None)
                self.imgResultantHistogramList.append(MultiClassPlotHistogramData())

            fName = os.path.join(self.inputDir, "fitResultantHistogram%s_k%02d.csv" % (self.runId, mixModel.size()))
            if (os.path.exists(fName)):
                logger.info("Reading resultant histogram data from file %s..." % fName )
                self.fitResultantHistCsvFileNameList.append(fName)
                self.fitResultantHistogramList.append(csvParseResultantHistogram(file(fName, "r").read()))
                logger.info(
                    "Finished reading %sx%s mixture-model intensity fit classified resultant histogram data from file %s."
                    %
                    (tuple(self.fitResultantHistogramList[-1].countArray.shape) + (fName,))
                )
            else:
                logger.info("Could not find resultant histogram data file %s...skipping." % fName )
                self.fitResultantHistCsvFileNameList.append(None)
                self.fitResultantHistogramList.append(MultiClassPlotHistogramData())

    def getMixtureModelList(self):
        return self.mixtureModelList

    def getMixtureModelGroupings(self):
        return self.mixtureModelIndexGroupings
    
    def getImageResultantHistogramList(self):
        return self.imgResultantHistogramList

    def getFitResultantHistogramList(self):
        return self.fitResultantHistogramList
    
    def getImageFitHistogram(self):
        return self.imgFitHistogram
    
    def getImageHistogram(self):
        return self.imgHistogram
    
    def getImageWsRegionSizeHistogram(self):
        return self.imgWsRegionSzHistogram

    def getFitWsRegionSizeHistogram(self):
        return self.fitWsRegionSzHistogram

class MixtureModelFitPlotter:
    def __init__(self, inputDir=None, outputDir=None):
        importMatplotlib()
        self.plotImageExt               = ".png"
        self.plotFileNameSuffix         = ""
        self.plotImageSuffix            = self.plotFileNameSuffix + self.plotImageExt
        self.mixtureModelList           = None
        self.imgFitHistogram            = None
        self.imgHistogram               = None
        self.imgResultantHistList       = None
        self.fitResultantHistList       = None
        self.mixtureModelIndexGroupings = None
        self.outputDir                  = outputDir
        self.weightTruncPercent         = 99.0

        if (inputDir != None):
            dataReader = MixtureModelFitDataReader(inputDir)
            dataReader.readData()
            
            self.mixtureModelList           = dataReader.getMixtureModelList()
            self.imgFitHistogram            = dataReader.getImageFitHistogram()
            self.imgHistogram               = dataReader.getImageHistogram()
            self.imgResultantHistList       = dataReader.getImageResultantHistogramList()
            self.fitResultantHistList       = dataReader.getFitResultantHistogramList()
            self.imgWsRegionSzHistogram     = dataReader.getImageWsRegionSizeHistogram()
            self.fitWsRegionSzHistogram     = dataReader.getFitWsRegionSizeHistogram()
            self.mixtureModelIndexGroupings = dataReader.getMixtureModelGroupings()

            if (self.outputDir == None):
                self.outputDir = os.path.join(inputDir, "FmmPlots")
        if (self.outputDir != None):
            if (not os.path.exists(self.outputDir)):
                os.makedirs(self.outputDir)
        self.resultantPlotScaleList = [1,]
        self.numPdfEvalPoints = 2048
        self.xTruncRange = None # (self.imgHistogram.getMinSamp(), self.imgHistogram.getMaxSamp())
        self.yTruncRange = None
        self.newBinSize = 1
        self.approxNumHistBins = 256
        self.setApproxNumHistogramBins(self.approxNumHistBins)
        self.dpi = 192
        self.numPdfEvalPoints = 2048
        self.fontSize = "x-small"
        self.idString = ""
        self.mixelFracList = [0.01, 0.05, 0.1, 0.2]

    def setDpi(self, dpi):
        self.dpi = dpi

    def setMixelFractionList(self, mixelFracList):
        self.mixelFracList = list(mixelFracList)
        
    def setIntensityTruncationRange(self, pairList):
        self.xTruncRange = pairList

    def setFrequencyTruncationRange(self, pairList):
        self.yTruncRange = pairList

    def setIdString(self, idStr):
        self.idString = idStr

    def setPlotFileNameSuffix(self, fileNameSuffix):
        self.plotFileNameSuffix = fileNameSuffix

    def setLegendFontSize(self, fontSize):
        self.fontSize = fontSize

    def calcIntensityTruncationRange(self):
        minTruncs = []
        maxTruncs = []
        for mixModel in self.mixtureModelList:
            modelTruncRange = calcGaussianMixtureModel1dTruncRange(mixModel, self.weightTruncPercent)
            minTruncs.append(modelTruncRange[0])
            maxTruncs.append(modelTruncRange[1])
        minTruncs.sort()
        maxTruncs.sort()
        logger.debug(str(self.mixtureModelList))
        logger.debug(str(minTruncs))
        logger.debug(str(maxTruncs))
        return (minTruncs[len(minTruncs)/2], maxTruncs[len(maxTruncs)/2])

    def calcFrequencyLims(self):
        mn = 0.0
        mx = 0.0
        if (not self.imgHistogram.empty()):
            x,y=self.imgHistogram.generateLinePlotData(self.xTruncRange)
            mn = numpy.min(y)
            mx = numpy.max(y)
        if (not self.imgFitHistogram.empty()):
            x,y = self.imgFitHistogram.generateLinePlotData(self.xTruncRange)
            mn = min([mn, numpy.min(y)])
            mx = max([mx, numpy.max(y)])
        x = scipy.linspace(self.xTruncRange[0], self.xTruncRange[1], self.numPdfEvalPoints)
        for mixModel in self.mixtureModelList:
            y = mixModel.pdf(x)
            mn = min([mn, numpy.min(y)])
            mx = max([mx, numpy.max(y)])
            
        return (mn,mx)

    def setWeightTruncPercent(self, truncPercent):
        self.weightTruncPercent = truncPercent

    def setApproxNumHistogramBins(self, approxNumBins):
        self.approxNumHistBins = approxNumBins
        if (self.xTruncRange == None):
            self.xTruncRange = self.calcIntensityTruncationRange()
        if (self.yTruncRange == None):
            self.yTruncRange = self.calcFrequencyLims()
        self.newBinSize = max([1,int(math.floor(float(self.xTruncRange[1]-self.xTruncRange[0])/self.approxNumHistBins))])

    def setNumPdfEvalPoints(self, numPoints):
        self.numPdfEvalPoints = numPoints

    def setResultantPlotScaleList(self, scaleList):
        self.resultantPlotScaleList = scaleList

    def doMixModelPlots(self, mixModel, histLinePlotData, xTruncRange, yTruncRange=None, histLabel="", plotTitle=None):
        x = scipy.linspace(xTruncRange[0], xTruncRange[1], self.numPdfEvalPoints)
        p = mixModel.pdf(x)
        plt.plot(histLinePlotData[0], histLinePlotData[1], label=histLabel)
        plt.plot(x,p, label="mix model", linewidth=11, color=(0.25,0.25,0.25), alpha=0.35)
        if (yTruncRange != None):
            plt.ylim(yTruncRange)
        if (plotTitle == None):
            if (mixModel.__class__.__name__.find("PwGaussianMixelMixtureModel") >= 0):
                plotTitle = ("Mixel Mixture Model Fit, $k=%02d$" % mixModel.size())
            else:
                plotTitle = ("Gaussian Mixture Model Fit, $k=%02d$" % mixModel.size())
        plt.title(plotTitle + self.idString)
        plt.legend(loc="best")
        plt.xlim(xTruncRange)
        if (yTruncRange != None):
            plt.ylim(yTruncRange)

    def doMixModelMixturesPlots(self, mixModel, xTruncRange, yTruncRange=None, lineWidth=3, alpha=0.5, frac=0.05):

        x = scipy.linspace(xTruncRange[0], xTruncRange[1], self.numPdfEvalPoints)
        
        dIdx = 0
        for dist in mixModel:
            if (mixModel.__class__.__name__.find("PwGaussianMixelMixtureModel") >= 0):
                p = mixModel.ipdf(x, dIdx, frac)
            else:
                p = dist.wpdf(x)
            if (dist.__class__.__name__.find("WeightedGaussian") >= 0):
                labelStr = \
                    "$\mu_{%s}=%6.1f$, $\sigma_{%s}=%5.1f$,$w_{%s}=%6.4f$" \
                    % \
                    (dIdx, dist.getMean(), dIdx, dist.getStandardDeviation(), dIdx, dist.getWeight())
            else:
                labelStr = "mixture-$%s$, $w_{%s}=%6.4f$" % (dIdx, dIdx, dist.getWeight())
            rootLogger.debug("dist.getMean = %s, dist.getStandardDeviation = %s" % (dist.getMean(), dist.getStandardDeviation()))
            plt.plot(
                x,
                p,
                color=getClassColourDict()[dIdx],
                linewidth=lineWidth,
                alpha=0.5,
                label=labelStr
            )
            dIdx += 1

        plt.xlim(xTruncRange)
        if (yTruncRange != None):
            plt.ylim(yTruncRange)
        plt.legend(loc="best", prop={"size":self.fontSize})

    def doMixelModelMixturesPlots(self, mixModel, xTruncRange, yTruncRange=None, lineWidth=3, alpha=0.5):

        x = scipy.linspace(xTruncRange[0], xTruncRange[1], self.numPdfEvalPoints)
        lineStyles = ["-","--", "-.",":"]
        fIdx = 0
        for frac in self.mixelFracList:
            dIdx = 0
            for dist in mixModel:
                if (mixModel.__class__.__name__.find("PwGaussianMixelMixtureModel") >= 0):
                    p = mixModel.ipdf(x, dIdx, frac)
                    wght = mixModel.calcWeightIntegral(dIdx, 0.0, frac)
                else:
                    wght = dist.getWeight()
                    p = dist.wpdf(x)
                if (dist.__class__.__name__.find("WeightedGaussian") >= 0):
                    labelStr = \
                        "$\mu_{%s}=%6.1f$, $\sigma_{%s}=%5.1f$,$f_{%s}=%6.4f$,$w_{%s}=%6.4f$" \
                        % \
                        (dIdx, dist.getMean(), dIdx, dist.getStandardDeviation(), dIdx, frac, dIdx, wght)
                else:
                    labelStr = "mixture-$%s$, $w_{%s}=%6.4f$" % (dIdx, dIdx, wght)
                rootLogger.debug("dist.getMean = %s, dist.getStandardDeviation = %s" % (dist.getMean(), dist.getStandardDeviation()))
                plt.plot(
                    x,
                    p,
                    color=getClassColourDict()[dIdx],
                    linewidth=lineWidth,
                    linestyle=lineStyles[fIdx % len(lineStyles)],
                    alpha=0.5,
                    label=labelStr
                )
                dIdx += 1
            fIdx += 1

        plt.xlim(xTruncRange)
        if (yTruncRange != None):
            plt.ylim(yTruncRange)
        plt.legend(loc="best", prop={"size":self.fontSize})

    def doResultantHistPlots(self, mixModel, hist, xTruncRange, yTruncRange=None):
        for c in range(0, mixModel.size()):
            x,y = hist.generateLinePlotData(c)
            plt.plot(
                x,
                y,
                color=getClassColourDict()[c],
                linewidth=5,
                alpha=0.5,
                label=None
            )
        plt.xlim(xTruncRange)
        if (yTruncRange != None):
            plt.ylim(yTruncRange)

        plt.title("Resultant Histograms and Mixture Model Fit, $k=%02d$%s" % (mixModel.size(), self.idString))
        plt.legend(loc="best", prop={"size":self.fontSize})

    def doPlots(self):
        self.plotImageSuffix = self.plotFileNameSuffix + self.plotImageExt
        xTruncRange = self.xTruncRange
        yTruncRange = self.yTruncRange
        yTruncRange = (0, yTruncRange[1])
        newBinSize = self.newBinSize
        logger.info("Intensity truncation range = %s." % (xTruncRange,))

        plt.clf()
        
        if (not self.imgWsRegionSzHistogram.empty()):
            logger.info("Plotting image watershed region size histogram...")
            x,y = self.imgWsRegionSzHistogram.generateLinePlotData()
            plt.xlabel("watershed region size")
            plt.ylabel("count")
            plt.plot(x,y, label="entire image watershed")

            for percentile in [100,99,95,90]:
                plt.title("Entire Image Watershed Region Size Histogram, %s percentile%s" % (percentile, self.idString))
                plt.xlim(0, self.imgWsRegionSzHistogram.getPercentileValue(percentile+0.1))
                fName = ("imgWsRegionSizeHistogram_pct%03d" % percentile) + self.plotImageSuffix
                logger.info("Saving " + fName + "...")
                plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
    
            plt.clf()
        
        if (not self.fitWsRegionSzHistogram.empty()):
            logger.info("Plotting image watershed region size histogram...")
            x,y = self.fitWsRegionSzHistogram.generateLinePlotData()
            plt.plot(x,y, label="entire image watershed")
            plt.title("Fit Watershed Region Size Histogram" + self.idString)
            plt.xlabel("fit region size")
            plt.ylabel("count")
            fName = "fitWsRegionSizeHistogram" + self.plotImageSuffix
            logger.info("Saving " + fName + "...")
            plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
    
            plt.clf()

        if (not self.imgFitHistogram.empty()):
            logger.info("Plotting image fit-histogram...")
            x,y = self.imgFitHistogram.generateLinePlotData()
            plt.plot(x,y, label="fit hist")
        logger.info("Plotting image histogram...")
        x,y = self.imgHistogram.generateLinePlotData()
        plt.plot(x,y, label="img hist")
        
        matplotlib.pyplot.legend(loc="best")
        plt.title("Image intensity histograms" + self.idString)
        
        fName = "histogramsFull" + self.plotImageSuffix
        logger.info("Saving " + fName + "...")
        plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
        plt.xlim(xTruncRange)
        plt.ylim(yTruncRange)
        fName = "histogramsTrunc" + self.plotImageSuffix
        logger.info("Saving " + fName + "...")
        plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
        
        plt.clf()

        if (not self.imgFitHistogram.empty()):
            logger.info("Plotting re-binned image fit-histogram...")
            x,y = self.imgFitHistogram.rebin(newBinSize).generateLinePlotData()
            plt.plot(x,y, label="fit hist")
        logger.info("Plotting re-binned image histogram...")
        x,y = self.imgHistogram.rebin(newBinSize).generateLinePlotData()
        plt.plot(x,y, label="img hist")
        
        matplotlib.pyplot.legend(loc="best")
        plt.title("Image intensity histograms" + self.idString)

        fName = ("histogramsFullBinned%s" % self.approxNumHistBins) + self.plotImageSuffix
        logger.info("Saving " + fName + "...")
        plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
        plt.xlim(xTruncRange)
        plt.ylim(yTruncRange)
        fName = ("histogramsTruncBinned%s" % self.approxNumHistBins) + self.plotImageSuffix
        logger.info("Saving " + fName + "...")
        plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)

        imgX,imgY = (x,y)
        numImgSamples = self.imgHistogram.getNumSamples()
        numFitSamples = self.imgFitHistogram.getNumSamples()
        if (not self.imgFitHistogram.empty()):
            x,y = self.imgFitHistogram.rebin(newBinSize).generateLinePlotData()
            fitPltLabel="fit hist"
        else:
            x,y = self.imgHistogram.rebin(newBinSize).generateLinePlotData()
            fitPltLabel="img hist"

        for mixModelIdx in range(0, len(self.mixtureModelList)):
            mixModel = self.mixtureModelList[mixModelIdx]
            plt.clf()
            logger.info("Plotting mixture model fit k=%02d" % mixModel.size())
            self.doMixModelPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange, histLinePlotData=(x,y), histLabel=fitPltLabel)
            fName = ("mixModelFit_k%02d" % mixModel.size()) + self.plotImageSuffix
            logger.info("Saving " + fName + "...")
            plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)

            logger.info("Plotting individual mixtures of k=%02d mixture model fit..." % mixModel.size())
            if (mixModel.__class__.__name__.find("PwGaussianMixelMixtureModel") >= 0):
                self.doMixelModelMixturesPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange)
            else:
                self.doMixModelMixturesPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange)
            yLim = list(self.yTruncRange)
            for scale in self.resultantPlotScaleList:
                scaleLim = (yLim[0], yLim[1]/scale)
                fName = ("mixModelFit_k%02d_mixes_scl%s" % (mixModel.size(), scale)) + self.plotImageSuffix
                logger.info("Saving " + fName + "...")
                plt.ylim(scaleLim)
                plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)

            plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)

            if (not self.imgResultantHistList[mixModelIdx].empty()):
                logger.info("Plotting resultant image-histogram classified data for k=%d mixture model..." % mixModel.size())
                logger.info("Num img samples = %8d, num resultant hist samples = %8d." % (numImgSamples, self.imgResultantHistList[mixModelIdx].getNumSamples()))
                plt.clf()
                self.doMixModelPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange, histLinePlotData=(imgX,imgY), histLabel="img hist")
                self.doMixModelMixturesPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange, lineWidth=1, alpha=1.0)
                self.doResultantHistPlots(mixModel, self.imgResultantHistList[mixModelIdx], xTruncRange=xTruncRange, yTruncRange=yTruncRange)
                plt.title("Image Intensities Resultant Histograms and Mixture Model Fit, $k=%02d$%s" % (mixModel.size(), self.idString))
                yLim = list(self.yTruncRange)
                for scale in self.resultantPlotScaleList:
                    scaleLim = (yLim[0], yLim[1]/scale)
                    fName = ("mixModelFit_k%02d_mixes_img_resultant_scl%s" % (mixModel.size(), scale)) + self.plotImageSuffix
                    logger.info("Saving " + fName + "...")
                    plt.ylim(scaleLim)
                    plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)

            if (not self.fitResultantHistList[mixModelIdx].empty()):
                logger.info("Plotting resultant fit-histogram classified data for k=%d mixture model..." % mixModel.size())
                logger.info("Num fit samples = %8d, num resultant hist samples = %8d." % (numFitSamples, self.fitResultantHistList[mixModelIdx].getNumSamples()))
                plt.clf()
                self.doMixModelPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange, histLinePlotData=(x,y), histLabel="fit hist")
                self.doMixModelMixturesPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange, lineWidth=1, alpha=1.0)
                self.doResultantHistPlots(mixModel, self.fitResultantHistList[mixModelIdx], xTruncRange=xTruncRange, yTruncRange=yTruncRange)
                plt.title("Fit Intensities Resultant Histograms and Mixture Model Fit, $k=%02d$%s" % (mixModel.size(), self.idString))
                yLim = list(self.yTruncRange)
                for scale in self.resultantPlotScaleList:
                    scaleLim = (yLim[0], yLim[1]/scale)
                    fName = ("mixModelFit_k%02d_mixes_fit_resultant_scl%s" % (mixModel.size(), scale)) + self.plotImageSuffix
                    logger.info("Saving " + fName + "...")
                    plt.ylim(scaleLim)
                    plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)

        if ((self.mixtureModelIndexGroupings != None) and (len(self.mixtureModelIndexGroupings) > 0)):
            mixModelIndexList = self.mixtureModelIndexGroupings.keys()
            for mixModelIdx in mixModelIndexList:
                ungroupedMixModel = self.mixtureModelList[mixModelIdx]
                mixModel = groupMixtureModel(ungroupedMixModel, self.mixtureModelIndexGroupings[mixModelIdx][1])
                plt.clf()
                logger.info("Plotting grouped mixture model fit group-size=%02d, k=%02d" % (mixModel.size(), ungroupedMixModel.size()))
                self.doMixModelPlots(
                    mixModel,
                    xTruncRange=xTruncRange,
                    yTruncRange=yTruncRange,
                    histLinePlotData=(x,y),
                    histLabel=fitPltLabel,
                    plotTitle = ("Grouped-Gaussian Mixture Model, $k=%02d$ mixture distributions." % mixModel.size())
                )
                fName = ("mixModelFit_group%02d_k%02d" % (mixModel.size(), ungroupedMixModel.size())) + self.plotImageSuffix
                logger.info("Saving " + fName + "...")
                plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
    
                logger.info("Plotting grouped mixtures of group-size=%02d, k=%02d mixture model fit..." % (mixModel.size(), ungroupedMixModel.size()))
                self.doMixModelMixturesPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange)
                yLim = list(self.yTruncRange)
                for scale in self.resultantPlotScaleList:
                    scaleLim = (yLim[0], yLim[1]/scale)
                    fName = ("mixModelFit_group%02d_k%02d_mixes_scl%s" % (mixModel.size(), ungroupedMixModel.size(), scale)) + self.plotImageSuffix
                    logger.info("Saving " + fName + "...")
                    plt.ylim(scaleLim)
                    plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
    
                plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
    
                if (False):
                    if (not self.imgResultantHistList[mixModelIdx].empty()):
                        logger.info("Plotting resultant image-histogram classified data for k=%d mixture model..." % mixModel.size())
                        logger.info("Num img samples = %8d, num resultant hist samples = %8d." % (numImgSamples, self.imgResultantHistList[mixModelIdx].getNumSamples()))
                        plt.clf()
                        self.doMixModelPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange, histLinePlotData=(imgX,imgY), histLabel="img hist")
                        self.doMixModelMixturesPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange, lineWidth=1, alpha=1.0)
                        self.doResultantHistPlots(mixModel, self.imgResultantHistList[mixModelIdx], xTruncRange=xTruncRange, yTruncRange=yTruncRange)
                        plt.title("Image Intensities Resultant Histograms and Mixture Model Fit, $k=%02d$%s" % (mixModel.size(),self.idString))
                        yLim = list(self.yTruncRange)
                        for scale in self.resultantPlotScaleList:
                            scaleLim = (yLim[0], yLim[1]/scale)
                            fName = ("mixModelFit_k%02d_mixes_img_resultant_scl%s" % (mixModel.size(), scale)) + self.plotImageSuffix
                            logger.info("Saving " + fName + "...")
                            plt.ylim(scaleLim)
                            plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
        
                    if (not self.fitResultantHistList[mixModelIdx].empty()):
                        logger.info("Plotting resultant fit-histogram classified data for k=%d mixture model..." % mixModel.size())
                        logger.info("Num fit samples = %8d, num resultant hist samples = %8d." % (numFitSamples, self.fitResultantHistList[mixModelIdx].getNumSamples()))
                        plt.clf()
                        self.doMixModelPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange, histLinePlotData=(x,y), histLabel="fit hist")
                        self.doMixModelMixturesPlots(mixModel, xTruncRange=xTruncRange, yTruncRange=yTruncRange, lineWidth=1, alpha=1.0)
                        self.doResultantHistPlots(mixModel, self.fitResultantHistList[mixModelIdx], xTruncRange=xTruncRange, yTruncRange=yTruncRange)
                        plt.title("Fit Intensities Resultant Histograms and Mixture Model Fit, $k=%02d$%s" % (mixModel.size(),self.idString))
                        yLim = list(self.yTruncRange)
                        for scale in self.resultantPlotScaleList:
                            scaleLim = (yLim[0], yLim[1]/scale)
                            fName = ("mixModelFit_k%02d_mixes_fit_resultant_scl%s" % (mixModel.size(), scale)) + self.plotImageSuffix
                            logger.info("Saving " + fName + "...")
                            plt.ylim(scaleLim)
                            plt.savefig(os.path.join(self.outputDir, fName), dpi=self.dpi)
