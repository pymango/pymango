
import mango.utils
from .Gaussian import *
import copy
import re
import sys
import scipy as sp
import numpy as np
import scipy.stats
from mango import mpi

try:
    from mango.fmm._PwGaussianMixelMixtureModel import csvParsePgmmm
except:
    csvParsePgmmm = None

logger, rootLogger = mpi.getLoggers(__name__)

class MixtureModel:
    def __init__(self):
        self.distList = []
    
    def size(self):
        return len(self.distList)
    
    def append(self, dist):
        self.distList.append(dist)
    
    def clear(self):
        self.distList = []
    
    def pdf(self, x):
        return sum([d.wpdf(x) for d in self.distList])
    
    def cdf(self, x):
        return sum([d.wcdf(x) for d in self.distList])
    
    def evalBinProbabilities(self, binEndPts):
        c = self.cdf(binEndPts)
        return c[1:]-c[0:c.size-1]

    def sort(self, cmp):
        self.distList.sort(cmp)
    
    def getGroupings(self):
        return None

    def __getitem__(self, i):
        return self.distList[i]

    def __iter__(self):
        return iter(self.distList)

    def getMixtureWeightArray(self, dtype="float64"):
        return sp.array([d.getWeight() for d in self.distList], dtype=dtype)

    def normaliseWeights(self, dtype="float64"):
        wf = sp.sum(self.getMixtureWeightArray(dtype))
        for d in self.distList:
            d.setWeight(d.getWeight()/wf)

class WeightedMixtureModel (MixtureModel, WeightedDistribution):
    def __init__(self, weight=1.0):
        MixtureModel.__init__(self)
        WeightedDistribution.__init__(self, weight)


class GaussianMixtureModel1d (MixtureModel):
    def __init__(self, means=None, stddevs=None, mixWeights=None):
        MixtureModel.__init__(self)
        if ((means != None) or (stddevs != None) or (mixWeights != None)):
            for i in range(len(means)):
                m = 0
                s = 1
                w = 1
                try:
                    m = means[i]
                except:
                    m = 0
                try:
                    s = stddevs[i]
                except:
                    s = 1
                try:
                    w = mixWeights[i]
                except:
                    w = 1
                
                self.append(WeightedGaussian1d(mean=m, stddev=s, weight=w))
    
    def sort(self, cmp=None):
        if (cmp == None):
            cmp = gaussian1dCompare
        MixtureModel.sort(self, cmp)
        
    def getMeanArray(self, dtype="float64"):
        return sp.array([d.getMean() for d in self.distList], dtype=dtype)

    def getVarianceArray(self, dtype="float64"):
        return sp.array([d.getVariance() for d in self.distList], dtype=dtype)

    def getStandardDeviationArray(self, dtype="float64"):
        return sp.array([d.getStandardDeviation() for d in self.distList], dtype=dtype)

    def getMixtureWeightArray(self, dtype="float64"):
        return sp.array([d.getWeight() for d in self.distList], dtype=dtype)
    
    def getParameterList(self):
        return [self.getMeanArray(), self.getStandardDeviationArray(), self.getMixtureWeightArray()]

    def __str__(self):
        s = ""
        if (self.size() > 0):
            mn = self.getMeanArray()
            sd = self.getStandardDeviationArray()
            wt = self.getMixtureWeightArray()
            s = "%9.2f" % mn[0]
            for mean in mn[1:]:
                s += ", %9.2f" % mean
            for stddev in sd:
                s += ", %9.2f" % stddev
            for weight in wt:
                s += ", %10.8f" % weight
            
        return s

    def setParameters(self, means, stddevs, mixWeights):
        self.clear()
        for i in range(len(means)):
            self.append(WeightedGaussian1d(mean=means[i], stddev=stddevs[i], weight=mixWeights[i]))

    def rvs(self, size=1):
        numDist = self.size()
        prms = sp.transpose(sp.array([[dist.getMean(), dist.getStandardDeviation(), dist.getWeight()] for dist in self.distList]))
        idx = np.dot(np.random.multinomial(1, prms[2,:], size=size),np.arange(numDist))
        rvs = scipy.zeros((size,), dtype=prms.dtype)
        for dIdx in range(numDist):
            msk = sp.where(idx == dIdx)
            numSamp = sp.sum(sp.where(idx == dIdx, 1, 0))
            rvs[msk] = scipy.stats.norm.rvs(loc=prms[0, dIdx], scale=prms[1,dIdx], size=numSamp)
        return rvs

def groupMixtureModel(ungroupedMixModel, groupMixIdxDict):
    grpMixModel = MixtureModel()
    grpIdxList = groupMixIdxDict.keys()
    grpIdxList.sort()
    for grpIdx in grpIdxList:
        wMixDist = WeightedMixtureModel()
        weight = 0.0
        for clsIdx in groupMixIdxDict[grpIdx]:
            weight += ungroupedMixModel[clsIdx].getWeight()
            wMixDist.append(ungroupedMixModel[clsIdx])
        wMixDist.normaliseWeights()
        wMixDist.setWeight(weight)
        grpMixModel.append(wMixDist)
    
    return grpMixModel

def csvParseGaussianMixtureModels(csvString):
    rootLogger.info("Parsing Gaussian Mixture Model...")
    lineList = csvString.split("\n")
    lineIdx = 0
    line = lineList[lineIdx].strip()
    headerLineRegEx = re.compile(".*mean.*,.*st.*dev.*,.*weight.*")
    while ((lineIdx < len(lineList)) and (headerLineRegEx.match(line) == None)) :
        lineIdx += 1
        line = lineList[lineIdx].strip()
    if (lineIdx >= len(lineList)):
        raise RuntimeError("Could not find header line match for reg-ex %s " % headerLineRegEx)
    
    mixList = [GaussianMixtureModel1d()]
    prmLineRegEx = re.compile("\s*([^\s,]*)\s*,\s*([^\s,]*)\s*,\s*([^\s,]*)\s*.*")
    while ((lineIdx+1) < len(lineList)):
        lineIdx += 1
        line = lineList[lineIdx].strip()
        
        if (len(line) > 0):
            mtch = prmLineRegEx.match(line)
            if (mtch != None):
                logger.debug("%s" % (mtch.groups(),))
                mixList[-1].append(
                    WeightedGaussian1d(
                        mean = float(mtch.group(1)),
                        stddev = float(mtch.group(2)),
                        weight = float(mtch.group(3))
                    )
                )
        else:
            mixList.append(GaussianMixtureModel1d())
    
    if (mixList[-1].size() == 0):
        mixList = mixList[0:-1]
    
    return mixList


def csvParseMixtureModels(csvString):
    lineList = csvString.split("\n")
    lineIdx = 0
    line = lineList[lineIdx].strip()
    pgmmmHeaderLineRegEx = re.compile(".*pgmmm-mean.*,.*pgmmm-st.*dev.*,.*pgmmm-weight.*")
    gmmHeaderLineRegEx = re.compile(".*mean.*,.*st.*dev.*,.*weight.*")
    regExPairList = [(pgmmmHeaderLineRegEx, csvParsePgmmm), (gmmHeaderLineRegEx, csvParseGaussianMixtureModels)]
    matchIdx = None
    while ((matchIdx == None) and (lineIdx < len(lineList))):
        for pairIdx in range(len(regExPairList)):
            if ((matchIdx == None) and (regExPairList[pairIdx][0].match(line) != None)):
                matchIdx = pairIdx
        if (matchIdx == None):
            lineIdx += 1
            line = lineList[lineIdx].strip()
    if (lineIdx >= len(lineList)):
        raise RuntimeError("Could not find header line match for any reg-ex in %s " % str([p[0] for p in regExPairList]))
    
    
    mm = None
    
    if (regExPairList[matchIdx][1] != None):
        mm = regExPairList[matchIdx][1](csvString)
    else:
        raise ValueError("Function not available to parse mixture model with header:\n" + line)
    
    return mm


def calcGaussianMixtureModel1dTruncRange(mixModel, weightTruncPercent, numStdDev=2.4):
    mixModel = copy.deepcopy(mixModel)
    mixModel.sort(weightCompare)
    percent = 0.0
    minT = sys.float_info.max
    maxT = sys.float_info.min
    for i in range(0, mixModel.size()):
        dist = mixModel[i]
        percent += 100.0*dist.getWeight()
        mn = dist.getMean() - numStdDev*dist.getStandardDeviation()
        mx = dist.getMean() + numStdDev*dist.getStandardDeviation()
        if (mn < minT):
            minT = mn
        if (mx > maxT):
            maxT = mx
        if (percent > weightTruncPercent):
            break
    return (minT, maxT)
