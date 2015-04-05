import scipy as sp
import scipy.stats
import scipy.optimize
import numpy as np
import math
import mango
import mango.image
import mango.mpi as mpi
import copy
from mango.math import *

logger, rootLogger = mpi.getLoggers(__name__)

class SphHistPearsonrMetric:
    def __init__(self, fixdSphHist, trnsSphHist, loPercentile=0.0, hiPercentile=100.0):
        # make deep copies of the spherical histograms
        rootLogger.debug("%s, %s, %s" % (fixdSphHist.getNumBins(), fixdSphHist.getBinCounts().size, fixdSphHist.getBinAreas().size, ))
        self.fixdSphHist = copy.deepcopy(fixdSphHist)
        rootLogger.debug("%s, %s, %s" % (self.fixdSphHist.getNumBins(), self.fixdSphHist.getBinCounts().size, self.fixdSphHist.getBinAreas().size, ))
        self.trnsSphHist = copy.deepcopy(trnsSphHist)
        self.loPercentile = loPercentile
        self.hiPercentile = hiPercentile

        self.fixdSphHist.sphere_c = (0,0,0)
        rootLogger.debug("%s, %s, %s" % (self.fixdSphHist.getNumBins(), self.fixdSphHist.getBinCounts().size, self.fixdSphHist.getBinAreas().size, ))
        self.fixdSphHist.setBinCounts(self.fixdSphHist.getBinCounts()/self.fixdSphHist.getBinAreas())
        self.trnsSphHist.sphere_c = (0,0,0)
        self.trnsSphHist.setBinCounts(self.trnsSphHist.getBinCounts()/self.trnsSphHist.getBinAreas())

        self.trnsDensity = self.trnsSphHist.getBinCounts()
        self.trnsCoords  = self.trnsSphHist.getBinCentres()
        
        loThrsh, hiThrsh = np.percentile(self.trnsDensity, (self.loPercentile, self.hiPercentile))
        msk = sp.where(sp.logical_and(self.trnsDensity > loThrsh, self.trnsDensity <= hiThrsh))
        self.trnsDensity = self.trnsDensity[msk]
        self.trnsCoords  = self.trnsCoords[msk]
        
        self.fixdLoThrsh, self.fixdHiThrsh = np.percentile(self.fixdSphHist.getBinCounts(), (self.loPercentile, self.hiPercentile))
        
    def evaluate(self, rotMatrix):
        """
        Evaluate the correlation for the given orientation (rotation).
        :rtype: :obj:`float`
        :return: Correlation of normalised bin counts.
        """
        trnsCoords  = rotMatrix.dot(self.trnsCoords.T).T
        fixdDensity = self.fixdSphHist.getBinCounts(trnsCoords)
        msk = sp.where(sp.logical_and(fixdDensity > self.fixdLoThrsh, fixdDensity <= self.fixdHiThrsh))
        
        return sp.stats.pearsonr(self.trnsDensity[msk], fixdDensity[msk])[0]
    
    def __call__(self, rMatrix):
        return -self.evaluate(rMatrix)

class SpOptimizeMetric:
    def __init__(self, sphHistMetric):
        self.sphHistMetric = sphHistMetric
    
    def __call__(self, prms):
        axis = prms[0:3]
        angle = prms[3]
        return self.sphHistMetric(axis_angle_to_rotation_matrix(axis, angle))


class SphericalHistRegisterer:
    def __init__(self, metric):
        self.metric = metric
        self.numSearchKeepers = 10
        self.numExhaustiveSearchOrientations = 45
    
    def doExhaustiveSearch(self, comm = mango.mpi.world):
        tCoord = self.metric.trnsSphHist.getBinCentres()[0]
        fCoords = self.metric.fixdSphHist.getBinCentres()
        
        rnk = 0
        sz = 1
        if (comm != None):
            rnk = comm.Get_rank()
            sz = comm.Get_size()
        numfCoords = fCoords.shape[0]
        bestList = []
        orientAngles = np.linspace(-np.pi, np.pi, self.numExhaustiveSearchOrientations)
        for fCoordIdx in range(rnk, numfCoords, sz):
            fCoord = fCoords[fCoordIdx]
            axisAngle = rotation_axis_and_angle(tCoord, fCoord)
            rMatrix0 = axis_angle_to_rotation_matrix(axisAngle[0], axisAngle[1])
            for angle in orientAngles:
                rMatrix1 = axis_angle_to_rotation_matrix(axisAngle[0], axisAngle[1])
                rMatrix  = rMatrix1.dot(rMatrix0)
                metricVal = self.metric(rMatrix)
                bestList.append([metricVal, axis_angle_from_rotation_matrix(rMatrix), rMatrix])
                bestList = sorted(bestList, key= lambda p: p[0])
                if (len(bestList) >= self.numSearchKeepers):
                    bestList = bestList[0:self.numSearchKeepers]
        
        if (comm != None):
            tupleListList = comm.allgather(bestList)
            bestList = []
            for tupList in tupleListList:
                bestList += tupList
        bestList = sorted(bestList, key= lambda p: p[0])[0:self.numSearchKeepers]
        
        return bestList

    def doRefinmentOptimization(self, bestList, comm = mango.mpi.world):
        rnk = 0
        sz = 1
        if (comm != None):
            rnk = comm.Get_rank()
            sz = comm.Get_size()

        spOptMetric = SpOptimizeMetric(self.metric)
        for tup in bestList[rnk:len(bestList):sz]:
            logger.debug("Pre  minimize tup = %s" % (tup[0:2],))
            x0 = sp.zeros((4,), dtype="float64")
            x0[0:3] = tup[1][0]
            x0[3] = tup[1][1]
            res = sp.optimize.minimize(spOptMetric, x0, method="Powell", options={'xtol':1.0e-4, 'ftol':1.0e-8})
            tup[0] = res.fun
            tup[1] = [res.x[0:3], res.x[3]]
            tup[2] = axis_angle_to_rotation_matrix(tup[1][0], tup[1][1])
            logger.debug("Post minimize tup = %s" % (tup[0:2],))
        
        if (comm != None):
            tupleListList = comm.allgather(bestList)
            bestList = []
            for tupList in tupleListList:
                bestList += tupList

        bestList = sorted(bestList, key= lambda p: p[0])
        
        return bestList
    
    def search(self, comm = mango.mpi.world):
        bestList = self.doExhaustiveSearch(comm)
        bestList = self.doRefinmentOptimization(bestList, comm)
        
        bestRotMatrix, bestMetricVal, bestAxisAngle = bestList[0][2],  bestList[0][0], bestList[0][1]
        return bestRotMatrix, bestMetricVal

