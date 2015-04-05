
import mango.utils
import scipy.stats
import math
from mango import mpi

logger, rootLogger = mpi.getLoggers(__name__)

class WeightedDistribution:
    def __init__(self, weight=1.0):
        self.weight = weight
        
    def getWeight(self):
        return self.weight
    
    def setWeight(self, weight):
        self.weight = weight

    def wpdf(self, x):
        return self.weight * self.pdf(x)
    
    def wcdf(self, x):
        return self.weight * self.cdf(x)

class Gaussian1d:
    def __init__(self, mean=0.0, stddev=1.0):
        self.mean = mean
        self.stddev = stddev
    
    def getMean(self):
        return self.mean

    def setMean(self, mean):
        self.mean = mean

    def getStandardDeviation(self):
        return self.stddev

    def setStandardDeviation(self, stddev):
        self.stddev = stddev

    def getVariance(self):
        return self.stddev * self.stddev

    def setVariance(self, variance):
        self.stddev = math.sqrt(variance)

    def pdf(self, x):
        return scipy.stats.norm.pdf(x, loc=self.mean, scale=self.stddev)
    
    def cdf(self, x):
        return scipy.stats.norm.cdf(x, loc=self.mean, scale=self.stddev)
    
    def rvs(self, size=1):
        return scipy.stats.norm.rvs(loc=self.getMean(), scale=self.getStandardDeviation(), size=size)

class Normal1d(Gaussian1d):
    pass

class WeightedGaussian1d (Gaussian1d, WeightedDistribution):
    def __init__(self, mean=0.0, stddev=1.0, weight=1.0):
        Gaussian1d.__init__(self, mean, stddev)
        WeightedDistribution.__init__(self, weight)

class WeightedNormal1d(WeightedGaussian1d):
    pass
    
def gaussian1dCompare(g0,g1):
    cmp = -1
    if (g0.getMean() > g1.getMean()):
        cmp = 1
    elif (g0.getMean() == g1.getMean()):
        if (g0.getStandardDeviation() > g1.getStandardDeviation()):
            cmp = 1
        elif (g0.getStandardDeviation() == g1.getStandardDeviation()):
            cmp = 0
    return cmp

def weightCompare(g0,g1):
    cmp = 1
    if (g0.getWeight() > g1.getWeight()):
        cmp = -1
    elif (g0.getWeight() == g1.getWeight()):
        cmp = 0
    return cmp
