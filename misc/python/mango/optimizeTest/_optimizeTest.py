#!/usr/bin/env python
import logging
import unittest
import random
import scipy as sp
import numpy as np
import scipy.optimize
import mango.mpi as mpi
import mango.optimize
from mango.image import *

logger, rootLogger = mpi.getLoggers(__name__)

class CosineExp:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return -sp.exp(-(x**2)/(8*sp.pi*sp.pi))*sp.cos(x)

class CosineExpOptimizer(mango.optimize.SingleStartOptimizer):
    def __init__(self):
        pass
    
    def optimize(self, x0):
        result = mango.optimize.OptimizeResult()
        res = sp.optimize.minimize(CosineExp(), x0, method="Powell")
        result.cost             = res.fun
        result.optim            = res.x
        result.start            = x0
        result.numIterations    = res.nit
        result.numFuncEvals     = res.nfev

        return result

class MultiStartOptimizerTest(unittest.TestCase):
    def testIt(self):
        ssOptimizer = CosineExpOptimizer()
        msOptimizer = mango.optimize.MultiStartOptimizer(ssOptimizer)
        exez = np.linspace(-2*sp.pi, 2*sp.pi, 40)
        
        rList = msOptimizer.optimize(exez)
        
        rootLogger.info("MultiStartOptimizer results:\n%s" % "\n".join(map(str, rList)))
        self.assertGreaterEqual(1.0e-6, rList[0].optim)


if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.optimize", "mango.optimizeTest"],
        logLevel=logging.INFO
    )
    unittest.main()
