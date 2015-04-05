#!/usr/bin/env python
import logging
import sys
import unittest
import mango.unittest
import scipy as sp
import scipy.stats
import numpy as np
import numpy.random
import random
import mango
import mango.mpi as mpi
import mango.fmm
import mango.image
import mango.data
import mango.io
import os, os.path

logger, rootLogger = mpi.getLoggers(__name__)

class GenChiSquaredTest(mango.unittest.TestCase):
    
    def testCdfPlt(self):
        dir = self.createTmpDir("testCdfPlt")
        if ((mpi.world == None) or (mpi.world.Get_rank() == 0)):
            import matplotlib
            import matplotlib.pyplot as plt
            for sigma in (0.1,0.2,0.4,0.5,1.0):
                kernel = mango.image.discrete_gaussian_kernel(sigma)
                coeffs = kernel.flatten()
                locals = sp.zeros((len(coeffs),), dtype="float64")
                gcsd = mango.fmm.gchisqrd(coeffs, locals)
                rootLogger.info("np.unique(kernel):\n%s" % (np.unique(kernel).flatten().tolist(),))
                rootLogger.info("gcsd.coefficients:\n%s" % (gcsd.coefficients.tolist(),))
                rootLogger.info("gcsd.noncentralities:\n%s" % (gcsd.noncentralities.tolist(),))
                rootLogger.info("gcsd.multiplicities:\n%s" % (gcsd.multiplicities.tolist(),))
    
                xxCdf = sp.linspace(0, 4, 401)
                rootLogger.info("Calculating gchi2, coeffs=%s..." % (tuple(coeffs.tolist()),))
                gcsdcdf = gcsd.cdf(xxCdf)
                yy = gcsdcdf[1:]-gcsdcdf[0:-1]
                rootLogger.info("Done calculating gchi2.")
                xx = 0.5*(xxCdf[1:] + xxCdf[0:-1])
    
                plt.plot(xx, yy)
            plt.savefig(os.path.join(dir, "gchi2cdf.png"))

    def testCdf(self):
        coeffs = sp.ones((8,), dtype="float64")
        locals = sp.zeros((8,), dtype="float64")
        gcsd = mango.fmm.gchisqrd(coeffs, locals)
        csd = sp.stats.chi2(len(coeffs))
        for v in (0.001, 1, 2, 4, 8, 16, 32, 64):
            rootLogger.info("gcsd.cdf(%8.3f)=%14.12f" % (v, gcsd.cdf(v)))
            rootLogger.info(" csd.cdf(%8.3f)=%14.12f" % (v,  csd.cdf(v)))

        coeffs = sp.ones((16,), dtype="float64")
        locals = sp.array(range(1,len(coeffs+1)), dtype="float64")
        locals[...] = locals*locals
        gcsd = mango.fmm.gchisqrd(coeffs, locals)
        ncx2 = sp.stats.ncx2(len(coeffs), sp.sum(locals))
        for v in (1000, 1100, 1200, 1350,1400, 1500, 1600):
            rootLogger.info("gcsd.cdf(%8.3f)=%14.12f" % (v, gcsd.cdf(v)))
            rootLogger.info("ncx2.cdf(%8.3f)=%14.12f" % (v, ncx2.cdf(v)))

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.unittest", "mango.mpi", "mango.fmm", "mango.fmmTest"],
        logLevel=logging.DEBUG
    )
    random.seed((mpi.rank+1)*23456243)
    numpy.random.seed((mpi.rank+1)*23456134)

    unittest.main()
