#!/usr/bin/env python
import logging
import sys
import unittest
import scipy as sp
import numpy as np
import mango.mpi as mpi
import mango.image
import mango.io

logger, rootLogger = mpi.getLoggers(__name__)

class SobelTest(unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((64,64,64))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def doTestSobelWithHalo(self, haloSz=0):
        if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
            if (haloSz < 0):
                haloSz = 0
            haloSz = sp.array((haloSz,)*3)
        
        imgDds = mango.zeros(shape=self.imgShape, mtype="tomo_float", halo=haloSz)
        
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], imgDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)
        
        sobelDds = mango.image.sobel(imgDds)
        
        self.assertTrue(sp.all(imgDds.dtype == sobelDds.dtype))
        self.assertTrue(sp.all(imgDds.mtype == sobelDds.mtype), "%s != %s" % (imgDds.mtype, sobelDds.mtype))
        self.assertTrue(sp.all(imgDds.halo == sobelDds.halo))
        self.assertTrue(sp.all(imgDds.shape == sobelDds.shape))
        self.assertTrue(sp.all(imgDds.origin == sobelDds.origin), "%s != %s" % (imgDds.origin, sobelDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == sobelDds.mpi.shape))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("sobelDds min = %s, sobelDds max = %s" % (np.min(sobelDds.asarray()[slc]), np.max(sobelDds.asarray()[slc])))
        logger.info("num non-zero sobelDds = %s" % sp.sum(sp.where(sobelDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray() == 0))
        self.assertTrue(sp.all(sobelDds.asarray()[slc] == 0))

        imgDds = mango.zeros(shape=self.imgShape, dtype="float64", halo=haloSz)
        sobelDds = mango.image.sobel(imgDds)

        self.assertTrue(sp.all(imgDds.dtype == sobelDds.dtype))
        self.assertTrue(sp.all(imgDds.mtype == sobelDds.mtype))
        self.assertTrue(sp.all(imgDds.halo == sobelDds.halo))
        self.assertTrue(sp.all(imgDds.shape == sobelDds.shape))
        self.assertTrue(sp.all(imgDds.origin == sobelDds.origin), "%s != %s" % (imgDds.origin, sobelDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == sobelDds.mpi.shape))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("sobelDds min = %s, sobelDds max = %s" % (np.min(sobelDds.asarray()[slc]), np.max(sobelDds.asarray()[slc])))
        logger.info("num non-zero sobelDds = %s" % sp.sum(sp.where(sobelDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray() == 0))
        self.assertTrue(sp.all(sobelDds.asarray()[slc] == 0))

    def testSobelWithHalo0(self):
        self.doTestSobelWithHalo(0)

    def testSobelWithHalo1(self):
        self.doTestSobelWithHalo(1)

    def testSobelWithHalo2(self):
        self.doTestSobelWithHalo(2)

if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    unittest.main()
