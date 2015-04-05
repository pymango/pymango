#!/usr/bin/env python
import logging
import sys
import os
import os.path
import mango.unittest
import scipy as sp
import numpy as np
import mango.mpi as mpi
import mango.image
import mango.io
import mango.data

logger, rootLogger = mpi.getLoggers(__name__)

class HistogramddTest(mango.unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((51,31,33))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def testHistgramdd(self):
        imgDds = mango.data.gaussian_noise(self.imgShape, mtype="tomo", mean=32000, stdd=100)
        se = mango.image.sphere_se(radius=3)
        meanImg = mango.image.mean_filter(imgDds, se)
        stddImg = mango.image.stdd_filter(imgDds, se)
        h, edges = mango.image.histogramdd([meanImg, stddImg], bins=[256,128])
        self.assertEqual(2, len(h.shape))
        self.assertEqual(h.shape[0], 256)
        self.assertEqual(h.shape[1], 128)
        self.assertEqual(sp.product(imgDds.shape), sp.sum(h))
        
if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.core", "mango.io", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
