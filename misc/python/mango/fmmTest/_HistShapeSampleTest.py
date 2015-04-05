#!/usr/bin/env python
import logging
import sys
import unittest
import mango.unittest
import scipy as sp
import numpy as np
import numpy.random
import random
import mango.mpi as mpi
import mango.fmm
import mango.data
import mango.io
import os, os.path

logger, rootLogger = mpi.getLoggers(__name__)

class HistShapeSampleTest(mango.unittest.TestCase):

    def setUp(self):
        self.imgShape = sp.array((160,163,155), dtype="int32")
        self.imgShape += 0.1*(mpi.size-1)
        self.imgShape = self.imgShape.tolist()
        
    def testCylinderSample(self):
        dir = self.createTmpDir(os.path.join("HistShapeSampleTest", "testCylinderSample"))
        dir = "."
        mtype = "tomo_float"
        mean = 0.0
        phaseMeans = [4000.0, 16000.0, 25000.0]
        stdd = 4000.0
 
        noisDds = \
            mango.data.gaussian_noise(shape=self.imgShape, mean=mean, stdd=stdd, mtype=mtype)

        imgDds = mango.zeros_like(noisDds)
        imgDds.setAllToValue(phaseMeans[0])
        mango.data.fill_circular_cylinder(
            input=imgDds,
            centre=imgDds.shape*0.5,
            radius=np.min(imgDds.shape)*0.50,
            axislen = imgDds.shape[0]+2,
            fill = phaseMeans[1]
        )
        mango.data.fill_circular_cylinder(
            input=imgDds,
            centre=imgDds.shape*0.5,
            radius=np.min(imgDds.shape)*0.40,
            axislen = imgDds.shape[0]+2,
            fill = phaseMeans[2]
        )

        imgDds.subd.asarray()[...] += noisDds.subd.asarray()
        se = mango.image.sphere_se(2.7)
        mnDds = mango.image.mean_filter(imgDds, se)
        sdDds = mango.image.stdd_filter(imgDds, se)
        
        H, (bins0, bins1) = mango.image.histogramdd([mnDds, sdDds], bins=[1024,4096])
        peakMtx = mango.fmm.calc_mean_stdd_peaks(H, bins0, bins1, 3)
        peakMtx = np.sort(peakMtx, axis=0)
        mango.io.writeDds(os.path.join(dir, "tomo_floatCyls.nc"), imgDds)
        rootLogger.info("peakMtx =\n%s" % (peakMtx,))
        rootLogger.info("binSz = %s" % (np.max(bins1[1:]-bins1[0:-1]),))
        rootLogger.info("num se elements = %s" % (se.getIndices().shape[0],))
        phaseMeans.sort()
        for i in range(0, peakMtx.shape[0]):
            self.assertTrue(np.abs(peakMtx[i,0]-phaseMeans[i])/phaseMeans[i] < 5.0e-2)
            self.assertTrue(np.abs(peakMtx[i,1]-stdd)/stdd < 5.0e-2)

        H01, (bins0,) = \
            mango.fmm.sample_histogram_circular_cylinder(
                input0 = mnDds,
                input1 = None,
                centre=imgDds.shape*0.5,
                radius=np.min(imgDds.shape)*0.40,
                axislen = imgDds.shape[0]+2,
                bins=(bins0,)
            )
        H02, (bins0,) = \
            mango.fmm.sample_histogram_annular_circular_cylinder(
                input0 = mnDds,
                input1 = None,
                centre=imgDds.shape*0.5,
                inner_radius=np.min(imgDds.shape)*0.40,
                annular_width=np.min(imgDds.shape)*0.1,
                axislen = imgDds.shape[0]+2,
                bins=(bins0,)
            )

        H2, (bins0, bins1) = \
            mango.fmm.sample_histogram_circular_cylinder(
                input0 = mnDds,
                input1 = sdDds,
                centre=imgDds.shape*0.5,
                radius=np.min(imgDds.shape)*0.40,
                axislen = imgDds.shape[0]+2,
                bins=(bins0, bins1)
            )

        if (mpi.rank == 0):
            import matplotlib
            import matplotlib.pyplot as plt

            y = (bins0[1:]+bins0[0:-1])*0.5
            x = (bins1[1:]+bins1[0:-1])*0.5
            plt.plot(y, sp.sum(H, axis=1), linewidth=7)
            plt.plot(y, H01, linewidth=3)
            plt.plot(y, H02, linewidth=3)
            #plt.show()


if __name__ == "__main__":
    #mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.unittest", "mango.mpi", "mango.fmm", "mango.fmmTest"],
        logLevel=logging.DEBUG
    )
    random.seed((mpi.rank+1)*23456243)
    numpy.random.seed((mpi.rank+1)*23456134)

    mango.unittest.main()
