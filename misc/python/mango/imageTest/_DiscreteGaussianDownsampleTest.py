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

class DiscreteGaussianDownsampleTest(unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((32,128,64))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def doTestDiscreteGaussianDownsampleWithHalo(self, haloSz=0):
        if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
            if (haloSz < 0):
                haloSz = 0
            haloSz = sp.array((haloSz,)*3)
        
        imgDds = mango.zeros(shape=self.imgShape, mtype="tomo_float", halo=haloSz)
        imgDds.md.setVoxelSize((1,1,1));
        imgDds.md.setVoxelSizeUnit("mm");
        
        logger.info("imgDds.mtype=%s" % imgDds.mtype)
        logger.info("imgDds.md.getVoxelSize()=%s" % imgDds.md.getVoxelSize())
        rspDds = \
            mango.image.gaussian_downsample(
                imgDds,
                interptype=mango.image.InterpolationType.LINEAR,
                voxsz=4.0*sp.array(imgDds.md.getVoxelSize())
            )
        logger.info("imgDds.shape=%s" % imgDds.shape)
        logger.info("rspDds.shape=%s" % rspDds.shape)

        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], rspDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)
        
        self.assertTrue(sp.all(imgDds.dtype == rspDds.dtype))
        self.assertTrue(sp.all(imgDds.mtype == rspDds.mtype), "%s != %s" % (imgDds.mtype, rspDds.mtype))
        self.assertTrue(sp.all(imgDds.halo == rspDds.halo))
        self.assertTrue(sp.all(((imgDds.shape)//4) == rspDds.shape))
        self.assertTrue(sp.all(imgDds.origin//4 == rspDds.origin), "%s != %s" % (imgDds.origin, rspDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == rspDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize()*4 == rspDds.md.getVoxelSize()))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("rspDds min = %s, rspDds max = %s" % (np.min(rspDds.asarray()[slc]), np.max(rspDds.asarray()[slc])))
        logger.info("num non-zero rspDds = %s" % sp.sum(sp.where(rspDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray() == 0))
        self.assertTrue(sp.all(rspDds.asarray()[slc] == 0))

        imgDds = mango.zeros(shape=self.imgShape, mtype="tomo", halo=haloSz, origin=(2,-8,4))
        imgDds.md.setVoxelSize((1,1,1));
        imgDds.md.setVoxelSizeUnit("mm");

        rspDds = \
            mango.image.gaussian_downsample(
                imgDds,
                interptype=mango.image.InterpolationType.CATMULL_ROM_CUBIC_SPLINE,
                factor=(0.5,0.5,0.5)
            )
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], rspDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)

        self.assertTrue(sp.all(imgDds.dtype == rspDds.dtype))
        self.assertTrue(sp.all(imgDds.mtype == rspDds.mtype))
        self.assertTrue(sp.all(imgDds.halo == rspDds.halo))
        self.assertTrue(sp.all(imgDds.shape//2 == rspDds.shape))
        self.assertTrue(sp.all(imgDds.origin//2 == rspDds.origin), "%s != %s" % (imgDds.origin//2, rspDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == rspDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize() == rspDds.md.getVoxelSize()/2))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("rspDds min = %s, rspDds max = %s" % (np.min(rspDds.asarray()[slc]), np.max(rspDds.asarray()[slc])))
        logger.info("num non-zero rspDds = %s" % sp.sum(sp.where(rspDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray() == 0))
        self.assertTrue(sp.all(rspDds.asarray()[slc] == 0))

    def testDiscreteGaussianDownsampleWithHalo0(self):
        self.doTestDiscreteGaussianDownsampleWithHalo(0)

    def testDiscreteGaussianDownsampleWithHalo1(self):
        self.doTestDiscreteGaussianDownsampleWithHalo(2)

    def testDiscreteGaussianDownsampleWithHalo2(self):
        self.doTestDiscreteGaussianDownsampleWithHalo(4)

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    unittest.main()
