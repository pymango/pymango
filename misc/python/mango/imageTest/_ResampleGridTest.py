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

class ResampleGridTest(unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((8,8,8))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def doTestResampleGridWithHalo(self, haloSz=0):
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
            mango.image.resample(
                imgDds,
                interptype=mango.image.InterpolationType.LINEAR,
                voxsz=0.5*sp.array(imgDds.md.getVoxelSize())
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
        self.assertTrue(sp.all((2*(imgDds.shape)) == rspDds.shape))
        self.assertTrue(sp.all(2*imgDds.origin == rspDds.origin), "%s != %s" % (imgDds.origin, rspDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == rspDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize() == rspDds.md.getVoxelSize()*2))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("rspDds min = %s, rspDds max = %s" % (np.min(rspDds.asarray()[slc]), np.max(rspDds.asarray()[slc])))
        logger.info("num non-zero rspDds = %s" % sp.sum(sp.where(rspDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray() == 0))
        self.assertTrue(sp.all(rspDds.asarray()[slc] == 0))

        imgDds = mango.zeros(shape=self.imgShape, mtype="tomo", halo=haloSz, origin=(2,-8,4))
        imgDds.md.setVoxelSize((1,1,1));
        imgDds.md.setVoxelSizeUnit("mm");

        rspDds = \
            mango.image.resample(
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
        self.assertTrue(sp.all(imgDds.origin//2 == rspDds.origin), "%s != %s" % (imgDds.origin, rspDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == rspDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize() == rspDds.md.getVoxelSize()/2))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("rspDds min = %s, rspDds max = %s" % (np.min(rspDds.asarray()[slc]), np.max(rspDds.asarray()[slc])))
        logger.info("num non-zero rspDds = %s" % sp.sum(sp.where(rspDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray() == 0))
        self.assertTrue(sp.all(rspDds.asarray()[slc] == 0))

    def testResampleGridWithHalo0(self):
        self.doTestResampleGridWithHalo(0)

    def testResampleGridWithHalo1(self):
        self.doTestResampleGridWithHalo(2)

    def testResampleGridWithHalo2(self):
        self.doTestResampleGridWithHalo(4)

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    unittest.main()
