#!/usr/bin/env python
import logging
import sys
import unittest
import scipy as sp
import numpy as np
import mango.mpi as mpi
import mango.image
import mango.data
import mango.io

logger, rootLogger = mpi.getLoggers(__name__)

class SubSampleTest(unittest.TestCase):
    def setUp(self):
        np.random.seed((mango.mpi.rank+1)*975421)
        subdShape = sp.array((16,64,32))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def getSteppedShape(self, shape, step):
        return sp.array([len(range(0, shape[i], step[i])) for i in range(len(shape))])
    
    def doTestSubSampleWithHalo(self, haloSz=0):
        rootLogger.info("*************************")
        rootLogger.info("haloSz=%s" % haloSz)
        if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
            if (haloSz < 0):
                haloSz = 0
            haloSz = sp.array((haloSz,)*3)
        
        imgDds = mango.data.gaussian_noise(shape=self.imgShape, mtype="tomo_float", halo=haloSz)
        imgDds.setBorderToValue(0)
        imgDds.updateOverlapRegions()

        imgDds.md.setVoxelSize((1,1,1));
        imgDds.md.setVoxelSizeUnit("mm");
        
        logger.info("imgDds.mtype=%s" % imgDds.mtype)
        logger.info("imgDds.md.getVoxelSize()=%s" % imgDds.md.getVoxelSize())
        ssDds = \
            mango.image.subsample(
                imgDds,
                start=(0,0,0),
                stop = imgDds.shape,
                step = (1,1,1)
            )
        logger.info("imgDds.shape=%s" % imgDds.shape)
        logger.info("ssDds.shape=%s" % ssDds.shape)

        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], ssDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)
        
        self.assertEqual(imgDds.dtype, ssDds.dtype)
        self.assertTrue(sp.all(imgDds.mtype == ssDds.mtype), "%s != %s" % (imgDds.mtype, ssDds.mtype))
        self.assertTrue(sp.all(imgDds.halo  == ssDds.halo))
        self.assertTrue(sp.all(self.getSteppedShape(imgDds.shape, (1,1,1)) == ssDds.shape))
        self.assertTrue(sp.all(imgDds.origin == ssDds.origin), "%s != %s" % (imgDds.origin, ssDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == ssDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize() == ssDds.md.getVoxelSize()))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("ssDds min = %s, ssDds max = %s" % (np.min(ssDds.asarray()[slc]), np.max(ssDds.asarray()[slc])))
        logger.info("num non-zero ssDds = %s" % sp.sum(sp.where(ssDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray()[slc] == ssDds.asarray()[slc]))

        imgDds = \
            mango.data.gaussian_noise(
                mean=32000., stdd=2000.,
                shape=self.imgShape,
                mtype="tomo",
                halo=haloSz,
                origin=(2,-8,4)
            )
        imgDds.setBorderToValue(32000)
        imgDds.updateOverlapRegions()
        imgDds.md.setVoxelSize((1,1,1));
        imgDds.md.setVoxelSizeUnit("mm");

        ssDds = \
            mango.image.subsample(
                imgDds,
                step=(2,2,2)
            )
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], ssDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)

        logger.info("imgDds.md.getVoxelSize()=%s%s" % (imgDds.md.getVoxelSize(), imgDds.md.getVoxelSizeUnit()))
        logger.info("ssDds.md.getVoxelSize()=%s%s" % (ssDds.md.getVoxelSize(), ssDds.md.getVoxelSizeUnit()))

        self.assertEqual(imgDds.dtype, ssDds.dtype)
        self.assertEqual(imgDds.mtype, ssDds.mtype)
        self.assertTrue(sp.all(imgDds.halo == ssDds.halo))
        self.assertTrue(sp.all(self.getSteppedShape(imgDds.shape, (2,2,2)) == ssDds.shape))
        self.assertTrue(sp.all(imgDds.origin == ssDds.origin), "%s != %s" % (imgDds.origin, ssDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == ssDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize()*2 == ssDds.md.getVoxelSize()))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("ssDds min = %s, ssDds max = %s" % (np.min(ssDds.asarray()[slc]), np.max(ssDds.asarray()[slc])))
        logger.info("num non-zero ssDds = %s" % sp.sum(sp.where(ssDds.asarray()[slc] != 0, 1, 0)))
        
        ssDds = \
            mango.image.subsample(
                imgDds,
                step=(3,5,7)
            )
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], ssDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("ssDds min = %s, ssDds max = %s" % (np.min(ssDds.asarray()[slc]), np.max(ssDds.asarray()[slc])))
        logger.info("num non-zero ssDds = %s" % sp.sum(sp.where(ssDds.asarray()[slc] != 0, 1, 0)))

        self.assertEqual(imgDds.dtype, ssDds.dtype)
        self.assertEqual(imgDds.mtype, ssDds.mtype)
        self.assertTrue(sp.all(imgDds.halo == ssDds.halo))
        self.assertTrue(sp.all(self.getSteppedShape(imgDds.shape, (3,5,7)) == ssDds.shape), "%s != %s" % (imgDds.shape//(3,5,7), ssDds.shape))
        self.assertTrue(sp.all(imgDds.origin == ssDds.origin), "%s != %s" % (imgDds.origin, ssDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == ssDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize()*(3,5,7) == ssDds.md.getVoxelSize()))

        ssDds = \
            mango.image.subsample(
                imgDds,
                step=(5,7,3),
                start=(3,5,9)
            )
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], ssDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("ssDds min = %s, ssDds max = %s" % (np.min(ssDds.asarray()[slc]), np.max(ssDds.asarray()[slc])))
        logger.info("num non-zero ssDds = %s" % sp.sum(sp.where(ssDds.asarray()[slc] != 0, 1, 0)))

        self.assertEqual(imgDds.dtype, ssDds.dtype)
        self.assertEqual(imgDds.mtype, ssDds.mtype)
        self.assertTrue(sp.all(imgDds.halo == ssDds.halo))
        self.assertTrue(sp.all(self.getSteppedShape(imgDds.shape-(3,5,9), (5,7,3)) == ssDds.shape))
        self.assertTrue(sp.all(imgDds.origin+(3,5,9) == ssDds.origin), "%s != %s" % (imgDds.origin, ssDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == ssDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize()*(5,7,3) == ssDds.md.getVoxelSize()))

        ssDds = \
            mango.image.subsample(
                imgDds,
                step=(5,7,3),
                start=(3,5,9),
                stop=imgDds.shape-(3,6,5)
            )
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], ssDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("ssDds min = %s, ssDds max = %s" % (np.min(ssDds.asarray()[slc]), np.max(ssDds.asarray()[slc])))
        logger.info("num non-zero ssDds = %s" % sp.sum(sp.where(ssDds.asarray()[slc] != 0, 1, 0)))

        self.assertEqual(imgDds.dtype, ssDds.dtype)
        self.assertEqual(imgDds.mtype, ssDds.mtype)
        self.assertTrue(sp.all(imgDds.halo == ssDds.halo))
        self.assertTrue(sp.all(self.getSteppedShape(imgDds.shape-(3,5,9)-(3,6,5), (5,7,3)) == ssDds.shape))
        self.assertTrue(sp.all(imgDds.origin+(3,5,9) == ssDds.origin), "%s != %s" % (imgDds.origin, ssDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == ssDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize()*(5,7,3) == ssDds.md.getVoxelSize()))

        rootLogger.info("*************************")

    def testSubSampleWithHalo0(self):
        self.doTestSubSampleWithHalo(0)

    def testSubSampleWithHalo2(self):
        self.doTestSubSampleWithHalo(4)

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    unittest.main()
