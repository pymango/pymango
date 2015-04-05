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

class CropTest(unittest.TestCase):
    def setUp(self):
        np.random.seed((mango.mpi.rank+1)*975421)
        subdShape = sp.array((16,64,32))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def getSteppedShape(self, shape, step):
        return sp.array([len(range(0, shape[i], step[i])) for i in range(len(shape))])
    
    def doTestCropWithHalo(self, haloSz=0):
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
        cropDds = \
            mango.image.crop(
                imgDds,
                offset=(0,0,0),
                shape = imgDds.shape
            )
        logger.info("imgDds.shape=%s" % imgDds.shape)
        logger.info("cropDds.shape=%s" % cropDds.shape)

        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], cropDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)
        
        self.assertEqual(imgDds.dtype, cropDds.dtype)
        self.assertEqual(imgDds.mtype, cropDds.mtype)
        self.assertTrue(sp.all(imgDds.halo  == cropDds.halo))
        self.assertTrue(sp.all(imgDds.origin == cropDds.origin), "%s != %s" % (imgDds.origin, cropDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == cropDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize() == cropDds.md.getVoxelSize()))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("cropDds min = %s, cropDds max = %s" % (np.min(cropDds.asarray()[slc]), np.max(cropDds.asarray()[slc])))
        logger.info("num non-zero cropDds = %s" % sp.sum(sp.where(cropDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray()[slc] == cropDds.asarray()[slc]))

        imgDds = \
            mango.data.gaussian_noise(
                mean=32000., stdd=2000.,
                shape=self.imgShape,
                dtype="uint16",
                halo=haloSz,
                origin=(2,-8,4)
            )
        imgDds.setBorderToValue(32000)
        imgDds.updateOverlapRegions()
        imgDds.md.setVoxelSize((1,1,1));
        imgDds.md.setVoxelSizeUnit("mm");

        cropDds = \
            mango.image.crop(
                imgDds,
                offset = imgDds.shape//4,
                shape = imgDds.shape//2
            )
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], cropDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)

        logger.info("imgDds.md.getVoxelSize()=%s%s" % (imgDds.md.getVoxelSize(), imgDds.md.getVoxelSizeUnit()))
        logger.info("cropDds.md.getVoxelSize()=%s%s" % (cropDds.md.getVoxelSize(), cropDds.md.getVoxelSizeUnit()))

        self.assertEqual(imgDds.dtype, cropDds.dtype)
        self.assertEqual(imgDds.mtype, cropDds.mtype)
        self.assertTrue(sp.all(imgDds.halo == cropDds.halo))
        self.assertTrue(sp.all(imgDds.shape//2 == cropDds.shape))
        self.assertTrue(sp.all(imgDds.origin+imgDds.shape//4 == cropDds.origin), "%s != %s" % (imgDds.origin, cropDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == cropDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize() == cropDds.md.getVoxelSize()))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("cropDds min = %s, cropDds max = %s" % (np.min(cropDds.asarray()[slc]), np.max(cropDds.asarray()[slc])))
        logger.info("num non-zero cropDds = %s" % sp.sum(sp.where(cropDds.asarray()[slc] != 0, 1, 0)))
        
        cropDds = \
            mango.image.crop(
                imgDds,
                offset=(3,5,7),
                shape=(imgDds.shape[0]-2, imgDds.shape[1]-8, imgDds.shape[2]-4)
            )
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], cropDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("cropDds min = %s, cropDds max = %s" % (np.min(cropDds.asarray()[slc]), np.max(cropDds.asarray()[slc])))
        logger.info("num non-zero cropDds = %s" % sp.sum(sp.where(cropDds.asarray()[slc] != 0, 1, 0)))

        self.assertEqual(imgDds.dtype, cropDds.dtype)
        self.assertEqual(imgDds.mtype, cropDds.mtype)
        self.assertTrue(sp.all(imgDds.halo == cropDds.halo))
        self.assertTrue(sp.all(imgDds.shape-(2,8,4) == cropDds.shape), "%s != %s" % (imgDds.shape//(3,5,7), cropDds.shape))
        self.assertTrue(sp.all(imgDds.origin+(3,5,7) == cropDds.origin), "%s != %s" % (imgDds.origin, cropDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == cropDds.mpi.shape))
        self.assertTrue(sp.all(imgDds.md.getVoxelSize() == cropDds.md.getVoxelSize()))

        rootLogger.info("*************************")

    def testCropWithHalo0(self):
        self.doTestCropWithHalo(0)

    def testCropWithHalo2(self):
        self.doTestCropWithHalo(4)

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    unittest.main()
