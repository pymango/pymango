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

class GatherSliceTest(unittest.TestCase):
    def setUp(self):
        np.random.seed((mango.mpi.rank+1)*975421)
        subdShape = sp.array((16,64,32))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def getSteppedShape(self, shape, step):
        return sp.array([len(range(0, shape[i], step[i])) for i in range(len(shape))])
    
    def doTestGatherSliceWithHalo(self, haloSz=0):
        if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
            if (haloSz < 0):
                haloSz = 0
            haloSz = sp.array((haloSz,)*3)
        
        
        for axis in range(0,3):
            rootLogger.info("*************************")
            rootLogger.info("axis=%s" % axis)

            imgDds = mango.zeros(shape=self.imgShape, origin=(-16,-12,18),dtype="int32", halo=haloSz)
            imgDds.setBorderToValue(0)
            imgDds.updateOverlapRegions()
    
            imgDds.md.setVoxelSize((1,1,1));
            imgDds.md.setVoxelSizeUnit("mm");
            logger.info("imgDds.mtype=%s" % imgDds.mtype)
            logger.info("imgDds.md.getVoxelSize()=%s" % imgDds.md.getVoxelSize())

            pyslc = [slice(None), slice(None), slice(None)]
            for i in range(0, imgDds.subd.shape[axis]):
                pyslc[axis] = slice(imgDds.subd.halo[axis]+i, imgDds.subd.halo[axis]+i+1)
                imgDds.asarray()[tuple(pyslc)] = imgDds.subd.origin[axis] + i

            for rank in [None, 0]:
                rootLogger.info("========================")
                rootLogger.info("rank=%s" % rank)

                slcRelIndex = imgDds.shape[axis]//2
                slcGlbIndex = imgDds.origin[axis] + slcRelIndex
                slcDds = \
                    mango.image.gather_slice(
                        imgDds,
                        axis=axis,
                        index=slcRelIndex,
                        rank = rank
                    )
                logger.info("slcGlbIndex=%s" % slcGlbIndex)
                logger.info("slcRelIndex=%s" % slcRelIndex)
                logger.info("imgDds.shape=%s" % imgDds.shape)
                logger.info("slcDds.shape=%s" % slcDds.shape)
                logger.info("imgDds.origin=%s" % imgDds.origin)
                logger.info("slcDds.origin=%s" % slcDds.origin)
        
                if ((not mpi.haveMpi4py) or (rank == None) or (rank == mango.mpi.world.Get_rank())):
                    self.assertEqual(imgDds.dtype, slcDds.dtype)
                    self.assertTrue(imgDds.mtype == slcDds.mtype, "%s != %s" % (imgDds.mtype, slcDds.mtype))
                    self.assertTrue(sp.all(imgDds.halo  == slcDds.halo))
                    self.assertEqual(1, slcDds.shape[axis])
                    self.assertEqual(imgDds.shape[(axis+1)%3], slcDds.shape[(axis+1)%3])
                    self.assertEqual(imgDds.shape[(axis+2)%3], slcDds.shape[(axis+2)%3])
                    self.assertEqual(slcGlbIndex, slcDds.origin[axis])
                    self.assertEqual(imgDds.origin[(axis+1)%3], slcDds.origin[(axis+1)%3])
                    self.assertEqual(imgDds.origin[(axis+2)%3], slcDds.origin[(axis+2)%3])
        
                    self.assertEqual(1, slcDds.mpi.shape[axis])
                    self.assertTrue(sp.all(imgDds.md.getVoxelSize() == slcDds.md.getVoxelSize()))
                    self.assertEqual(imgDds.md.getVoxelSizeUnit(), slcDds.md.getVoxelSizeUnit())
    
                    # Calculate the sp.array slicing for the non-halo (exclusive)
                    # part of the subdomain.
                    slc = []
                    for d in range(len(haloSz)):
                        slc.append(slice(haloSz[d], slcDds.asarray().shape[d]-haloSz[d]))
                    
                    slc = tuple(slc)
        
                    self.assertEqual(np.min(slcDds.asarray()[slc]), slcGlbIndex)
                    self.assertEqual(np.max(slcDds.asarray()[slc]), slcGlbIndex)
                    self.assertTrue(sp.all(slcDds.asarray()[slc] == slcGlbIndex))
        

    def testGatherSliceWithHalo0(self):
        self.doTestGatherSliceWithHalo(0)

    def testGatherSliceWithHalo2(self):
        self.doTestGatherSliceWithHalo(4)

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    unittest.main()
