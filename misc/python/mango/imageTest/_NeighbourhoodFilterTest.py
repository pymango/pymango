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


class NeighbourhoodFilterTest(unittest.TestCase):
    def setUp(self):
        self.mode = "mirror"
        self.imgShape = (120, 65, 67)
        self.diffTol = 0
    
    def tearDown(self):
        pass

    def doMangoFiltering(self, input, se, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0)):
        output = self.callMangoFilter(input, se, stride=stride, boffset=boffset, eoffset=eoffset)
        output.updateHaloRegions()
        output.mirrorOuterLayersToBorder(False)
        output.updateHaloRegions()
        return output
    
    def doScipyFiltering(self, input, se):
        inDds = mango.copy(input, halo=se.getHaloSize())
        inDds.updateHaloRegions()
        inDds.mirrorOuterLayersToBorder(False)
        inDds.updateHaloRegions()
        outputArr = self.callScipyFilter(inDds, se)
        output = mango.empty_like(inDds)
        output.asarray()[...] = outputArr
        output = mango.copy(output, halo=input.halo)
        output.updateHaloRegions()
        output.mirrorOuterLayersToBorder(False)
        output.updateHaloRegions()
       
        return output

    def doFilterTestWithHalo(self, haloSz=0):
        if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
            if (haloSz < 0):
                haloSz = 0
            haloSz = sp.array((haloSz,)*3)
        
        imgDds = mango.data.gaussian_noise(shape=self.imgShape, mean=32000, stdd=4000.0, mtype="tomo", halo=haloSz)
        
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], imgDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)
        
        se = mango.image.sphere_se(radius=1.0)
        mfDds = self.doMangoFiltering(imgDds, se)
        sfDds = self.doScipyFiltering(imgDds, se)
        
        self.assertTrue(sp.all(imgDds.dtype == mfDds.dtype))
        self.assertTrue(sp.all(imgDds.mtype == mfDds.mtype), "%s != %s" % (imgDds.mtype, mfDds.mtype))
        self.assertTrue(sp.all(imgDds.halo == mfDds.halo))
        self.assertTrue(sp.all(imgDds.shape == mfDds.shape))
        self.assertTrue(sp.all(imgDds.origin == mfDds.origin), "%s != %s" % (imgDds.origin, mfDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == mfDds.mpi.shape))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("mfDds min = %s, mfDds max = %s" % (np.min(mfDds.asarray()[slc]), np.max(mfDds.asarray()[slc])))
        logger.info("sfDds min = %s, sfDds max = %s" % (np.min(sfDds.asarray()[slc]), np.max(sfDds.asarray()[slc])))
        logger.info("num non-zero mfDds-sfDds = %s" % sp.sum(sp.where((mfDds.asarray()[slc]-sfDds.asarray()[slc]) != 0, 1, 0)))
        logger.info(
            "abs(sfDds-mfDds) min = %s, abs(sfDds-mfDds) max = %s"
            %
            (
                np.min(sp.absolute(mfDds.asarray()[slc]-sfDds.asarray()[slc])),
                np.max(sp.absolute(mfDds.asarray()[slc]-sfDds.asarray()[slc]))
            )
        )
        
        tmpDds = mango.copy(mfDds, dtype="int32")
        tmpDds.updateHaloRegions()
        tmpDds.mirrorOuterLayersToBorder(False)
        self.assertLessEqual(
            np.max(sp.absolute(tmpDds.asarray()[slc] - sfDds.asarray()[slc])),
            self.diffTol
        )
        self.assertLessEqual(
            np.max(sp.absolute(tmpDds.asarray() - sfDds.asarray())),
            self.diffTol
        )

        imgDds = mango.data.gaussian_noise(shape=self.imgShape, mean=5000, stdd=4500, dtype="int32", halo=haloSz, origin=(-32,8,64))
        se = mango.image.sphere_se(radius=3.0)
        mfDds = self.doMangoFiltering(imgDds, se)
        sfDds = self.doScipyFiltering(imgDds, se)

        self.assertTrue(sp.all(imgDds.dtype == mfDds.dtype))
        #self.assertTrue(sp.all(imgDds.mtype == mfDds.mtype))
        self.assertTrue(sp.all(imgDds.halo == mfDds.halo))
        self.assertTrue(sp.all(imgDds.shape == mfDds.shape))
        self.assertTrue(sp.all(imgDds.origin == mfDds.origin), "%s != %s" % (imgDds.origin, mfDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == mfDds.mpi.shape))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("mfDds min = %s, mfDds max = %s" % (np.min(mfDds.asarray()[slc]), np.max(mfDds.asarray()[slc])))
        logger.info("sfDds min = %s, sfDds max = %s" % (np.min(sfDds.asarray()[slc]), np.max(sfDds.asarray()[slc])))
        logger.info("num non-zero mfDds-sfDds = %s" % sp.sum(sp.where((mfDds.asarray()[slc]-sfDds.asarray()[slc]) != 0, 1, 0)))
        logger.info(
            "abs(sfDds-mfDds) min = %s, abs(sfDds-mfDds) max = %s"
            %
            (
                np.min(sp.absolute(mfDds.asarray()-sfDds.asarray())),
                np.max(sp.absolute(mfDds.asarray()-sfDds.asarray()))
            )
        )

        self.assertLessEqual(
            np.max(sp.absolute(mfDds.asarray()[slc] - sfDds.asarray()[slc])),
            self.diffTol
        )

        self.assertLessEqual(
            np.max(sp.absolute(mfDds.asarray() - sfDds.asarray())),
            self.diffTol
        )

class _NeighbourhoodFilterTestImpl:
    def testStrideAndOffset(self):
        org = sp.array((-122,333,117), dtype="int32")
        shp = sp.array((64,65,67), dtype="int32")
        imgDds = mango.data.gaussian_noise(shape=shp, origin=org, mean=5000, stdd=4500, dtype="int32")
        se = mango.image.order_se(order=4)
        boffset = (1,2,-3)
        mfDds = self.doMangoFiltering(imgDds, se, boffset=boffset)
        rootLogger.info("=====boffset=%s=======" % (boffset,))
        rootLogger.info("imgDds.origin=%s, imgDds.shape=%s" % (imgDds.origin, imgDds.shape))
        rootLogger.info("mfDds.origin=%s, mfDds.shape=%s" % (mfDds.origin, mfDds.shape))
        self.assertTrue(sp.all(mfDds.origin == (org + boffset)))
        self.assertTrue(sp.all(mfDds.shape == (shp - boffset)))

        eoffset = (-3,1,-2)
        mfDds = self.doMangoFiltering(imgDds, se, eoffset=eoffset)
        rootLogger.info("=====eoffset=%s=======" % (eoffset,))
        rootLogger.info("imgDds.origin=%s, imgDds.shape=%s" % (imgDds.origin, imgDds.shape))
        rootLogger.info("mfDds.origin=%s, mfDds.shape=%s" % (mfDds.origin, mfDds.shape))
        self.assertTrue(sp.all(mfDds.origin == (org)))
        self.assertTrue(sp.all(mfDds.shape == (shp + eoffset)))

        mfDds = self.doMangoFiltering(imgDds, se, boffset=boffset, eoffset=eoffset)
        rootLogger.info("=====boffset=%s, eoffset=%s=======" % (boffset, eoffset,))
        rootLogger.info("imgDds.origin=%s, imgDds.shape=%s" % (imgDds.origin, imgDds.shape))
        rootLogger.info("mfDds.origin=%s, mfDds.shape=%s" % (mfDds.origin, mfDds.shape))
        self.assertTrue(sp.all(mfDds.origin == (org + boffset)))
        self.assertTrue(sp.all(mfDds.shape == (shp -boffset + eoffset)))

        stride = (2,4,3)
        mfDds = self.doMangoFiltering(imgDds, se, stride=stride)
        rootLogger.info("=====stride=%s=======" % (stride,))
        rootLogger.info("imgDds.origin=%s, imgDds.shape=%s" % (imgDds.origin, imgDds.shape))
        rootLogger.info("mfDds.origin=%s, mfDds.shape=%s" % (mfDds.origin, mfDds.shape))
        self.assertTrue(sp.all(mfDds.origin == np.round(sp.array(org,dtype="float64")/stride)))
        self.assertTrue(sp.all(mfDds.shape == sp.ceil(shp/sp.array(stride, dtype="float32"))))

        stride = (2,4,3)
        mfDds = self.doMangoFiltering(imgDds, se, stride=stride, boffset=boffset, eoffset=eoffset)
        rootLogger.info("=====stride=%s, boffset=%s, eoffset=%s=======" % (stride,boffset,eoffset))
        rootLogger.info("imgDds.origin=%s, imgDds.shape=%s" % (imgDds.origin, imgDds.shape))
        rootLogger.info("mfDds.origin=%s, mfDds.shape=%s" % (mfDds.origin, mfDds.shape))
        self.assertTrue(sp.all(mfDds.origin == np.round(sp.array((org+boffset), dtype="float64")/stride)))
        self.assertTrue(sp.all(mfDds.shape == sp.ceil((shp-boffset+eoffset)/sp.array(stride, dtype="float32"))))

    def testFilteringHalo0(self):
        self.doFilterTestWithHalo(0)

    def testFilteringHalo2(self):
        self.doFilterTestWithHalo(2)

    def testFilteringHalo5(self):
        self.doFilterTestWithHalo(5)

class MeanFilterTest(NeighbourhoodFilterTest,_NeighbourhoodFilterTestImpl):
    def setUp(self):
        NeighbourhoodFilterTest.setUp(self)
        self.diffTol = 1

    def callMangoFilter(self, input, se, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0)):
        output = mango.image.mean_filter(input, se, mode=self.mode, stride=stride, boffset=boffset, eoffset=eoffset)
        return output

    def callScipyFilter(self, inDds, se):
        footprint = se.toFootprint()
        kernel = sp.zeros(footprint.shape, dtype="float64")
        kernel[sp.where(footprint)] = 1.0
        rootLogger.info("Num footprint elements = %s" % sp.sum(sp.where(footprint, 1, 0)))
        kernel /= sp.sum(kernel)
        outputArr = sp.ndimage.convolve(inDds.asarray(), weights=kernel, mode=self.mode)
        return outputArr
    
class MedianFilterTest(NeighbourhoodFilterTest,_NeighbourhoodFilterTestImpl):
    def setUp(self):
        NeighbourhoodFilterTest.setUp(self)
        self.diffTol = 0

    def callMangoFilter(self, input, se, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0)):
        output = mango.image.median_filter(input, se, mode=self.mode, stride=stride, boffset=boffset, eoffset=eoffset)
        return output

    def callScipyFilter(self, inDds, se):
        footprint = se.toFootprint()
        rootLogger.info("Num footprint elements = %s" % sp.sum(sp.where(footprint, 1, 0)))
        rootLogger.info("Footprint.shape = %s" % (footprint.shape,))
        outputArr = sp.ndimage.median_filter(inDds.asarray(), footprint=footprint, mode=self.mode)
        return outputArr
    

class ConvolutionFilterTest(NeighbourhoodFilterTest,_NeighbourhoodFilterTestImpl):
    def setUp(self):
        NeighbourhoodFilterTest.setUp(self)
        self.diffTol = 1

    def getWeights(self, se):
        shape = se.toFootprint().shape
        return mango.image.discrete_gaussian_kernel(sigma=np.min(shape)/(2*3.25))
    
    def callMangoFilter(self, input, se, stride=(1,1,1), boffset=(0,0,0), eoffset=(0,0,0)):
        output = mango.image.convolve(input, self.getWeights(se), mode=self.mode, stride=stride, boffset=boffset, eoffset=eoffset)
        return output

    def callScipyFilter(self, inDds, se):
        kernel = self.getWeights(se)
        rootLogger.info("Num kernel elements = %s" % sp.sum(sp.where(kernel, 1, 0)))
        rootLogger.info("kernel.shape = %s" % (kernel.shape,))
        outputArr = sp.ndimage.convolve(inDds.asarray(), weights=kernel, mode=self.mode)
        return outputArr
    

if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    unittest.main()
