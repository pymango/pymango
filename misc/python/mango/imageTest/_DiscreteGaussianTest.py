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


class DiscreteGaussianKernelTest(unittest.TestCase):
    def testDiscreteGaussianKernel(self):
        k001 = mango.image.discrete_gaussian_kernel(1.0, errtol=0.01)
        rootLogger.info("kernel.shape=%s" % (k001.shape,))
        self.assertEqual(3, len(k001.shape))
        self.assertEqual(k001.shape[0], k001.shape[1])
        self.assertEqual(k001.shape[1], k001.shape[2])
        self.assertTrue(sp.all(k001 > 0))

        k0001 = mango.image.discrete_gaussian_kernel(1.0, errtol=0.001)
        rootLogger.info("kernel.shape=%s" % (k0001.shape,))
        self.assertEqual(3, len(k0001.shape))
        self.assertEqual(k0001.shape[0], k0001.shape[1])
        self.assertEqual(k0001.shape[1], k0001.shape[2])
        self.assertTrue(sp.all(k0001 > 0))
        
        self.assertGreater(k0001.shape[0], k001.shape[1])

    def testWidth(self):
        dgK = mango.image.discrete_gaussian_kernel(sigma=0.1)
        self.assertEqual(3, len(dgK.shape))
        self.assertEqual(3, dgK.shape[0])
        self.assertEqual(3, dgK.shape[1])
        self.assertEqual(3, dgK.shape[2])
        
        dgK = mango.image.discrete_gaussian_kernel(sigma=0.63)
        self.assertEqual(3, len(dgK.shape))
        self.assertEqual(5, dgK.shape[0])
        self.assertEqual(5, dgK.shape[1])
        self.assertEqual(5, dgK.shape[2])
    
    def testKernel(self):
        scale=0.4
        dgK = mango.image.discrete_gaussian_kernel(sigma=scale, dim=2)
        logger.info("kernel=\n%s" % dgK)
        self.assertEqual(2, len(dgK.shape))

    def testKernelCoeffs(self):
        for scale in [0.35, 0.5, 0.75, 1]:
            for dim in [0,1,2,3,4]:
                dgK = mango.image.discrete_gaussian_kernel(sigma=scale, dim=dim, errtol=0.001)
                self.assertAlmostEqual(1.0, sp.sum(dgK), 8)
                mxElem = sp.argmax(dgK)
                self.assertTrue(sp.all(sp.array(dgK.shape)//2 == sp.unravel_index(mxElem, dgK.shape)))


class DiscreteGaussianTest(unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((64,64,64))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def doTestDiscreteGaussianWithHalo(self, haloSz=0):
        if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
            if (haloSz < 0):
                haloSz = 0
            haloSz = sp.array((haloSz,)*3)
        
        imgDds = mango.zeros(shape=self.imgShape, mtype="tomo_float", halo=haloSz)
        
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], imgDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)
        
        dgDds = mango.image.discrete_gaussian(imgDds, sigma=0.5)
        
        self.assertTrue(sp.all(imgDds.dtype == dgDds.dtype))
        self.assertTrue(sp.all(imgDds.mtype == dgDds.mtype), "%s != %s" % (imgDds.mtype, dgDds.mtype))
        self.assertTrue(sp.all(imgDds.halo == dgDds.halo))
        self.assertTrue(sp.all(imgDds.shape == dgDds.shape))
        self.assertTrue(sp.all(imgDds.origin == dgDds.origin), "%s != %s" % (imgDds.origin, dgDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == dgDds.mpi.shape))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("dgDds min = %s, dgDds max = %s" % (np.min(dgDds.asarray()[slc]), np.max(dgDds.asarray()[slc])))
        logger.info("num non-zero dgDds = %s" % sp.sum(sp.where(dgDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray() == 0))
        self.assertTrue(sp.all(dgDds.asarray()[slc] == 0))

        imgDds = mango.zeros(shape=self.imgShape, mtype="tomo", halo=haloSz, origin=(-32,8,64))
        dgDds = mango.image.discrete_gaussian(imgDds, sigma=1.0)

        self.assertTrue(sp.all(imgDds.dtype == dgDds.dtype))
        self.assertTrue(sp.all(imgDds.mtype == dgDds.mtype))
        self.assertTrue(sp.all(imgDds.halo == dgDds.halo))
        self.assertTrue(sp.all(imgDds.shape == dgDds.shape))
        self.assertTrue(sp.all(imgDds.origin == dgDds.origin), "%s != %s" % (imgDds.origin, dgDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == dgDds.mpi.shape))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("dgDds min = %s, dgDds max = %s" % (np.min(dgDds.asarray()[slc]), np.max(dgDds.asarray()[slc])))
        logger.info("num non-zero dgDds = %s" % sp.sum(sp.where(dgDds.asarray()[slc] != 0, 1, 0)))
        
        self.assertTrue(sp.all(imgDds.asarray() == 0))
        self.assertTrue(sp.all(dgDds.asarray()[slc] == 0))

    def testDiscreteGaussianWithHalo0(self):
        self.doTestDiscreteGaussianWithHalo(0)

    def testDiscreteGaussianWithHalo1(self):
        self.doTestDiscreteGaussianWithHalo(2)

    def testDiscreteGaussianWithHalo2(self):
        self.doTestDiscreteGaussianWithHalo(5)

    def testConvolution(self):
        inDds = mango.data.gaussian_noise(shape=(128,128,128), mean=32000.0, stdd=1000.0, mtype="float64")
        kernelSigma = 0.2
        kernel = mango.image.discrete_gaussian_kernel(kernelSigma)

        mn,mx = (np.min(inDds.asarray()), np.max(inDds.asarray()))
        if (inDds.mpi.comm != None):
            mn = inDds.mpi.comm.allreduce(mn, mpi.MIN)
            mx = inDds.mpi.comm.allreduce(mx, mpi.MAX)
        rootLogger.info("inDds (min, max)=(%s,%s)" % (mn,mx))
        rootLogger.info("Convolving Dds (inDds.asarray().shape=%s), with kernel, kernel.shape=%s..." % (inDds.asarray().shape, kernel.shape,))       
        cDds = mango.image.convolve(inDds, kernel)
        rootLogger.info("Blurring Dds with Discrete Gaussian...")
        dgDds = mango.image.discrete_gaussian(inDds, kernelSigma)

        mn,mx = (np.min(cDds.asarray()), np.max(cDds.asarray()))
        if (cDds.mpi.comm != None):
            mn = cDds.mpi.comm.allreduce(mn, mpi.MIN)
            mx = cDds.mpi.comm.allreduce(mx, mpi.MAX)
        rootLogger.info("cDds (min, max)=(%s,%s)" % (mn,mx))

        self.assertTrue(sp.all(sp.absolute(cDds.asarray()-dgDds.asarray()) < 1.0e-10))
        self.assertTrue(sp.all(cDds.asarray() == dgDds.asarray()))

if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    unittest.main()
