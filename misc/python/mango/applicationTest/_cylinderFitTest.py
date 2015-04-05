#!/usr/bin/env python
import os
import logging
import mango.unittest
import scipy as sp
import numpy as np
import mango
import mango.data
import mango.optimize
import mango.mpi as mpi
from mango.application.cylinder_fit import *

logger, rootLogger = mpi.getLoggers(__name__)

if (mango.haveRestricted):
    class CylinderFitGradientMetricTest(mango.unittest.TestCase):
        """
        Test for the :class:`mango.application.cylinder_fit.CylinderFitGradientMetric`.
        """
        def setUp(self):
            pass

        def testNonDistributedMetric(self):
            if ((mpi.world == None) or (mpi.world.Get_size() == 1)):
                img = mango.zeros((128,64,64), mtype="tomo")
                img.md.setVoxelSize((1,1,1), "voxel")
                c = img.origin + img.shape*0.5
                r = np.min(img.shape)*0.4
                a = (1,0,0)
                mango.data.fill_circular_cylinder(img, c, r, img.shape[0]*1.1, fill=32000, coordsys="abs", unit="voxel")
        
                metric = cylinder_fit_gradient_metric()
                metric.setGradientFromImage(img)
                x = metric.getDefaultX()
                rootLogger.info("Default x = %s" % (x,))
                x[metric.CENTRE_X_IDX] = c[2]
                x[metric.CENTRE_Y_IDX] = c[1]
                x[metric.CENTRE_Z_IDX] = c[0]
                x[metric.AXIS_X_IDX] = a[2]
                x[metric.AXIS_Y_IDX] = a[1]
                x[metric.AXIS_Z_IDX] = a[0]
                bestSoln = (0,0)
                for rr in range(-5, 6):
                    x[metric.RADIUS_IDX] = r + rr
                    val = metric(x)
                    if (val > bestSoln[0]):
                        bestSoln = (val, x[metric.RADIUS_IDX])
                    rootLogger.info("metric[%s] = %16.2f" % (x, val))
                
                self.assertEqual(bestSoln[1], r)
                self.assertLess(0, bestSoln[0])
    
        def testNonDistributedOptimization(self):
            if ((mpi.world == None) or (mpi.world.Get_size() == 1)):
                img = mango.zeros((128,64,64), mtype="tomo")
                img.md.setVoxelSize((1,1,1), "voxel")
                c = img.origin + img.shape*0.5
                r = np.min(img.shape)*0.4
                a = (1,0,0)
                mango.data.fill_circular_cylinder(img, c, r, img.shape[0]*1.1, fill=32000, coordsys="abs", unit="voxel")
        
                metric = cylinder_fit_gradient_metric()
                metric.setGradientFromImage(img)
                x = metric.getDefaultX()
                rootLogger.info("Default x = %s" % (x,))
                x[metric.RADIUS_IDX]   = r
                x[metric.CENTRE_X_IDX] = c[2]
                x[metric.CENTRE_Y_IDX] = c[1]
                x[metric.CENTRE_Z_IDX] = c[0]
                x[metric.AXIS_X_IDX] = a[2]
                x[metric.AXIS_Y_IDX] = a[1]
                x[metric.AXIS_Z_IDX] = a[0]
                solnX = x.copy()
                bestRes = None
    
                for rr in range(-5, 6):
                    x[metric.RADIUS_IDX] = r + rr
                    res = sp.optimize.minimize(lambda x: -sp.absolute(metric(x)), x[0:6], method="Powell", options={'xtol':1.0e-1, 'ftol':1.0e-2})
                    #res = sp.optimize.minimize(metric, x[0:6], jac=False, method="BFGS", options={'gtol':1.0e3, 'eps':(0.05,0.05,0.05,0.001,0.001,0.001)})
                    rootLogger.info("result.success = %s, result.message = %s" % (res.success, res.message))
                    rootLogger.info("optimized (metric_val, x) =  (%16.2f, %s) = " % (res.fun, res.x))
                    rootLogger.info("res.nfev = %8d" % (res.nfev,))
                    
                    if ((bestRes == None) or (res.fun < bestRes.fun)):
                        bestRes = res
                        
                bestRes.x[3:] /= sp.sqrt(sp.sum(bestRes.x[3:]*bestRes.x[3:]))
                rootLogger.info("soln x = %s" % (solnX[0:bestRes.x.size],))
                rootLogger.info("best x = %s" % (bestRes.x,))
    
                self.assertTrue(sp.all(sp.absolute(bestRes.x - solnX[0:bestRes.x.size]) < 0.1))
                self.assertGreater(0, bestRes.fun)
    
        def testDistributedOptimization(self):
            if ((mpi.size % 4) == 0):
                mpidims = (0,2,2)
            elif ((mpi.size % 2) == 0):
                mpidims = (0,0,2)
            else:
                mpidims = (0,0,0)
            img = mango.zeros((120+8*mpi.size,64,64), mtype="tomo", mpidims=mpidims)
            img.md.setVoxelSize((1,1,1), "voxel")
            c = img.origin + img.shape*0.5
            r = np.min(img.shape)*0.4
            a = (1,0,0)
            mango.data.fill_circular_cylinder(img, c, r, img.shape[0]*1.1, fill=32000, coordsys="abs", unit="voxel")
    
            metric = cylinder_fit_gradient_metric()
            metric = DistributedGradCylFitMetric(metric, comm=img.mpi.comm, root=0)
            metric.setGradientFromImage(img)

            x = metric.getDefaultX()
            rootLogger.info("Default x = %s" % (x,))
            x[metric.RADIUS_IDX]   = r
            x[metric.CENTRE_X_IDX] = c[2]
            x[metric.CENTRE_Y_IDX] = c[1]
            x[metric.CENTRE_Z_IDX] = c[0]
            x[metric.AXIS_X_IDX] = a[2]
            x[metric.AXIS_Y_IDX] = a[1]
            x[metric.AXIS_Z_IDX] = a[0]
            solnX = x.copy()
            bestRes = None
            for rr in range(-5, 6):
                x[metric.RADIUS_IDX] = r + rr
                res = mango.optimize.distributed_minimize(metric, x[0:5], method="Powell", options={'xtol':1.0e-1, 'ftol':1.0e-2})
                #res = sp.optimize.minimize(metric, x[0:6], jac=False, method="BFGS", options={'gtol':1.0e3, 'eps':(0.05,0.05,0.05,0.001,0.001,0.001)})
                rootLogger.info("result.success = %s, result.message = %s" % (res.success, res.message))
                rootLogger.info("optimized (metric_val, x) =  (%16.2f, %s) = " % (res.fun, res.x))
                rootLogger.info("res.nfev = %8d" % (res.nfev,))

                if ((bestRes == None) or (res.fun < bestRes.fun)):
                    bestRes = res

            #bestRes.x[3:] /= sp.sqrt(sp.sum(bestRes.x[3:]*bestRes.x[3:]))
            rootLogger.info("soln x = %s" % (solnX[0:bestRes.x.size],))
            rootLogger.info("best x = %s" % (bestRes.x,))
    
            self.assertTrue(sp.all(sp.absolute(bestRes.x - solnX[0:bestRes.x.size]) < 0.1))
            self.assertGreater(0, bestRes.fun)

    class CylinderFitTest(mango.unittest.TestCase):
        """
        Test for the :func:`mango.application.cylinder_fit.cylinder_fit` function.
        """
        def setUp(self):
            pass

        def testDistributedMetricFit1(self):
            dir = self.createTmpDir("CylinderFitTest_testDistributedMetricFit1")
            #dir = "."
            rootLogger.info("Generating synthetic image...")
            #img = mango.zeros((128+((mpi.size//8)*16),64,64), mtype="tomo")
            img = mango.zeros((256+((mpi.size//8)*16),128,128), mtype="tomo")
            #img = mango.zeros((1024,512,512), mtype="tomo")
            img.md.setVoxelSize((0.002,0.002,0.002), "mm")
            c = (img.origin + img.shape*0.5)*img.md.getVoxelSize()
            r = np.min(img.shape*img.md.getVoxelSize())*0.4
            a = (1,0,0)
            img.subd_h.asarray()[...] = 10000
            mango.data.fill_circular_cylinder(img, c, r, img.shape[0]*1.1, fill=25000, coordsys="abs", unit=img.md.getVoxelSizeUnit())
            img.subd.asarray()[...] += mango.data.gaussian_noise_like(img, mean=0, stdd=300).subd.asarray()
            mango.io.writeDds(os.path.join(dir,"tomoCyl1.nc"), img)
            rootLogger.info("Done generating synthetic image.")

            fit, cylImg = cylinder_fit(img, 1, retcylimg=True, metricThreshRelTol=0.50)
            mango.io.writeDds(os.path.join(dir,"segmentedCyl1AnnularMsk.nc"), cylImg)
            rootLogger.info("Img voxel size     = %s %s" % (img.md.getVoxelSize(), img.md.getVoxelSizeUnit()))
            rootLogger.info("Cyl img voxel size = %s %s" % (cylImg.md.getVoxelSize(), cylImg.md.getVoxelSizeUnit()))
            rootLogger.info("c=%s, r=%s, a=%s" % (c,r,a))
            rootLogger.info("Fit Results:\n%s" % ("\n".join(map(str,fit)),))
            f0 = fit[0]
            self.assertAlmostEqual(c[1], f0[1][0][1], 3)
            self.assertAlmostEqual(c[2], f0[1][0][2], 3)
            self.assertAlmostEqual(r, f0[1][1], 3)
            self.assertAlmostEqual(sp.absolute(sp.dot(a, f0[1][3])), 1.0, 3)

        def _testDistributedMetricFit3(self):
            dir = self.createTmpDir("CylinderFitTest_testDistributedMetricFit3")
            #dir = "."
            rootLogger.info("Generating synthetic image...")
            #img = mango.zeros((128+((mpi.size//8)*16),64,64), mtype="tomo")
            img = mango.zeros((256+((mpi.size//8)*16),128,128), mtype="tomo")
            #img = mango.zeros((1024,512,512), mtype="tomo")
            img.md.setVoxelSize((0.002,0.002,0.002), "mm")
            c = (img.origin + img.shape*0.5)*img.md.getVoxelSize()
            r = np.min(img.shape*img.md.getVoxelSize())*0.4
            a = sp.array((1.,0.,0.))
            
            cList = [c, c, c+(img.shape*0.02)*img.md.getVoxelSize()]
            rList = [r, 0.8*r, 0.65*r]
            aList = [a, a, sp.array([1.0, 0.01,0.02])]
            fList = [20000, 15000, 18000]
            img.subd_h.asarray()[...] = 10000
            for i in range(len(cList)):
                rMtx = mango.math.rotation_matrix_from_cross_prod(sp.array([1.,0.,0.]), aList[i])
                mango.data.fill_circular_cylinder(img, cList[i], rList[i], img.shape[0]*1.25, fill=fList[i], rotation=rMtx, coordsys="abs", unit=img.md.getVoxelSizeUnit())
            img.subd.asarray()[...] += mango.data.gaussian_noise_like(img, mean=0, stdd=300).subd.asarray()
            mango.io.writeDds(os.path.join(dir, "tomoCyl3.nc"), img)
            rootLogger.info("Done generating synthetic image.")

            fit, cylImg = \
                cylinder_fit(
                    img,
                    numcyl=3,
                    retcylimg=True,
                    distributedMetricEvaluation=True,
                    metricThreshRelTol=0.5
                )
            mango.io.writeDds(os.path.join(dir,"segmentedCyl3AnnularMsk.nc"), cylImg)
            rootLogger.info("Voxel size = %s %s" % (img.md.getVoxelSize(), img.md.getVoxelSizeUnit()))
            for i in range(len(cList)):
                rootLogger.info("c%s=%s, r%s=%s, a%s=%s" % (i,cList[i],i,rList[i],i,aList[i]))
            rootLogger.info("Fit Results:\n%s" % ("\n".join(map(str,fit)),))

        def _testNonDistributedMetricFit3(self):
            dir = self.createTmpDir("CylinderFitTest_testNonDistributedMetricFit3")
            #dir = "."
            rootLogger.info("Generating synthetic image...")
            #img = mango.zeros((128+((mpi.size//8)*16),64,64), mtype="tomo")
            img = mango.zeros((256+((mpi.size//8)*16),128,128), mtype="tomo")
            #img = mango.zeros((1024,512,512), mtype="tomo")
            img.md.setVoxelSize((0.002,0.002,0.002), "mm")
            c = (img.origin + img.shape*0.5)*img.md.getVoxelSize()
            r = np.min(img.shape*img.md.getVoxelSize())*0.4
            a = sp.array((1.,0.,0.))
            
            cList = [c, c, c+(img.shape*0.02)*img.md.getVoxelSize()]
            rList = [r, 0.8*r, 0.65*r]
            aList = [a, a, sp.array([1.0, 0.01,0.02])]
            fList = [20000, 15000, 18000]
            img.subd_h.asarray()[...] = 10000
            for i in range(len(cList)):
                rMtx = mango.math.rotation_matrix_from_cross_prod(sp.array([1.,0.,0.]), aList[i])
                mango.data.fill_circular_cylinder(img, cList[i], rList[i], img.shape[0]*1.25, fill=fList[i], rotation=rMtx, coordsys="abs", unit=img.md.getVoxelSizeUnit())
            img.subd.asarray()[...] += mango.data.gaussian_noise_like(img, mean=0, stdd=300).subd.asarray()
            mango.io.writeDds(os.path.join(dir, "tomoCyl3.nc"), img)
            rootLogger.info("Done generating synthetic image.")

            fit, cylImg = \
                cylinder_fit(
                    img,
                    numcyl=3,
                    retcylimg=True,
                    distributedMetricEvaluation=False,
                    metricThreshRelTol=0.5
                )
            mango.io.writeDds(os.path.join(dir,"segmentedCyl3AnnularMsk.nc"), cylImg)
            rootLogger.info("Voxel size = %s %s" % (img.md.getVoxelSize(), img.md.getVoxelSizeUnit()))
            for i in range(len(cList)):
                rootLogger.info("c%s=%s, r%s=%s, a%s=%s" % (i,cList[i],i,rList[i],i,aList[i]))
            rootLogger.info("Fit Results:\n%s" % ("\n".join(map(str,fit)),))

    class CylinderFitMultiResTest(mango.unittest.TestCase):
        """
        Test for the :func:`mango.application.cylinder_fit.cylinder_fit_multi_res` function.
        """
        def setUp(self):
            pass

        def testMultiResFit1(self):
            dir = self.createTmpDir("CylinderFitMultiResTest_testMultiResFit1")
            #dir = "."
            rootLogger.info("Generating synthetic image...")
            #img = mango.zeros((128+((mpi.size//8)*16),64,64), mtype="tomo")
            img = mango.zeros((512+((mpi.size//8)*16),256,256), mtype="tomo")
            #img = mango.zeros((1024,512,512), mtype="tomo")
            img.md.setVoxelSize((0.002,0.002,0.002), "mm")
            c = (img.origin + img.shape*0.5)*img.md.getVoxelSize()
            r = np.min(img.shape*img.md.getVoxelSize())*0.4
            a = (1,0,0)
            img.subd_h.asarray()[...] = 10000
            mango.data.fill_circular_cylinder(img, c, r, img.shape[0]*1.1, fill=25000, coordsys="abs", unit=img.md.getVoxelSizeUnit())
            img.subd.asarray()[...] += mango.data.gaussian_noise_like(img, mean=0, stdd=300).subd.asarray()
            mango.io.writeDds(os.path.join(dir,"tomoCyl1.nc"), img)
            rootLogger.info("Done generating synthetic image.")

            fit, cylImg = cylinder_fit_multi_res(img, 1, resolutions=[64, 128, 512], retcylimg=True, metricThreshRelTol=0.50)
            mango.io.writeDds(os.path.join(dir,"segmentedCyl1AnnularMsk.nc"), cylImg)
            rootLogger.info("Img voxel size     = %s %s" % (img.md.getVoxelSize(), img.md.getVoxelSizeUnit()))
            rootLogger.info("Cyl img voxel size = %s %s" % (cylImg.md.getVoxelSize(), cylImg.md.getVoxelSizeUnit()))
            rootLogger.info("c=%s, r=%s, a=%s" % (c,r,a))
            rootLogger.info("Fit Results:\n%s" % ("\n".join(map(str,fit)),))
            f0 = fit[0]
            self.assertAlmostEqual(c[1], f0[1][0][1], 3)
            self.assertAlmostEqual(c[2], f0[1][0][2], 3)
            self.assertAlmostEqual(r, f0[1][1], 3)
            self.assertAlmostEqual(sp.absolute(sp.dot(a, f0[1][3])), 1.0, 3)

        def _testMultiResFit3(self):
            dir = self.createTmpDir("CylinderFitMultiResTest_testMultiResFit3")
            #dir = "."
            rootLogger.info("Generating synthetic image...")
            #img = mango.zeros((128+((world.size//8)*16),64,64), mtype="tomo")
            img = mango.zeros((256+((world.size//8)*16),128,128), mtype="tomo")
            #img = mango.zeros((1024,512,512), mtype="tomo")
            img.md.setVoxelSize((0.002,0.002,0.002), "mm")
            c = (img.origin + img.shape*0.5)*img.md.getVoxelSize()
            r = np.min(img.shape*img.md.getVoxelSize())*0.4
            a = sp.array((1.,0.,0.))
            
            cList = [c, c, c+(img.shape*0.02)*img.md.getVoxelSize()]
            rList = [r, 0.8*r, 0.65*r]
            aList = [a, a, sp.array([1.0, 0.01,0.02])]
            fList = [20000, 15000, 18000]
            img.subd_h.asarray()[...] = 10000
            for i in range(len(cList)):
                rMtx = mango.math.rotation_matrix_from_cross_prod(sp.array([1.,0.,0.]), aList[i])
                mango.data.fill_circular_cylinder(img, cList[i], rList[i], img.shape[0]*1.25, fill=fList[i], rotation=rMtx, coordsys="abs", unit=img.md.getVoxelSizeUnit())
            img.subd.asarray()[...] += mango.data.gaussian_noise_like(img, mean=0, stdd=300).subd.asarray()
            mango.io.writeDds(os.path.join(dir, "tomoCyl3.nc"), img)
            rootLogger.info("Done generating synthetic image.")

            fit, cylImg = \
                cylinder_fit_multi_res(
                    img,
                    3,
                    resolutions=[64, 128, 512],
                    retcylimg=True,
                    metricThreshRelTol=0.50
                )
            mango.io.writeDds(os.path.join(dir,"segmentedCyl3AnnularMsk.nc"), cylImg)
            rootLogger.info("Voxel size = %s %s" % (img.md.getVoxelSize(), img.md.getVoxelSizeUnit()))
            for i in range(len(cList)):
                rootLogger.info("c%s=%s, r%s=%s, a%s=%s" % (i,cList[i],i,rList[i],i,aList[i]))
            rootLogger.info("Fit Results:\n%s" % ("\n".join(map(str,fit)),))

if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.application"],
        logLevel=logging.INFO
    )
    mango.setLoggingVerbosityLevel("high")
    mango.unittest.main()
