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

class PValueTest(mango.unittest.TestCase):

    def testGaussianPValue(self):
        for typePair in [(None, "float32"), ("tomo", None)]:
            mtype = typePair[0]
            dtype = typePair[1]
            mean = 32000.0
            stdd = 1000.0
            noisDds = mango.data.gaussian_noise(shape=(105,223,240), mean=mean, stdd=stdd, mtype=mtype, dtype=dtype)
            
            pvalDds = \
                mango.fmm.gaussian_pvalue(
                    noisDds,
                    mean=mean,
                    stdd=stdd,
                    sidedness=mango.fmm.PValueSidednessType.RIGHT_SIDEDNESS
                )
            
            alpha = 0.05
            count = sp.sum(sp.where(pvalDds.asarray() <= alpha, 1, 0))
            if (pvalDds.mpi.comm != None):
                count = pvalDds.mpi.comm.allreduce(count)
            
            expCount = sp.product(noisDds.shape)*alpha
            count = float(count)
            relErr = sp.absolute(expCount-float(count))/sp.absolute(max(expCount,count))
            rootLogger.info("relErr = %s" % relErr)
            self.assertTrue(relErr < 0.10)

    def testConvolvedGaussianPValue(self):
        import matplotlib
        import matplotlib.pyplot as plt
        
        for typePair in [("float64", None), ("tomo_float", None)]:
            plt.clf()
            mtype = typePair[0]
            dtype = typePair[1]
            mean = 32000.0
            stdd = 1000.0
            noisDds = mango.data.gaussian_noise(shape=(100,101,102), mean=mean, stdd=stdd, mtype=mtype, dtype=dtype)
            kernelSigma = 0.6
            kernel = mango.image.discrete_gaussian_kernel(kernelSigma)
            rootLogger.info("kernal.shape=%s" % (kernel.shape,))
            rootLogger.info("sp.sum(kernal)=%s" % (sp.sum(kernel),))
            rootLogger.info("kernal (min,max)=(%s,%s)" % (np.min(kernel), np.max(kernel),))
            convMeanDds, convStddDds = mango.image.discrete_gaussian_mean_stdd(noisDds, kernelSigma)
            convStddDds = mango.image.discrete_gaussian_stdd(noisDds, kernelSigma, mean=mean)

            estMean = sp.sum(noisDds.asarray())
            if (noisDds.mpi.comm != None):
                estMean = noisDds.mpi.comm.allreduce(estMean)
            estMean /= float(sp.product(noisDds.shape))

            d = noisDds.asarray()-estMean
            estStdd = sp.sum(d*d)
            if (noisDds.mpi.comm != None):
                estStdd = noisDds.mpi.comm.allreduce(estStdd)
            estStdd = sp.sqrt(estStdd/float(sp.product(noisDds.shape)-1))

            convAvgStdd = sp.sum(convStddDds.asarray())
            if (convStddDds.mpi.comm != None):
                convAvgStdd = convStddDds.mpi.comm.allreduce(convAvgStdd)
            convAvgStdd /= float(sp.product(convStddDds.shape))
            convolveMean = sp.sum(kernel*mean)
            convolveStdd = sp.sqrt(sp.sum((kernel*kernel)*(stdd*stdd)))
            
            if ((convStddDds.mpi.comm == None) or (convStddDds.mpi.comm.Get_size() == 1)):
                rootLogger.info("Calculating histogram...")
                hst,edg = sp.histogram((convStddDds.asarray()*convStddDds.asarray()).flatten(), bins=1024)
                rootLogger.info("Done calculating histogram...edge (min,max)=(%s,%s)" % (np.min(edg), np.max(edg)))
                hst = sp.array(hst, copy=True, dtype="float64")
                hst /= sp.sum(hst)
                hst = hst[0:hst.size//2]
                edg = edg[0:hst.size + 1]
                xx = (edg[1:]+edg[0:-1])*0.5
                plt.plot(xx, hst)
                rootLogger.info("Calculating chi2...")
                chi2xx0 = ((sp.product(kernel.shape)-1)/(stdd*stdd))*edg
                df = sp.product(kernel.shape)-1
                chi2cdf0 = sp.stats.chi2.cdf(chi2xx0, df)
                y0 = chi2cdf0[1:]-chi2cdf0[0:-1]
                dof = ((sp.sum(kernel)**2)-(sp.sum(kernel*kernel)))/(sp.sum(kernel))
                coeffs = kernel.flatten()
                rootLogger.info("Calculating gchi2...")
                gchi2xx1 = (dof/(stdd*stdd))*edg
                gchi2cdf1 = mango.fmm.gchisqrd(coeffs, sp.zeros_like(coeffs)).cdf(gchi2xx1)
                y1 = gchi2cdf1[1:]-gchi2cdf1[0:-1]
                rootLogger.info("Done calculating gchi2.")
                
                #plt.plot(xx, y0)
                plt.plot(xx, y1)
                plt.title("sp.sum(hst)=%s, sp.sum(chi2)=%s" % (sp.sum(hst), sp.sum(y0)))
                dir = self.createTmpDir("testConvolvedGaussianPValue")
                plt.savefig(os.path.join(dir,"convStddDds_%s.png" % mtype), dpi=128)
    
                plt.figure()
                hst,edg = sp.histogram(convMeanDds.asarray().flatten(), bins=256)
                hst = sp.array(hst, copy=True, dtype="float64")
                ####hst /= sp.array(edg[1:]-edg[0:-1], dtype="float64")
                hst /= sp.sum(hst)
                xx = (edg[1:]+edg[0:-1])*0.5
                plt.plot(xx, hst)
                nd0 = sp.stats.norm.cdf(edg, loc = mean, scale = sp.sqrt(sp.sum(kernel*kernel)*stdd*stdd))
                y0 = nd0[1:]-nd0[0:-1]
                plt.plot(xx,y0)
                
                
                plt.savefig(os.path.join(dir,"convMeanDds_%s.png" % mtype), dpi=128)

            rootLogger.info("%14s=%10.5f, %14s=%10.5f" % ("mean", mean, "stdd", stdd))
            rootLogger.info("%14s=%10.5f, %14s=%10.5f" % ("convolve-mean", convolveMean, "convolve-stdd", convolveStdd))
            rootLogger.info("%14s=%10.5f, %14s=%10.5f" % ("est-stdd", estStdd, "conv-avg-stdd", convAvgStdd))

            rootLogger.info("Calculating gaussian_pvalue image for from mean-image...")
            meanPValDds = \
                mango.fmm.gaussian_pvalue(
                    convMeanDds,
                    mean=convolveMean,
                    stdd=convolveStdd,
                    sidedness=mango.fmm.PValueSidednessType.RIGHT_SIDEDNESS
                )
            
            alpha = 0.05
            count = sp.sum(sp.where(meanPValDds.asarray() <= alpha, 1, 0))
            if (meanPValDds.mpi.comm != None):
                count = meanPValDds.mpi.comm.allreduce(count)
            
            expCount = sp.product(noisDds.shape)*alpha
            count = float(count)
            relErr = sp.absolute(expCount-float(count))/sp.absolute(max(expCount,count))
            rootLogger.info("relErr = %s" % relErr)
            self.assertTrue(relErr < 0.10)

            convVarDds = mango.copy(convStddDds)
            convVarDds.asarray()[...] *= convVarDds.asarray()*((sp.product(kernel.shape)-1)/(stdd*stdd))
            rootLogger.info("Calculating generalised_chi_squared_pvalue image from variance-image...")

            ###
            ### Takes too long to compute pvalue image!!!!
            ###
#             stddPValDds = \
#                 mango.fmm.generalised_chi_squared_pvalue(
#                     convVarDds,
#                     coefficients=coeffs,
#                     noncentralities=sp.zeros_like(coeffs),
#                     sidedness=mango.fmm.PValueSidednessType.RIGHT_SIDEDNESS
#                 )
#             
#             alpha = 0.05
#             count = sp.sum(sp.where(stddPValDds.asarray() <= alpha, 1, 0))
#             if (stddPValDds.mpi.comm != None):
#                 count = stddPValDds.mpi.comm.allreduce(count)
#             
#             expCount = sp.product(stddPValDds.shape)*alpha
#             count = float(count)
#             relErr = sp.absolute(expCount-float(count))/sp.absolute(max(expCount,count))
#             rootLogger.info("relErr = %s" % relErr)
#             self.assertTrue(relErr < 0.10)

    def testStructuringElementPValue(self):
        import matplotlib
        import matplotlib.pyplot as plt
        
        dir = self.createTmpDir("testStructuringElementPValue")
        for typePair in [("float64", None), ("tomo_float", None)]:
            plt.clf()
            mtype = typePair[0]
            dtype = typePair[1]
            mean = 32000.0
            stdd = 1000.0
            noisDds = mango.data.gaussian_noise(shape=(100,101,102), mean=mean, stdd=stdd, mtype=mtype, dtype=dtype)
            se = mango.image.sphere_se(radius=2.0)
            seNumElems = sp.sum(sp.where(se.toFootprint(), 1, 0))
            rootLogger.info("se.shape=%s" % (se.toFootprint().shape,))
            rootLogger.info("sp.sum(se.toFootprint())=%s" % (seNumElems,))
            seMeanDds = mango.image.mean_filter(noisDds, se)
            seStddDds = mango.image.stdd_filter(noisDds, se)

            estMean = sp.sum(noisDds.asarray())
            if (noisDds.mpi.comm != None):
                estMean = noisDds.mpi.comm.allreduce(estMean)
            estMean /= float(sp.product(noisDds.shape))

            d = noisDds.asarray()-estMean
            estStdd = sp.sum(d*d)
            if (noisDds.mpi.comm != None):
                estStdd = noisDds.mpi.comm.allreduce(estStdd)
            estStdd = sp.sqrt(estStdd/float(sp.product(noisDds.shape)-1))

            seAvgStdd = sp.sum(seStddDds.asarray())
            if (seStddDds.mpi.comm != None):
                seAvgStdd = seStddDds.mpi.comm.allreduce(seAvgStdd)
            seAvgStdd /= float(sp.product(seStddDds.shape))
            seTheoryMean = mean
            seTheoryStdd = sp.sqrt(seNumElems*((1.0/seNumElems)**2)*(stdd*stdd))
            
            if ((seStddDds.mpi.comm == None) or (seStddDds.mpi.comm.Get_size() == 1)):
                rootLogger.info("Calculating histogram...")
                hst,edg = sp.histogram((seStddDds.asarray()*seStddDds.asarray()).flatten(), bins=1024)
                rootLogger.info("Done calculating histogram...edge (min,max)=(%s,%s)" % (np.min(edg), np.max(edg)))
                hst = sp.array(hst, copy=True, dtype="float64")
                hst /= sp.sum(hst)
                hst = hst[0:hst.size//2]
                edg = edg[0:hst.size + 1]
                xx = (edg[1:]+edg[0:-1])*0.5
                plt.plot(xx, hst)
                rootLogger.info("Calculating chi2...")
                chi2xx0 = ((seNumElems-1)/(stdd*stdd))*edg
                df = seNumElems-1
                chi2cdf0 = sp.stats.chi2.cdf(chi2xx0, df)
                y0 = chi2cdf0[1:]-chi2cdf0[0:-1]
                rootLogger.info("Done calculating chi2.")
                
                plt.plot(xx, y0)
                plt.title("sp.sum(hst)=%s, sp.sum(chi2)=%s" % (sp.sum(hst), sp.sum(y0)))
                plt.savefig(os.path.join(dir,"seStddDds_%s.png" % mtype), dpi=128)
    
                plt.figure()
                hst,edg = sp.histogram(seMeanDds.asarray().flatten(), bins=256)
                hst = sp.array(hst, copy=True, dtype="float64")
                ####hst /= sp.array(edg[1:]-edg[0:-1], dtype="float64")
                hst /= sp.sum(hst)
                xx = (edg[1:]+edg[0:-1])*0.5
                plt.plot(xx, hst)
                nd0 = sp.stats.norm.cdf(edg, loc = mean, scale = sp.sqrt(seNumElems*((1.0/seNumElems)**2)*stdd*stdd))
                y0 = nd0[1:]-nd0[0:-1]
                plt.plot(xx,y0)
                
                
                plt.savefig(os.path.join(dir,"seMeanDds_%s.png" % mtype), dpi=128)

            rootLogger.info("%14s=%10.5f, %14s=%10.5f" % ("mean", mean, "stdd", stdd))
            rootLogger.info("%14s=%10.5f, %14s=%10.5f" % ("seTheory-mean", seTheoryMean, "seTheory-stdd", seTheoryStdd))
            rootLogger.info("%14s=%10.5f, %14s=%10.5f" % ("est-stdd", estStdd, "se-avg-stdd", seAvgStdd))

            rootLogger.info("Calculating gaussian_pvalue image for from mean-image...")
            meanPValDds = \
                mango.fmm.gaussian_pvalue(
                    seMeanDds,
                    mean=seTheoryMean,
                    stdd=seTheoryStdd,
                    sidedness=mango.fmm.PValueSidednessType.RIGHT_SIDEDNESS
                )
            
            alpha = 0.05
            count = sp.sum(sp.where(meanPValDds.asarray() <= alpha, 1, 0))
            if (meanPValDds.mpi.comm != None):
                count = meanPValDds.mpi.comm.allreduce(count)
            
            expCount = sp.product(noisDds.shape)*alpha
            count = float(count)
            relErr = sp.absolute(expCount-float(count))/sp.absolute(max(expCount,count))
            rootLogger.info("relErr = %s" % relErr)
            self.assertTrue(relErr < 0.10)

            seVarDds = mango.copy(seStddDds)
            seVarDds.asarray()[...] *= seVarDds.asarray()*((seNumElems)-1)/(stdd*stdd)
            rootLogger.info("Calculating chi_squared_pvalue image from variance-image...")

            ###
            ### Takes too long to compute pvalue image!!!!
            ###
            stddPValDds = \
                mango.fmm.chi_squared_pvalue(
                    seVarDds,
                    dof=float(seNumElems-1),
                    sidedness=mango.fmm.PValueSidednessType.RIGHT_SIDEDNESS
                )
             
            alpha = 0.05
            count = sp.sum(sp.where(stddPValDds.asarray() <= alpha, 1, 0))
            if (stddPValDds.mpi.comm != None):
                count = stddPValDds.mpi.comm.allreduce(count)
             
            expCount = sp.product(stddPValDds.shape)*alpha
            count = float(count)
            relErr = sp.absolute(expCount-float(count))/sp.absolute(max(expCount,count))
            rootLogger.info("relErr = %s" % relErr)
            self.assertTrue(relErr < 0.10)

    def testChiSquaredPValue(self):
        for typePair in [(None, "float32"), ("float64", None)]:
            mtype = typePair[0]
            dtype = typePair[1]
            dof = 12.0
            noisDds = mango.data.chi_squared_noise(shape=(105,223,240), dof=dof, mtype=mtype, dtype=dtype)
            
            pvalDds = \
                mango.fmm.chi_squared_pvalue(
                    noisDds,
                    dof=dof,
                    sidedness=mango.fmm.PValueSidednessType.RIGHT_SIDEDNESS
                )
            
            alpha = 0.05
            count = sp.sum(sp.where(pvalDds.asarray() <= alpha, 1, 0))
            if (pvalDds.mpi.comm != None):
                count = pvalDds.mpi.comm.allreduce(count)
            
            expCount = sp.product(noisDds.shape)*alpha
            count = float(count)
            relErr = sp.absolute(expCount-float(count))/sp.absolute(max(expCount,count))
            rootLogger.info("relErr = %s" % relErr)
            self.assertTrue(relErr < 0.10)

class GradientChiSquaredPValueTest(mango.unittest.TestCase):

    def testDiscreteGaussianGradientChiSquaredPValue(self):
        dir = self.createTmpDir("testDiscreteGaussianGradientChiSquaredPValue")
        gSigma = 0.5
        grdKernelZ = mango.image.discrete_gaussian_gradient_kernel(axis=0, sigma=gSigma)
        sumSqrd = sp.sum(grdKernelZ*grdKernelZ)
        rootLogger.info("grdKernelZ = \n%s" % (grdKernelZ,))

        mean = 32000.0
        stdd = 4000.0
        imgDds = mango.data.gaussian_noise(shape=(100,200,200), mean=mean, stdd=stdd, mtype="tomo_float", halo=(3,3,3))

        grdDds = mango.image.discrete_gaussian_gradient_magnitude(input=imgDds, sigma=gSigma)

        mango.io.writeDds(os.path.join(dir, "tomo_floatImg.nc"), imgDds)
        mango.io.writeDds(os.path.join(dir, "tomo_floatImgGrd.nc"), grdDds)

        grdDds.updateHaloRegions()
        grdDds.mirrorOuterLayersToBorder(False)
        grdDds.subd.asarray()[...] =  grdDds.subd.asarray()*grdDds.subd.asarray()
        grdDds.subd.asarray()[...] /= (sumSqrd*(stdd*stdd))

        if ((grdDds.mpi.comm == None) or (grdDds.mpi.comm.Get_size() <= 1)):
            import matplotlib
            import matplotlib.pyplot as plt

            hst,edg = sp.histogram(grdDds.subd.asarray().flatten(), bins=1024)
            rootLogger.info("Done calculating histogram...edge (min,max)=(%s,%s)" % (np.min(edg), np.max(edg)))
            hst = sp.array(hst, copy=True, dtype="float64")
            hst /= sp.sum(hst)
            hst = hst[0:hst.size//2]
            edg = edg[0:hst.size + 1]
            xx = (edg[1:]+edg[0:-1])*0.5
            plt.plot(xx, hst, label="gradient histogram")
            rootLogger.info("Calculating chi2...")
            chi2xx0 = edg
            df = 3
            chi2cdf0 = sp.stats.chi2.cdf(chi2xx0, df)
            y0 = chi2cdf0[1:]-chi2cdf0[0:-1]

            plt.plot(xx, y0, label="Chi-squared distribution")
            plt.legend()
            plt.savefig(os.path.join(dir,"gradient_hist_and_chi_sqrd_dist.png"), dpi=100)


if __name__ == "__main__":
    #mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.unittest", "mango.mpi", "mango.fmm", "mango.fmmTest"],
        logLevel=logging.DEBUG
    )
    random.seed((mpi.rank+1)*23456243)
    numpy.random.seed((mpi.rank+1)*23456134)

    mango.unittest.main()
