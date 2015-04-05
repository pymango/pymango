#!/usr/bin/env python
import logging
import sys
import os

import mango.unittest
import scipy as sp
import numpy as np
import mango.mpi as mpi
import mango.image
import mango.data
import mango.io
from mango.image._sphericalHistRegistration import *

logger, rootLogger = mpi.getLoggers(__name__)

def createPhantom(imgShape, mtype="tomo", origin=(0,0,0), boneFill=1024, voidFill=0):
    img = mango.empty(shape=imgShape, mtype=mtype, origin=origin)
    img.setAllToValue(voidFill)
    c = (sp.array(img.origin) + img.origin + img.shape-1)*0.5
    r = sp.array(img.shape-1)*0.35

    mango.data.fill_ellipsoid(img, centre=c, radius=r, fill=boneFill)
    mango.data.fill_ellipsoid(img, centre=c, radius=r*0.95, fill=voidFill)
    mango.data.fill_ellipsoid(img, centre=(c[0]-r[0]*1.5, c[1], c[2]), radius=r, fill=voidFill)

    mango.data.fill_ellipsoid(img, centre=(c[0]+r[0]*0.70, c[1], c[2]), radius=r*0.25, fill=boneFill)
    mango.data.fill_ellipsoid(img, centre=(c[0]+r[0]*0.70, c[1], c[2]), radius=r*0.19, fill=voidFill)

    mango.data.fill_ellipsoid(
        img,
        centre=c+r*0.6,
        radius=r*0.35,
        fill=boneFill
    )
    mango.data.fill_ellipsoid(
        img,
        centre=c+r*0.6,
        radius=r*0.2,
        fill=voidFill
    )
    
    mango.data.fill_ellipsoid(
        img,
        centre=c-r*0.6,
        radius=r*0.15,
        fill=2*boneFill
    )
    mango.data.fill_ellipsoid(
        img,
        centre=c-r*0.6,
        radius=r*0.05,
        fill=voidFill
    )

    return img, c, r

if mango.haveCGAL:

    class SphericalHistogramRegistrationTest(mango.unittest.TestCase):
        def setUp(self):
            subdShape = sp.array((128,180,180))
            mpiCartShape = mango.mpi.getCartShape(dimension=3)
            mpiCartShape = sp.array(mpiCartShape)
    
            self.imgShape = mpiCartShape*subdShape
    
        def testRotationMatrix(self):
            a = sp.array((1,4,3), dtype="float64")
            b = sp.array((-4,-6,2), dtype="float64")
            r = rotation_matrix(a,b)
            self.assertTrue(sp.all(sp.absolute(r.dot(r.T)-sp.eye(3,3)) < 1.0e-12))
            
            rootLogger.info("r=\n%s" % (r,))
            rootLogger.info("a=%s, b=%s, r.dot(a)=%s" % (a, b, r.dot(a)))

            rootLogger.info("r.dot(r.T)=\n%s" % (r.dot(r.T),))
            
            self.assertTrue(np.linalg.norm(r.dot(a))-np.linalg.norm(a) < 1.0e-12)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            self.assertTrue(sp.all(sp.absolute(r.dot(a.T)-b.T) < 1.0e-12))
            
        def testPearsonrMetric(self):
            #tmpDir = "."
            tmpDir = self.createTmpDir("testSphericalHistogramPopulate")
    
            img, c, r = createPhantom(self.imgShape, voidFill=mango.mtype("tomo").maskValue())
            
            mango.io.writeDds(os.path.join(tmpDir, "tomoSphHistRegPhantomHead.nc"), img)
    
            sphhist = mango.image.spherical_histogram(sphere_c=c-img.origin, sphere_r=1)
            mango.image.intensity_spherical_histogram(img, sphhist)
    
            sphhist.reduce(img.mpi.comm)
            if (mango.haveVTK):
                sphhist.writeVtkXMLUnstructuredGridToFile(os.path.join(tmpDir, "sphhistregphant_intensity.vtu"), img.mpi.comm, 0)
            self.assertEqual(mango.sum(img), sphhist.getNumSamples())
    
            metric = SphHistPearsonrMetric(sphhist, sphhist)
            self.assertEqual(1.0, metric.evaluate(sp.eye(3,3,dtype="float64")))
            
            for angle in range(3,360,3):
                rMatrix = mango.image.rotation_matrix(angle, axis=0, dim=3)
                pearsonr = metric.evaluate(rMatrix)
                rootLogger.info("angle=%s, pearsonr = %s" % (angle,pearsonr))

                self.assertGreater(1.0, pearsonr)

        def testExhaustiveSearchReg(self):
            tmpDir = "."
            #tmpDir = self.createTmpDir("testExhaustiveSearchReg")
    
            img, c, r = createPhantom(self.imgShape, voidFill=mango.mtype("tomo").maskValue())
            img.md.setVoxelSize((1,1,1), "mm")
            mango.io.writeDds(os.path.join(tmpDir, "tomoSphHistRegPhantomHead.nc"), img)
    
            sphhist = mango.image.spherical_histogram(sphere_c=c-img.origin, sphere_r=1, tri_rad_bnd=0.15, tri_cdst_bnd=0.15)
            rootLogger.info("Num spherical hist bins = %s" % sphhist.getNumBins())
            mango.image.intensity_spherical_histogram(img, sphhist)

            sphhist.reduce(img.mpi.comm)
            if (mango.haveVTK):
                sphhist.writeVtkXMLUnstructuredGridToFile(os.path.join(tmpDir, "sphhistregphant_intensity.vtu"), img.mpi.comm, 0)
            self.assertEqual(mango.sum(img), sphhist.getNumSamples())
    
            metric = SphHistPearsonrMetric(sphhist, sphhist)
            reger = SphericalHistRegisterer(metric)
            
            regRotMtx, negPearsonr = reger.search()
            self.assertEqual(-1.0, negPearsonr)
            self.assertTrue(
                sp.all(sp.absolute(sp.eye(3,3) - regRotMtx) < 0.02)
            )
            
            rMtx = \
                mango.image.rotation_matrix(5.0,2).dot(
                    mango.image.rotation_matrix(-10.0, 1).dot(mango.image.rotation_matrix(45.0, 0))
                )
            fixdSphHist = copy.deepcopy(sphhist)
            sphhist.clear()
            trnsSphHist = copy.deepcopy(sphhist)
            trnsSphHist.sphere_c = c-img.origin
            rImg = mango.image.affine_transform(img, rMtx, offset=trnsSphHist.sphere_c, interptype=mango.image.InterpolationType.NEAREST_NEIGHBOUR)
            mango.io.writeDds(os.path.join(tmpDir, "tomoSphHistRegRotatedPhantom.nc"), rImg)
            
            mango.image.intensity_spherical_histogram(rImg, trnsSphHist)
            trnsSphHist.reduce(rImg.mpi.comm)
            metric = SphHistPearsonrMetric(fixdSphHist, trnsSphHist)
            reger = SphericalHistRegisterer(metric)
            
            regRotMtx, negPearsonr = reger.search()
            rootLogger.info("negPearsonr = %s" % negPearsonr)
            self.assertGreater(-0.8, negPearsonr)
            regV = regRotMtx.T.dot((1,1,1))
            rV = rMtx.dot((1,1,1))
            angleDiff = sp.arctan2(sp.linalg.norm(np.cross(rV, regV)), np.dot(rV, regV))*180.0/sp.pi
            rootLogger.info("angleDiff = %s" % angleDiff)
            self.assertGreater(5.0, angleDiff)

    
if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
