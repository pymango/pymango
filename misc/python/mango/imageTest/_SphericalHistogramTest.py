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
import copy
logger, rootLogger = mpi.getLoggers(__name__)

def createPhantom(imgShape, mtype="tomo", origin=(0,0,0), boneFill=1024, voidFill=0):
    img = mango.empty(shape=imgShape, mtype=mtype, origin=origin)
    img.setAllToValue(voidFill)
    c = (sp.array(img.origin) + img.origin + img.shape-1)*0.5
    r = sp.array(img.shape-1)*0.5

    mango.data.fill_ellipsoid(img, centre=c, radius=r, fill=boneFill)
    mango.data.fill_ellipsoid(img, centre=c, radius=r*0.95, fill=voidFill)
    mango.data.fill_ellipsoid(img, centre=(c[0]-r[0]*1.5, c[1], c[2]), radius=r, fill=voidFill)

    return img, c, r

if mango.haveCGAL:

    class SphericalHistogramTest(mango.unittest.TestCase):
        def setUp(self):
            subdShape = sp.array((60,80,80))
            mpiCartShape = mango.mpi.getCartShape(dimension=3)
            mpiCartShape = sp.array(mpiCartShape)
    
            self.imgShape = mpiCartShape*subdShape
    
        def testSphericalHistogram(self):
            #tmpDir = "."
            tmpDir = self.createTmpDir("testSphericalHistogram")
    
            img, c, r = createPhantom(self.imgShape, voidFill=mango.mtype("tomo").maskValue())
            
            mango.io.writeDds(os.path.join(tmpDir, "tomoPhantomHead.nc"), img)
    
            sphhist = mango.image.spherical_histogram(sphere_c=c-img.origin, sphere_r=1)
            mango.image.intensity_spherical_histogram(img, sphhist)
    
            sphhist.reduce(img.mpi.comm)
            
            sphhistcpy = copy.deepcopy(sphhist)
            
            self.assertTrue(sp.all(sphhist.getBinCounts() == sphhistcpy.getBinCounts()))
            self.assertTrue(sp.all(sphhist.getBinCentres() == sphhistcpy.getBinCentres()))
            
            tmp_c = sphhistcpy.sphere_c
            sphhistcpy.sphere_c = tmp_c + sp.array((0.25,0.5,0.125))*0.33333333333333
            self.assertEqual(sphhist.getBinCentres().size, sphhistcpy.getBinCentres().size)
            self.assertEqual(sphhist.getBinAreas().size, sphhistcpy.getBinAreas().size)

            self.assertTrue(
                sp.all(
                    sp.sqrt(
                        (sphhistcpy.getBinCentres() - sphhistcpy.sphere_c)**2
                    )
                    <=
                    sphhistcpy.sphere_r
                )
            )
            self.assertTrue(
                sp.all(
                    sp.absolute(
                        (sphhistcpy.getBinCentres() - sphhistcpy.sphere_c)
                        -
                        (sphhist.getBinCentres() - sphhist.sphere_c)
                    )
                    <=
                    1.0e-12
                )
            )

    class ImageSphericalHistogramTest(mango.unittest.TestCase):
        def setUp(self):
            subdShape = sp.array((60,80,80))
            mpiCartShape = mango.mpi.getCartShape(dimension=3)
            mpiCartShape = sp.array(mpiCartShape)
    
            self.imgShape = mpiCartShape*subdShape
    
        def testSphericalHistogramPopulate(self):
            #tmpDir = "."
            tmpDir = self.createTmpDir("testSphericalHistogramPopulate")
    
            img, c, r = createPhantom(self.imgShape, voidFill=mango.mtype("tomo").maskValue())
            
            mango.io.writeDds(os.path.join(tmpDir, "tomoPhantomHead.nc"), img)
    
            sphhist = mango.image.spherical_histogram(sphere_c=c-img.origin, sphere_r=1)
            mango.image.intensity_spherical_histogram(img, sphhist)
    
            sphhist.reduce(img.mpi.comm)
            if (mango.haveVTK):
                try:
                    vtkUnstrGrid = sphhist.asVtkUnstructuredGrid()
                    rootLogger.info(str(vtkUnstrGrid))
                except Exception as e:
                    rootLogger.info(str(e))
                sphhist.writeVtkXMLUnstructuredGridToFile(os.path.join(tmpDir, "sphhist_intensity.vtu"), img.mpi.comm, 0)
            self.assertEqual(mango.sum(img), sphhist.getNumSamples())
    
            sphhist = mango.image.spherical_histogram(sphere_c=c-img.origin, sphere_r=1)
            mango.image.distance_spherical_histogram(img, sphhist)
    
            sphhist.reduce(img.mpi.comm)
            if (mango.haveVTK):
                try:
                    vtkUnstrGrid = sphhist.asVtkUnstructuredGrid()
                    rootLogger.info(str(vtkUnstrGrid))
                except Exception as e:
                    rootLogger.info(str(e))
                sphhist.writeVtkXMLUnstructuredGridToFile(os.path.join(tmpDir, "sphhist_distance.vtu"), img.mpi.comm, 0)

            sphhist = mango.image.spherical_histogram(sphere_c=c-img.origin, sphere_r=1)
            mango.image.intensity_mult_distance_spherical_histogram(img, sphhist)
    
            sphhist.reduce(img.mpi.comm)
            if (mango.haveVTK):
                try:
                    vtkUnstrGrid = sphhist.asVtkUnstructuredGrid()
                    rootLogger.info(str(vtkUnstrGrid))
                except Exception as e:
                    rootLogger.info(str(e))
                sphhist.writeVtkXMLUnstructuredGridToFile(os.path.join(tmpDir, "sphhist_intensity_mult_distance.vtu"), img.mpi.comm, 0)
    
if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
