#!/usr/bin/env python
import logging
import sys
import mango.unittest
import scipy as sp
import numpy as np
import mango.mpi as mpi
import mango.image
import mango.data
import mango.io

logger, rootLogger = mpi.getLoggers(__name__)

class MomentOfInertiaTest(mango.unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((46,64,80))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def testCentreOfMass(self):
        img = mango.zeros(shape=self.imgShape, mtype="tomo", origin=(20,11,5))
        c = (sp.array(img.origin) + img.origin + img.shape-1)*0.5
        r = sp.array(img.shape-1)*0.5
        
        mango.data.fill_ellipsoid(img, centre=c, radius=r, fill=512, coordsys="abs")
        # mango.io.writeDds("tomoComEllipsoid.nc", img)
        
        com = mango.image.centre_of_mass(img)
        rootLogger.info("c = %s, com = %s" % (c, com))
        self.assertTrue(sp.all(sp.absolute(c - com) <= 1.0e-10))

    def testMomentOfInertiaAxisAlignedEllipsoid(self):
        img = mango.zeros(shape=self.imgShape*2, mtype="tomo", origin=(20,11,5))
        c = (sp.array(img.origin) + img.origin + img.shape-1)*0.5
        r = sp.array(img.shape-1)*0.25
        
        mango.data.fill_ellipsoid(img, centre=c, radius=r, fill=512, coordsys="abs")
        # mango.io.writeDds("tomoMoiEllipsoid.nc", img)
        
        pmoi, pmoi_axes, com = mango.image.moment_of_inertia(img)
        rootLogger.info("pmoi = \n%s" % (pmoi,))
        rootLogger.info("pmoi_axes = \n%s" % (pmoi_axes,))
        rootLogger.info("c = %s, com = %s" % (c, com))
        self.assertTrue(sp.all(sp.absolute(c - com) <= 1.0e-10))
        self.assertLess(pmoi[0], pmoi[1])
        self.assertLess(pmoi[1], pmoi[2])
        self.assertTrue(sp.all(sp.absolute(pmoi_axes[:,0]-(0,0,1)) <= 1.0e-10))
        self.assertTrue(sp.all(sp.absolute(pmoi_axes[:,1]-(0,1,0)) <= 1.0e-10))
        self.assertTrue(sp.all(sp.absolute(pmoi_axes[:,2]-(1,0,0)) <= 1.0e-10))

    if (mango.haveRegistration):
        def testMomentOfInertiaRotatedEllipsoid(self):
            img = mango.zeros(shape=self.imgShape*2, mtype="tomo", origin=(0,0,0))
            img.md.setVoxelSize((1,1,1))
            img.md.setVoxelSizeUnit("mm")
            c = (sp.array(img.origin) + img.origin + img.shape-1)*0.5
            r = sp.array(img.shape-1)*0.25
            
            mango.data.fill_ellipsoid(img, centre=c, radius=r, fill=512)
            rMatrix = \
                (
                    mango.image.rotation_matrix(-25, 2).dot(
                    mango.image.rotation_matrix( 10, 1).dot(
                    mango.image.rotation_matrix( 45, 0)
                    ))
                )
            img = mango.image.affine_transform(img, rMatrix, offset=c-img.origin, interptype=mango.image.InterpolationType.CATMULL_ROM_CUBIC_SPLINE)
            #mango.io.writeDds("tomoMoiRotEllipsoid.nc", img)
            
            pmoi, pmoi_axes, com = mango.image.moment_of_inertia(img)
            rootLogger.info("rmtx = \n%s" % (rMatrix,))
            rootLogger.info("pmoi = \n%s" % (pmoi,))
            rootLogger.info("pmoi_axes = \n%s" % (pmoi_axes,))
            rootLogger.info("c = %s, com = %s" % (c, com))
            self.assertTrue(sp.all(sp.absolute(c - com) <= 1.0e-10))
            self.assertLess(pmoi[0], pmoi[1])
            self.assertLess(pmoi[1], pmoi[2])
            self.assertTrue(sp.all(sp.absolute(pmoi_axes[:,0]-rMatrix[:,2]) <= 1.0e-3))
            self.assertTrue(sp.all(sp.absolute(pmoi_axes[:,1]-rMatrix[:,1]) <= 1.0e-3))
            self.assertTrue(sp.all(sp.absolute(pmoi_axes[:,2]-rMatrix[:,0]) <= 1.0e-3))
    
if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
