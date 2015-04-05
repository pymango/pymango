#!/usr/bin/env python
import logging
import sys
import os
import os.path
import mango.unittest
import scipy as sp
import numpy as np
import mango.mpi as mpi
import mango.image
import mango.io
import mango.data

logger, rootLogger = mpi.getLoggers(__name__)

if (mango.haveCGAL):
    
    class ConvexHullTest(mango.unittest.TestCase):
        def setUp(self):
            subdShape = sp.array((64,80,71))
            mpiCartShape = mango.mpi.getCartShape(dimension=3)
            mpiCartShape = sp.array(mpiCartShape)
    
            self.imgShape = mpiCartShape*subdShape
            self.radius = (np.min(self.imgShape)-1)//2
            self.centre = 0.5*self.imgShape
    
        def doTestConvexHullWithHalo(self, haloSz=0):
            if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
                if (haloSz < 0):
                    haloSz = 0
                haloSz = sp.array((haloSz,)*3)
            
            subDirName = "doTestConvexHullWithHalo_%s" % ("x".join(map(str, haloSz)),)
            outDir = self.createTmpDir(subDirName)
            #outDir = "."
    
            segDds = mango.zeros(self.imgShape, mtype="segmented", halo=haloSz)
            segDds.setAllToValue(segDds.mtype.maskValue())
            mango.data.fill_ellipsoid(segDds, centre=self.centre, radius=(self.radius*1.125,)*3, fill=0)
            mango.data.fill_ellipsoid(segDds, centre=self.centre, radius=(self.radius,)*3, fill=1)
            mango.io.writeDds(os.path.join(outDir,"segmentedSphere.nc"), segDds)
            for axis in (0,1,2, "3d"):
                if (axis == "3d"):
                    chDds = mango.image.convex_hull_3d(segDds, inputmsk=0, inhull=1, outhull=0)
                    fname = "segmentedSphereConvexHull3d.nc"
                else:
                    chDds = mango.image.convex_hull_2d(segDds, axis=0, inputmsk=0, inhull=1, outhull=0)
                    fname = "segmentedSphereConvexHull2d_axis%s.nc" % axis
                mango.io.writeDds(os.path.join(outDir,fname), chDds)
                segDtDds = mango.image.distance_transform_edt(segDds, 1)
                chDtDds = mango.image.distance_transform_edt(chDds, 1)
                
                segDds.updateHaloRegions()
                segDtDds.updateHaloRegions()
                chDtDds.updateHaloRegions()
                segDds.setBorderToValue(segDds.mtype.maskValue())
                segDtDds.setBorderToValue(segDtDds.mtype.maskValue())
                chDtDds.setBorderToValue(chDtDds.mtype.maskValue())
    
    
                
                slc = []
                for d in range(len(haloSz)):
                    slc.append(slice(haloSz[d], segDds.asarray().shape[d]-haloSz[d]))
                
                slc = tuple(slc)
        
                self.assertTrue(
                    sp.all(
                        (
                            sp.where(
                                segDtDds.asarray() >= 0,
                                sp.absolute(segDtDds.asarray() - chDtDds.asarray()),
                                0
                            )
                        )
                        <=
                        2
                    )
                )
    
                self.assertTrue(
                    sp.all(
                        (
                            sp.where(
                                sp.logical_and(segDtDds.asarray() == -1, chDtDds.asarray() >= 0),
                                chDtDds.asarray(),
                                0
                            )
                        )
                        <=
                        2
                    )
                )
                
                self.assertTrue(sp.all(segDds.halo == chDds.halo))
                self.assertTrue(sp.all(segDds.shape == chDds.shape))
                self.assertTrue(sp.all(segDds.origin == chDds.origin), "%s != %s" % (segDds.origin, chDds.origin))
                self.assertTrue(sp.all(segDds.mpi.shape == chDds.mpi.shape))

        def testConvexHullHalo0(self):
            self.doTestConvexHullWithHalo(0)

        def testConvexHullHalo2(self):
            self.doTestConvexHullWithHalo(2)

        def testConvexHull3dOnly(self):
            
            subDirName = "doTestConvexHull3dOnly"
            outDir = self.createTmpDir(subDirName)
    
            segDds = mango.zeros(self.imgShape, mtype="segmented", halo=0)
            segDds.setAllToValue(segDds.mtype.maskValue())
            centre = (segDds.origin+segDds.shape)*0.5
            shp0 = segDds.shape * 0.90
            shp0[1] *= 0.25
            org0 = centre - shp0*0.5
            shp1 = segDds.shape * 0.90
            shp1[2] *= 0.25
            org1 = centre - shp1*0.5
            mango.data.fill_box(segDds, centre=centre, shape=shp0, fill=1, coordsys="abs")
            mango.data.fill_box(segDds, centre=centre, shape=shp1, fill=1, coordsys="abs")
            mango.io.writeDds(os.path.join(outDir,"segmentedCross.nc"), segDds)
            segDds = mango.io.readDds(os.path.join(outDir,"segmentedCross.nc"))
            chDds = mango.image.convex_hull_3d(segDds, inputmsk=0, inhull=1, outhull=0)
            fname = "segmentedCrossConvexHull3d.nc"
            mango.io.writeDds(os.path.join(outDir,fname), chDds)


if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.io", "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.setLoggingVerbosityLevel("high")
    mango.unittest.main()
