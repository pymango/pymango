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

class MaxCoveringRadiusTest(mango.unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((64,80,71))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape
        self.radius = (np.min(self.imgShape)-1)//2
        self.centre = 0.5*self.imgShape

    def doTestMcrWithHalo(self, haloSz=0, filecache=False):
        if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
            if (haloSz < 0):
                haloSz = 0
            haloSz = sp.array((haloSz,)*3)
        
        subDirName = "doTestMcrWithHalo_%s_%s" % ("x".join(map(str, haloSz)), str(filecache))
        outDir = self.createTmpDir(subDirName)
        # outDir = subDirName
        if (mpi.world != None):
            if ((mpi.world.Get_rank() == 0) and (not os.path.exists(outDir))):
                os.makedirs(subDirName)
            mpi.world.barrier()

        segDds = mango.zeros(self.imgShape, mtype="segmented", halo=haloSz)
        segDds.setAllToValue(segDds.mtype.maskValue())
        mango.data.fill_ellipsoid(segDds, centre=self.centre, radius=(self.radius*1.05,)*3, fill=0)
        mango.data.fill_ellipsoid(segDds, centre=self.centre, radius=(self.radius,)*3, fill=1)
        mango.io.writeDds(os.path.join(outDir,"segmentedSphere.nc"), segDds)
        dtDds = mango.image.distance_transform_edt(segDds, 1)
        mango.io.writeDds(os.path.join(outDir,"distance_mapSphereEdt.nc"), dtDds)
        mcrDds = mango.image.max_covering_radius(dtDds, maxdist=self.radius*1.5, filecache=filecache)
        mango.io.writeDds(os.path.join(outDir,"distance_mapSphereEdtMcr.nc"), mcrDds)
        
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], segDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)

        self.assertTrue(
            sp.all(
                (
                    self.radius
                    -
                    sp.where(
                        sp.logical_and(mcrDds.asarray() > 0, mcrDds.asarray() != mcrDds.mtype.maskValue()),
                        sp.ceil(mcrDds.asarray()),
                        self.radius
                    )
                )
                <=
                0.1*self.radius # Test is true in the masked areas.
            )
        )
        
        self.assertTrue(sp.all(segDds.halo == mcrDds.halo))
        self.assertTrue(sp.all(segDds.shape == mcrDds.shape))
        self.assertTrue(sp.all(segDds.origin == mcrDds.origin), "%s != %s" % (segDds.origin, mcrDds.origin))
        self.assertTrue(sp.all(segDds.mpi.shape == mcrDds.mpi.shape))

    def testMcrRamOnly(self):
        self.doTestMcrWithHalo(0)

    def testMcrCacheFiles(self):
        self.doTestMcrWithHalo(0, True)


if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
