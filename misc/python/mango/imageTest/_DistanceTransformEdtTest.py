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

class DistanceTransformEdtTest(mango.unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((64,80,71))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape
        self.radius = (np.min(self.imgShape)-1)//2
        self.centre = 0.5*self.imgShape

    def doTestEdtWithHalo(self, haloSz=0):
        if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
            if (haloSz < 0):
                haloSz = 0
            haloSz = sp.array((haloSz,)*3)
        
        subDirName = "doTestEdtWithHalo_%s" % ("x".join(map(str, haloSz)), )
        outDir = self.createTmpDir(subDirName)
        #outDir = subDirName
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
        segDds.updateHaloRegions()
        dtDds.updateHaloRegions()
        segDds.setBorderToValue(segDds.mtype.maskValue())
        dtDds.setBorderToValue(dtDds.mtype.maskValue())
        
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], segDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)

        arr = dtDds.subd_h.asarray()
        sbeg = dtDds.subd_h.origin
        send = sbeg + dtDds.subd_h.shape
        
        coords = np.ogrid[sbeg[0]:send[0],sbeg[1]:send[1],sbeg[2]:send[2]]
    
        distDds = mango.copy(dtDds)
        distArr = distDds.subd_h.asarray()
        distArr[...] = \
            sp.where(
                sp.logical_and(dtDds.subd_h.asarray() != dtDds.mtype.maskValue(), dtDds.subd_h.asarray() >= 0),
                self.radius
                -
                sp.sqrt(
                    ((coords[0]-self.centre[0])**2)
                    +
                    ((coords[1]-self.centre[1])**2)
                    +
                    ((coords[2]-self.centre[2])**2)
                ),
                dtDds.subd_h.asarray()
            )
        mango.io.writeDds(os.path.join(outDir,"distance_mapSphereDistRef.nc"), distDds)
    
        rootLogger.info("max diff = %s" % (np.max(sp.absolute(distArr - dtDds.subd_h.asarray())),))
        self.assertTrue(
            sp.all(
                sp.absolute(distArr - dtDds.subd_h.asarray())
                <=
                1.0
            )
        )
        self.assertTrue(
            sp.all(
                sp.logical_not(sp.logical_xor(
                    segDds.asarray() == 1,
                    dtDds.asarray()  >=  0
                ))
            )
        )

        self.assertTrue(
            sp.all(
                sp.logical_not(sp.logical_xor(
                    segDds.asarray() == segDds.mtype.maskValue(),
                    dtDds.asarray()  == dtDds.mtype.maskValue()
                ))
            )
        )

        self.assertTrue(
            sp.all(
                sp.logical_not(sp.logical_xor(
                    segDds.asarray() == 0,
                    dtDds.asarray()  == -1
                ))
            )
        )
        
        self.assertTrue(sp.all(segDds.halo == dtDds.halo))
        self.assertTrue(sp.all(segDds.shape == dtDds.shape))
        self.assertTrue(sp.all(segDds.origin == dtDds.origin), "%s != %s" % (segDds.origin, dtDds.origin))
        self.assertTrue(sp.all(segDds.mpi.shape == dtDds.mpi.shape))

    def testEdtHalo0(self):
        self.doTestEdtWithHalo(0)

    def testEdtHalo1(self):
        self.doTestEdtWithHalo(1)

    def testEdtHalo4(self):
        self.doTestEdtWithHalo(4)

if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
