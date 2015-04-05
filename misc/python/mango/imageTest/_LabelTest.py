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

class LabelTest(mango.unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((64,64,64))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def doTestLabelWithHalo(self, haloSz=0):
        if (isinstance(haloSz, int) or ((sys.version_info.major < 3) and isinstance(haloSz, long))):
            if (haloSz < 0):
                haloSz = 0
            haloSz = sp.array((haloSz,)*3)
        
        checkShape = (30,40,50)
        imgDds = mango.data.createCheckerDds(shape=self.imgShape, checkShape=(30,40,50), black=0, white=1, mtype="segmented", halo=haloSz)
        
        slc = []
        for d in range(len(haloSz)):
            slc.append(slice(haloSz[d], imgDds.asarray().shape[d]-haloSz[d]))
        
        slc = tuple(slc)
        
        labelDds = mango.image.label(imgDds, 1)
        dir = self.createTmpDir("doTestLabelWithHalo")
        #dir = "."
        mango.io.writeDds(os.path.join(dir, "segmentedChkBrd.nc"), imgDds)
        mango.io.writeDds(os.path.join(dir, "labelsLblChkBrd.nc"), labelDds)
        
        lblMtype = mango.mtype("labels")
        lblDtype = lblMtype.dtype
        
        self.assertEqual(labelDds.dtype, lblDtype, "%s != %s" % (imgDds.dtype, lblDtype))
        self.assertEqual(labelDds.mtype, lblMtype, "%s != %s" % (imgDds.mtype, lblMtype))
        self.assertTrue(sp.all(imgDds.halo == labelDds.halo))
        self.assertTrue(sp.all(imgDds.shape == labelDds.shape))
        self.assertTrue(sp.all(imgDds.origin == labelDds.origin), "%s != %s" % (imgDds.origin, labelDds.origin))
        self.assertTrue(sp.all(imgDds.mpi.shape == labelDds.mpi.shape))

        logger.info("imgDds min   = %s, imgDds max   = %s" % (np.min(imgDds.asarray()), np.max(imgDds.asarray())))
        logger.info("labelDds min = %s, labelDds max = %s" % (np.min(labelDds.asarray()[slc]), np.max(labelDds.asarray()[slc])))
        logger.info("num non-zero labelDds = %s" % sp.sum(sp.where(labelDds.asarray()[slc] != 0, 1, 0)))
        
        uniqueLbls = sp.unique(labelDds.asarray())
        if (labelDds.mpi.comm != None):
            uLbls = []
            for uLbl in labelDds.mpi.comm.allgather(uniqueLbls.tolist()):
                uLbls += uLbl
            rootLogger.info("uLbls = %s" % str(uLbls))
            uniqueLbls = sp.unique(sp.array(uLbls, dtype=uniqueLbls.dtype))
        rootLogger.info("Labels = %s" % str(uniqueLbls))
        self.assertEqual(sp.product(imgDds.shape//checkShape + 1)//2, uniqueLbls.size-1)

        
    def testLabelWithHalo0(self):
        self.doTestLabelWithHalo(0)

    def testLabelWithHalo1(self):
        self.doTestLabelWithHalo(1)

    def testLabelWithHalo2(self):
        self.doTestLabelWithHalo(2)

class EliminateLabelsBySizeTest(mango.unittest.TestCase):
    def setUp(self):
        subdShape = sp.array((96,101,97))
        mpiCartShape = mango.mpi.getCartShape(dimension=3)
        mpiCartShape = sp.array(mpiCartShape)

        self.imgShape = mpiCartShape*subdShape

    def testElimateLabelsBySize(self):
        outDir = self.createTmpDir("testElimateLabelsBySize")
        #outDir = "."
        segDds = mango.zeros(shape=self.imgShape, mtype="segmented")
        c0 = segDds.shape * 0.25
        r0 = [0.5*0.25*(np.min(segDds.shape)-3),]*3
        c1 = segDds.shape * 0.75
        r1 = [0.5*r0[0],]*3
        mango.data.fill_ellipsoid(segDds, centre=c0, radius=r0, fill=1)
        mango.data.fill_ellipsoid(segDds, centre=c1, radius=r1, fill=1)
        
        mango.io.writeDds(os.path.join(outDir, "segmentedSpheres.nc"), segDds)
        
        lblDds = mango.image.label(segDds, val=1)
        mango.io.writeDds(os.path.join(outDir, "labelsSpheres.nc"), lblDds)

        lbl,cnt = mango.itemfreq(lblDds)
        lbl = sp.array(lbl, dtype=lblDds.dtype)
        cnt = sp.array(cnt, dtype="int32")
        logger.info("lbl,cnt = %s, %s" % (lbl,cnt))
        self.assertEqual(3, lbl.size)
        self.assertEqual(3, cnt.size)
        self.assertEqual(0, lbl[0])
        
        if (cnt[1] < cnt[2]):
            mni = 1
            mxi = 2
        else:
            mni = 2
            mxi = 1
        
        elblDds = mango.image.eliminate_labels_by_size(lblDds, int(cnt[mni]-1), int(cnt[mxi]-1), lblDds.mtype.maskValue(), labels_are_connected=True)
        elbl,ecnt = mango.itemfreq(elblDds)
        elbl = sp.array(elbl, dtype=lblDds.dtype)
        ecnt = sp.array(ecnt, dtype="int32")
        logger.info("elbl,ecnt = %s, %s" % (elbl,ecnt))
        self.assertEqual(2, elbl.size)
        self.assertEqual(2, ecnt.size)
        self.assertEqual(0, elbl[0])
        self.assertEqual(cnt[0], ecnt[0])
        self.assertEqual(lbl[mxi], elbl[1])
        self.assertEqual(cnt[mxi], ecnt[1])
        self.assertEqual(cnt[mni], mango.count_masked(elblDds))
        
        
        
if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.core", "mango.io", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
