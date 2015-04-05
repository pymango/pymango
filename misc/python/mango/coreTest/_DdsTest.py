#!/usr/bin/env python
import logging
import mango.unittest
import scipy as sp
import numpy as np
import numpy.random
import mango.mpi as mpi
import mango
import os
import os.path

logger, rootLogger = mpi.getLoggers(__name__)

class DdsTest(mango.unittest.TestCase):
    def setUp(self):
        dSz = 4
        if mango.mpi.haveMpi4py:
            dSz = 4*mango.mpi.world.Get_size()
        self.shape = (dSz, 2*dSz, 3*dSz)

    def testDdsClassAttr(self):
        dds = mango.zeros(shape=self.shape)
        rootLogger.info(str(dir(dds)))
        self.assertTrue(hasattr(dds.__class__, "dtype"))
        self.assertTrue(hasattr(dds.__class__, "__getitem__"))
        self.assertTrue(hasattr(dds.__class__, "__setitem__"))
        self.assertTrue(hasattr(dds.__class__, "fill"))
        self.assertTrue(hasattr(dds, "mtype"))
        self.assertTrue(hasattr(dds, "shape"))
        self.assertTrue(hasattr(dds, "origin"))
        self.assertTrue(hasattr(dds, "halo"))
        self.assertTrue(hasattr(dds, "subd"))
        self.assertTrue(hasattr(dds, "mpi"))
        self.assertTrue(hasattr(dds, "md"))

    def testDdsCreateComplex(self):
        """
        Test the :func:`mango.empty` and :func:`mango.ones`
        and :func:`mango.zeros` methods for dtype=complex64/complex128.
        """

        dds = mango.empty(shape=self.shape, dtype="complex64")
        cVal = np.complex64(2 + 1j*4)
        dds.setAllToValue(cVal)
        self.assertEqual(cVal, dds[0])
        self.assertEqual(cVal, dds[0,0,0])
        self.assertEqual(cVal, dds[(0,0,0)])

        dds = mango.zeros(shape=self.shape, dtype="complex64")
        self.assertEqual(np.complex64(0), dds[0])
        self.assertEqual(np.complex64(0), dds[0,0,0])
        self.assertEqual(np.complex64(0), dds[(0,0,0)])

        dds = mango.ones(shape=self.shape, dtype="complex64")
        self.assertEqual(np.complex64(1), dds[0])
        self.assertEqual(np.complex64(1), dds[0,0,0])
        self.assertEqual(np.complex64(1), dds[(0,0,0)])

        dds = mango.empty(shape=self.shape, dtype="complex128")
        cVal = np.complex128(2 + 1j*4)
        dds.setAllToValue(cVal)
        self.assertEqual(cVal, dds[0])
        self.assertEqual(cVal, dds[0,0,0])
        self.assertEqual(cVal, dds[(0,0,0)])

        dds = mango.zeros(shape=self.shape, dtype="complex128")
        self.assertEqual(np.complex128(0), dds[0])
        self.assertEqual(np.complex128(0), dds[0,0,0])
        self.assertEqual(np.complex128(0), dds[(0,0,0)])

        dds = mango.ones(shape=self.shape, dtype="complex128")
        self.assertEqual(np.complex128(1), dds[0])
        self.assertEqual(np.complex128(1), dds[0,0,0])
        self.assertEqual(np.complex128(1), dds[(0,0,0)])

    def testDdsCreateFloat16(self):
        """
        Test the :func:`mango.empty` and :func:`mango.ones`
        and :func:`mango.zeros` methods for :samp:`dtype="float16"`
        and :samp:`mtype="float16"`.
        """
        if (mango.haveFloat16):
            dds = mango.empty(shape=self.shape, dtype="float16")
            fVal = np.float16(2.25)
            dds.setAllToValue(fVal)
            self.assertEqual(fVal, dds[0])
            self.assertEqual(fVal, dds[0,0,0])
            self.assertEqual(fVal, dds[(0,0,0)])

            dds = mango.zeros(shape=self.shape, mtype="float16")
            fVal = np.float16(2.25)
            dds.asarray()[...] += fVal
            self.assertEqual(fVal, dds[0])
            self.assertEqual(fVal, dds[0,0,0])
            self.assertEqual(fVal, dds[(0,0,0)])

            dds = mango.ones(shape=self.shape, mtype="float16")
            fVal = np.float16(2.25)
            dds.asarray()[...] *= fVal
            self.assertEqual(fVal, dds[0])
            self.assertEqual(fVal, dds[0,0,0])
            self.assertEqual(fVal, dds[(0,0,0)])

    def testDdsCreate(self):
        """
        Test the :func:`mango.empty` and :func:`mango.ones`
        and :func:`mango.zeros` methods.
        """

        dds = mango.zeros(shape=self.shape, dtype="uint16")
        self.assertEqual(0, dds[0])
        self.assertEqual(0, dds[0,0,0])
        self.assertEqual(0, dds[(0,0,0)])

        dds = mango.ones(shape=self.shape, dtype="uint8")
        self.assertEqual(1, dds[0])
        self.assertEqual(1, dds[0,0,0])
        self.assertEqual(1, dds[(0,0,0)])

        self.assertEqual(tuple(self.shape), tuple(dds.shape))        
        self.assertEqual(None, dds.mtype)

        dds = mango.ones(shape=self.shape, mtype="tomo_float")
        self.assertEqual(tuple(self.shape), tuple(dds.shape))
        self.assertEqual(mango.mtype("tomo_float").dtype, dds.dtype)
        self.assertEqual(mango.mtype("tomo_float"), dds.mtype)

        dds = mango.ones(shape=self.shape, mtype="tomo_float", origin=(20,50,-100))
        elDds = mango.empty_like(dds)
        self.assertTrue(sp.all(elDds.origin == dds.origin))
        self.assertTrue(sp.all(elDds.shape == dds.shape))
        self.assertTrue(sp.all(elDds.subd.origin == dds.subd.origin))
        self.assertTrue(sp.all(elDds.subd.shape == dds.subd.shape))

        elDds = mango.empty_like(dds, origin = dds.origin + (10,20,40))
        self.assertTrue(sp.all(elDds.origin == dds.origin + (10,20,40)))
        self.assertTrue(sp.all(elDds.shape == dds.shape))
        self.assertTrue(sp.all(elDds.subd.origin == dds.subd.origin + (10,20,40)))
        self.assertTrue(sp.all(elDds.subd.shape == dds.subd.shape))
        self.assertNotEqual(dds.mtype, None)
        self.assertEqual(dds.mtype, elDds.mtype)

    def testDdsCreateWithSubdInfo(self):
        """
        Test the :func:`mango.empty` using the :samp:`subdorigin`
        and :samp:`subdshape` arguments.
        """

        # Create initial fixture.
        ddsOrig = \
            mango.empty(
                shape=(self.shape[0]+3, self.shape[1]+2, self.shape[0]+1),
                origin=(-32,-64,-128),
                dtype="uint16"
            )
        np.random.seed((mango.mpi.rank+1)*975421)
        ddsOrig.asarray()[:] = \
            np.random.randint(
                np.iinfo(ddsOrig.dtype).min,
                np.iinfo(ddsOrig.dtype).max,
                ddsOrig.asarray().shape
            )
        
        subdZSz = ddsOrig.shape[0]//mango.mpi.size
        mySubdZSz = subdZSz
        if (mango.mpi.rank == mango.mpi.size-1):
            mySubdZSz += (ddsOrig.shape[0] % mango.mpi.size)
        
        mpidims = sp.array((mango.mpi.size, 1, 1))
        subdshape = sp.array((mySubdZSz, ddsOrig.shape[1], ddsOrig.shape[2]))
        subdorigin = ddsOrig.origin + sp.array((subdZSz,subdshape[1],subdshape[2]))*(mango.mpi.rank, 0, 0)
        ddsSubd =\
            mango.empty(
                shape       = ddsOrig.shape,
                origin      = ddsOrig.origin,
                dtype       = ddsOrig.dtype,
                subdshape   = subdshape,
                subdorigin  = subdorigin,
                mpidims     = mpidims
            )
        self.assertTrue(sp.all(ddsSubd.origin == ddsOrig.origin))
        self.assertTrue(sp.all(ddsSubd.shape == ddsOrig.shape))
        self.assertTrue(sp.all(ddsSubd.subd.origin == subdorigin))
        self.assertTrue(sp.all(ddsSubd.subd.shape == subdshape))
        
        if (mpi.haveMpi4py):
            self.assertEqual(
                mango.mpi.world.allreduce(ddsOrig.asarray().size),
                mango.mpi.world.allreduce(ddsSubd.asarray().size)
            )
        else:
            self.assertEqual(
                ddsOrig.asarray().size,
                ddsSubd.asarray().size
            )
        
        logger.info("ddsOrig.origin=%s, ddsOrig.subd.origin=%s" % (ddsOrig.origin, ddsOrig.subd.origin))
        logger.info("ddsSubd.origin=%s, ddsSubd.subd.origin=%s" % (ddsSubd.origin, ddsSubd.subd.origin))
        ddsSubd.fill(ddsOrig)
        ddsCmp = mango.zeros_like(ddsOrig)
        ddsCmp.fill(ddsSubd)
        self.assertTrue(sp.all(ddsOrig.asarray() == ddsCmp.asarray()))

    def testDdsIndexingSetAndGet(self):
        """
        Test the :meth:`Dds.__getitem__` and :meth:`Dds.__setitem__` methods.
        """

        dds = mango.zeros(shape=self.shape, dtype="uint16")
        dds[0,0,0] = 5
        self.assertEqual(5, dds[0])
        self.assertEqual(5, dds[0,0,0])
        self.assertEqual(5, dds[(0,0,0)])

        dds = mango.ones(shape=self.shape, dtype="uint8")
        self.assertEqual(1, dds[0])
        self.assertEqual(1, dds[0,0,0])
        self.assertEqual(1, dds[(0,0,0)])

    def testDdsCopyCreate(self):
        """
        Test the :func:`mango.copy` method.
        """

        dds0 = mango.ones(shape=self.shape, dtype="uint8")
        dds1 = mango.copy(dds0, dtype="uint16")
        self.assertEqual(dds0[0], dds1[0])
        self.assertEqual(dds0[0,0,0], dds1[0,0,0])
        self.assertEqual(dds0[(0,0,0)], dds1[(0,0,0)])

    def testDdsSetGlobalOrigin(self):
        """
        Test the :attr:`mango.Dds.origin` attribute for read/write access.
        """
        origin = (5,3,1)
        dds0 = mango.ones(shape=self.shape, dtype="uint8", origin=origin)
        self.assertTrue(sp.all(origin == dds0.origin))
        dds1 = mango.copy(dds0, dtype="uint16")
        self.assertTrue(sp.all(origin == dds1.origin))
        
        newOrigin = sp.array((8,16,2), dtype="int32")
        dds0.origin = newOrigin
        self.assertTrue(sp.all(newOrigin == dds0.origin))
        self.assertTrue(sp.all(dds1.subd.origin+(newOrigin-origin) == dds0.subd.origin))

    def testDdsSetGlobalFaceValue(self):
        """
        Test the :meth:`mango.Dds.setFacesToValue` method.
        """
        outDir = self.createTmpDir("testDdsSetGlobalFaceValue")
        ddsorig = mango.zeros(shape=self.shape, mtype="tomo", origin=(-8,5,-32))
        for depth in (1,int(np.min(ddsorig.shape)//2)):
            for axis in (0,1,2):
                dds = mango.copy(ddsorig)
                vallo = 16+axis
                rootLogger.info("LO: axis=%s, val=%s, depth=%s" % (axis, vallo, depth))
                dds.setFacesToValue(vallo, axislo=axis, depth=depth)
                #mango.io.writeDds(os.path.join(outDir, "tomoLoFaceAxis%s.nc" % axis), dds)
                shpFace = dds.shape
                shpFace[axis] = depth
                mpidimsFace = [0,0,0]
                mpidimsFace[axis] = 1
                rootLogger.info("LO: shpFace=%s, mpidimsFace=%s" % (shpFace, mpidimsFace))
                ddsFace = mango.copy(dds, shape=shpFace, mpidims=mpidimsFace)
                self.assertTrue(sp.all(ddsFace.subd.asarray() == vallo))
                shpNotFace = dds.shape
                shpNotFace[axis] = dds.shape[axis] - depth
                orgNotFace = dds.origin
                orgNotFace[axis] = dds.origin[axis] + depth

                mpidimsNotFace = [0,0,0]
                mpidimsNotFace[axis] = 1
                rootLogger.info("LO: shpNotFace=%s, mpidimsNotFace=%s, orgNotFace=%s" % (shpNotFace, mpidimsNotFace,orgNotFace))
                ddsNotFace = mango.copy(dds, origin=orgNotFace, shape=shpNotFace, mpidims=mpidimsNotFace)
                self.assertTrue(sp.all(ddsNotFace.subd.asarray() == 0))

                dds = mango.copy(ddsorig)
                valhi = 2*vallo
                rootLogger.info("HI: axis=%s, val=%s, depth=%s" % (axis, valhi, depth))
                dds.setFacesToValue(valhi, axishi=axis, depth=depth)
                #mango.io.writeDds(os.path.join(outDir, "tomoLoFaceAxis%s.nc" % axis), dds)
                shpFace = dds.shape
                shpFace[axis] = depth
                mpidimsFace = [0,0,0]
                mpidimsFace[axis] = 1
                orgFace = dds.origin
                orgFace[axis] = dds.origin[axis] + dds.shape[axis]-depth
                rootLogger.info("HI: shpFace=%s, mpidimsFace=%s, orgFace=%s" % (shpFace, mpidimsFace, orgFace))
                ddsFace = mango.copy(dds, origin=orgFace, shape=shpFace, mpidims=mpidimsFace)
                self.assertTrue(sp.all(ddsFace.subd.asarray() == valhi))
                shpNotFace = dds.shape
                shpNotFace[axis] = dds.shape[axis] - depth
                mpidimsNotFace = [0,0,0]
                mpidimsNotFace[axis] = 1
                rootLogger.info("HI: shpNotFace=%s, mpidimsNotFace=%s, orgNotFace=%s" % (shpNotFace, mpidimsNotFace,orgNotFace))
                ddsNotFace = mango.copy(dds, shape=shpNotFace, mpidims=mpidimsNotFace)
                self.assertTrue(sp.all(ddsNotFace.subd.asarray() == 0))

    def testDdsFill(self):
        """
        Test the :meth:`Dds.fill` method.
        """

        ddsDst = mango.zeros(shape=self.shape, dtype="uint16", mpidims=(0,1,1))
        self.assertEqual(0, ddsDst[0,0,0])

        ddsSrc = mango.ones(shape=self.shape, dtype="uint16", mpidims=(1,0,1))
        self.assertEqual(1, ddsSrc[0,0,0])

        ddsDst.fill(ddsSrc)
        self.assertEqual(1, ddsDst[0,0,0])

    def createTmpArrayFromDds(self):
        dds = mango.zeros(shape=self.shape, dtype="uint16", mpidims=(0,1,1))
        a = dds.asarray()
        del dds
        return a

    def testDdsMirrorOuterLayersToBorder(self):
        """
        Test the :meth:`Dds.mirrorOuterLayersToBorder` method.
        """
        dds = mango.zeros(shape=self.shape, dtype="uint16", mpidims=(0,1,1), halo=2)
        dds.setBorderToValue(1)
        self.assertTrue(sp.sum(sp.where(dds.asarray() != 0, 1, 0)) > 0)
        dds.mirrorOuterLayersToBorder(False)
        
        logger.info("Num non-zero = %s" % sp.sum(sp.where(dds.asarray() != 0, 1, 0)))
        logger.info("Idx non-zero = %s" % (sp.where(dds.asarray() != 0),))
        self.assertTrue(sp.all(dds.asarray() == 0))
        

    def testDdsAsArray(self):
        """
        Test the :meth:`Dds.asarray` method.
        """

        dds = mango.zeros(shape=self.shape, dtype="uint16", mpidims=(0,1,1))
        self.assertEqual(0, dds[0,0,0])

        a = dds.asarray()
        a += 10
        self.assertEqual(10, dds[0,0,0])
        a -= 10
        self.assertEqual(0, dds[0,0,0])
        
        a = dds.asarray()
        del dds
        a += 10
        self.assertEqual(10, a[0,0,0])

        a0 = self.createTmpArrayFromDds()
        self.assertEqual(0, a0[0,0,0])
        a0 += 16
        self.assertEqual(16, a0[0,0,0])
        
        a += 20
        self.assertEqual(30, a[0,0,0])

    def testDdsAsArrayMemoryDealloc(self):
        """
        Alloc and dealloc :obj:`Dds` objects and check memory
        consumption.
        """
        for i in range(0,16):
            rootLogger.info("Creating Dds...")
            dds = mango.zeros(shape=(256,512,512), dtype="uint16", mpidims=(0,0,0))
            rootLogger.info("Done creating Dds.")
            self.assertEqual(0, dds[0,0,0])
    
            a = dds.asarray()
            rootLogger.info("Deleting Dds...")
            del dds
            rootLogger.info("Done deleting Dds.")
            a += 10
            self.assertEqual(10, a[0,0,0])
            a -= 10
            self.assertEqual(0, a[0,0,0])

    def testDdsSubdProperty(self):
        """
        Test :prop:`Dds.subd` object.
        """
        dds = mango.zeros(shape=self.shape, dtype="uint16", mpidims=(0,1,1))
        self.assertEqual(0, dds[0,0,0])

        rootLogger.info("dds.shape=%s, dds.mpi.shape=%s" % (dds.shape, dds.mpi.shape))
        logger.info("dds.mpi.rank=%s, dds.mpi.index=%s" % (dds.mpi.rank, dds.mpi.index))
        logger.info("dds.subd.origin=%s, dds.subd.shape=%s" % (dds.subd.origin, dds.subd.shape))
        
        dds = mango.zeros(shape=self.shape, dtype="uint16", mpidims=(0,0,0))
        self.assertEqual(0, dds[0,0,0])

        rootLogger.info("dds.shape=%s, dds.mpi.shape=%s" % (dds.shape, dds.mpi.shape))
        logger.info("dds.mpi.rank=%s, dds.mpi.index=%s" % (dds.mpi.rank, dds.mpi.index))
        logger.info("dds.subd.origin=%s, dds.subd.shape=%s" % (dds.subd.origin, dds.subd.shape))

    def testDdsSubdAttributes(self):
        """
        Test :prop:`Dds.subd` and :prop:`Dds.subd_h` objects.
        """
        dds = mango.zeros(shape=self.shape, dtype="uint16", mpidims=(0,1,1))
        self.assertEqual(0, dds[0,0,0])
        dds[0,0,0] = 1
        self.assertEqual(1, dds[0,0,0])
        self.assertEqual(1, dds.subd[0,0,0])
        self.assertEqual(1, dds.subd_h[0,0,0])
        
        for h in (0, 1, 2, 3):
            for halo in ((h,h,h), (2*h,h//2,h), (h,2*h,h//2)):
                dds = mango.zeros(shape=self.shape, dtype="uint16", halo=halo)
                rootLogger.info("====")
                rootLogger.info("dds.halo                                = %s" % (dds.halo,))
                rootLogger.info("dds.subd.origin                         = %s" % (dds.subd.origin,))
                rootLogger.info("dds.subd.shape                          = %s" % (dds.subd.shape,))
                rootLogger.info("dds.subd.asarray().shape                = %s" % (dds.subd.asarray().shape,))
                rootLogger.info("dds.subd_h.asarray().shape              = %s" % (dds.subd_h.asarray().shape,))
                rootLogger.info("dds.subd_h.asarray().shape-(2*dds.halo) = %s" % (dds.subd_h.asarray().shape-(2*dds.halo),))
                rootLogger.info("dds.subd_h.origin                       = %s" % (dds.subd_h.origin,))
                rootLogger.info("dds.subd_h.shape                        = %s" % (dds.subd_h.shape,))

                self.assertTrue(sp.all(dds.subd.origin == (dds.subd_h.origin+dds.halo)))
                self.assertTrue(sp.all(dds.subd.shape == (dds.subd_h.shape-(2*dds.halo))))
                self.assertTrue(sp.all(dds.subd.asarray().shape == (dds.subd_h.asarray().shape-(2*dds.halo))))
                
                self.assertEqual(0, dds.subd_h.asarray()[halo[0],halo[1],halo[2]])
                self.assertEqual(0, dds.subd.asarray()[0,0,0])
                dds.subd.asarray()[0,0,0] = 1
                self.assertEqual(1, dds.subd.asarray()[0,0,0])
                self.assertEqual(1, dds.subd_h.asarray()[halo[0],halo[1],halo[2]])


    def testDdsCommunicator(self):
        """
        Test :prop:`Dds.mpi.comm` object.
        """
        dds = mango.zeros(shape=self.shape, dtype="uint16", mpidims=(0,1,1))
        self.assertEqual(0, dds[0,0,0])

        comm = dds.mpi.comm;
        if (mango.mpi.haveMpi4py):
            self.assertNotEqual(None, comm)
            rootLogger.info("dds.mpi.comm=%s" % (dds.mpi.comm,))
            self.assertEqual(mango.mpi.world.size, dds.mpi.comm.size)
            self.assertEqual(mango.mpi.world.rank, dds.mpi.comm.rank)
        else:
            self.assertEqual(None, comm)

    def testDdsMetaData(self):
        dds = mango.zeros(shape=self.shape, dtype="uint16")
        self.assertNotEqual(None, dds.md)
        rootLogger.info(
            "dds.md.getVoxelSize()=%s, dds.md.getVoxelSizeUnit()=%s"
            %
            (dds.md.getVoxelSize(), dds.md.getVoxelSizeUnit())
        )
        vSz = (0.5, 0.125, 0.25)
        dds.md.setVoxelSize(vSz)
        dds.md.setVoxelSizeUnit("um")
        rootLogger.info(
            "dds.md.getVoxelSize()=%s, dds.md.getVoxelSizeUnit()=%s"
            %
            (dds.md.getVoxelSize(), dds.md.getVoxelSizeUnit())
        )
        self.assertEqual("micrometre", dds.md.getVoxelSizeUnit())
        self.assertTrue(sp.all(dds.md.getVoxelSize() - vSz  < 1.0e-6))
        dds.md.setVoxelSize(1)
        dds.md.setVoxelSizeUnit("millimetre")
        rootLogger.info(
            "dds.md.getVoxelSize()=%s, dds.md.getVoxelSizeUnit()=%s"
            %
            (dds.md.getVoxelSize(), dds.md.getVoxelSizeUnit())
        )
        self.assertEqual("millimetre", dds.md.getVoxelSizeUnit())
        self.assertTrue(sp.all(dds.md.getVoxelSize() == 1))

    def testReductionMinMax(self):
        dds = mango.ones(shape=self.shape, dtype="uint16", halo=2)
        dds.setBorderToValue(0)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 2
        mn,mx = mango.minmax(dds)
        self.assertEqual(1, mn)
        self.assertEqual(2, mx)

        dds = mango.ones(shape=self.shape, mtype="tomo", halo=2)
        dds.setBorderToValue(0)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 2
        mn,mx = mango.minmax(dds)
        rootLogger.info("mn,mx = %s" % ((mn,mx),))
        self.assertEqual(1, mn)
        self.assertEqual(2, mx)
        
        dds = mango.ones(shape=self.shape, mtype="tomo", halo=2)
        dds.setBorderToValue(0)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 2
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx[0], centreIdx[1], centreIdx[2]+1] = dds.mtype.maskValue()
        mn,mx = mango.minmax(dds)
        rootLogger.info("mn,mx = %s" % ((mn,mx),))
        self.assertEqual(1, mn)
        self.assertEqual(2, mx)
        mn,mx = dds.minmax()
        self.assertEqual(1, mn)
        self.assertEqual(dds.mtype.maskValue(), mx)

    def testReductionSum(self):
        dds = mango.ones(shape=self.shape, dtype="uint16", halo=2)
        dds.setBorderToValue(100)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 2
        expectSum = sp.product(dds.shape)+2-1
        s = mango.sum(dds, dtype="uint64")
        self.assertEqual(expectSum, s)

        dds = mango.ones(shape=self.shape, mtype="tomo", halo=2)
        dds.setBorderToValue(100)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 9
        expectSum = sp.product(dds.shape)+9-1
        s = mango.sum(dds, dtype="uint64")
        self.assertEqual(expectSum, s)
        
        dds = mango.ones(shape=self.shape, mtype="tomo", halo=2)
        dds.setBorderToValue(0)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 9
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx[0], centreIdx[1], centreIdx[2]+1] = dds.mtype.maskValue()
        expectSum = sp.product(dds.shape)+9-1-1
        s = mango.sum(dds, dtype="uint64")
        self.assertEqual(expectSum, s)

    def testReductionSum2(self):
        dds = mango.ones(shape=self.shape, dtype="uint16", halo=2)
        dds.setBorderToValue(100)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 2
        expectSum = sp.product(dds.shape)+2*2-1
        s = mango.sum2(dds, dtype="uint64")
        self.assertEqual(expectSum, s)

        dds = mango.ones(shape=self.shape, mtype="tomo", halo=2)
        dds.setBorderToValue(100)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 9
        expectSum = sp.product(dds.shape)+9*9-1
        s = mango.sum2(dds, dtype="uint64")
        self.assertEqual(expectSum, s)
        
        dds = mango.ones(shape=self.shape, mtype="tomo", halo=2)
        dds.setBorderToValue(0)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 9
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx[0], centreIdx[1], centreIdx[2]+1] = dds.mtype.maskValue()
        expectSum = sp.product(dds.shape)+9*9-1-1
        s = mango.sum2(dds, dtype="uint64")
        self.assertEqual(expectSum, s)

    def testReductionCountMasked(self):
        dds = mango.ones(shape=self.shape, dtype="uint16", halo=2)
        dds.setBorderToValue(100)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = 2
        expectMskCount = 0
        expectNonMskCount = sp.product(dds.shape) - expectMskCount
        mc = mango.count_masked(dds)
        nmc = mango.count_non_masked(dds)
        self.assertEqual(expectMskCount, mc)
        self.assertEqual(expectNonMskCount, nmc)

        dds = mango.ones(shape=self.shape, mtype="tomo", halo=2)
        dds.setBorderToValue(100)
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = dds.mtype.maskValue()
        expectMskCount = 1
        expectNonMskCount = sp.product(dds.shape)-expectMskCount
        mc = mango.count_masked(dds)
        nmc = mango.count_non_masked(dds)
        self.assertEqual(expectMskCount, mc)
        self.assertEqual(expectNonMskCount, nmc)
        
        dds = mango.ones(shape=self.shape, mtype="tomo", halo=2)
        dds.setBorderToValue(dds.mtype.maskValue())
        centreIdx = tuple(sp.array(dds.subd.shape)//2)
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx] = dds.mtype.maskValue()
        if ((dds.mpi.comm == None) or (dds.mpi.comm.Get_rank() == 0)):
            dds[centreIdx[0], centreIdx[1], centreIdx[2]+1] = dds.mtype.maskValue()
        expectMskCount = 2
        expectNonMskCount = sp.product(dds.shape)-expectMskCount
        mc = mango.count_masked(dds)
        nmc = mango.count_non_masked(dds)
        self.assertEqual(expectMskCount, mc)
        self.assertEqual(expectNonMskCount, nmc)


    def testDdsResizeHalo(self):

        for ddsH in [(0,0,0), (4,4,4), [2,1,0], sp.array([3,6,7]), [6,9,8]]:
            for h in [1, sp.array([1,1,1]), sp.array([2,1,0]), [3,6,7]]:
                for s in [False, True]:
                    dds = \
                        mango.data.gaussian_noise(
                            shape=2*sp.array(self.shape),
                            mean = 32000.0,
                            stdd = 100.0,
                            mtype="tomo",
                            origin=(5,-11,7),
                            halo=ddsH
                        )
        
                    ddsResz = mango.copy(dds)
                    dds.updateHaloRegions()
                    dds.mirrorOuterLayersToBorder(True)
                    ddsResz.updateOverlapRegions()
                    ddsResz.mirrorOuterLayersToBorder(True)
                    
                    ddsResz.resizeHalo(h, s)
                    if (not hasattr(h, "__len__")):
                        h = sp.array([h,]*3)
                    h = sp.array(h)
                    self.assertListEqual(list(ddsResz.halo), list(h))
                    self.assertListEqual(list(dds.shape), list(ddsResz.shape))
                    self.assertListEqual(list(dds.origin), list(ddsResz.origin))
        
                    self.assertListEqual(list(dds.subd.shape), list(ddsResz.subd.shape))
                    self.assertListEqual(list(dds.subd.origin), list(ddsResz.subd.origin))
        
                    self.assertListEqual(list(dds.subd_h.shape-2*(dds.subd_h.halo-h)), list(ddsResz.subd_h.shape))
                    self.assertListEqual(list(dds.subd_h.origin+dds.subd_h.halo-h), list(ddsResz.subd_h.origin))
                
                    self.assertTrue(sp.all(dds.subd.asarray() == ddsResz.subd.asarray()))
    
                    slc = []
                    slcResz = []
                    for i in range(0,3):
                        if (dds.halo[i] > h[i]):
                            slc.append(slice(dds.halo[i]-h[i], -(dds.halo[i]-h[i])))
                            slcResz.append(slice(None))
                        elif (dds.halo[i] < h[i]):
                            slc.append(slice(None))
                            slcResz.append(slice(h[i]-dds.halo[i], -(h[i]-dds.halo[i])))
                        else:
                            slc.append(slice(None))
                            slcResz.append(slice(None))
    
                    self.assertListEqual(list(dds.subd_h.asarray()[slc].shape), list(ddsResz.subd_h.asarray()[slcResz].shape))
                    self.assertTrue(sp.all(dds.subd_h.asarray()[slc] == ddsResz.subd_h.asarray()[slcResz]))
            

class DdsUtilsTest(mango.unittest.TestCase):
    def setUp(self):
        dSz = 4
        if mango.mpi.haveMpi4py:
            dSz = 4*mango.mpi.world.Get_size()
        self.shape = (dSz, 2*dSz, 3*dSz)
    
    def testAllGather(self):
        dds = mango.ones(shape=self.shape, mtype="tomo", origin=(5,11,7), halo=2)
        
        allDds = mango.gather(dds)
        self.assertTrue(sp.all(dds.shape  == allDds.shape))
        self.assertTrue(sp.all(dds.origin == allDds.origin))
        self.assertTrue(sp.all(dds.halo   == allDds.halo))
        self.assertTrue(sp.all(dds.shape  == allDds.subd.shape))
        self.assertTrue(sp.all(dds.origin == allDds.subd.origin))
        self.assertTrue(sp.all(dds.halo   == allDds.subd.halo))

        self.assertEqual(dds.dtype, allDds.dtype)
        self.assertEqual(dds.mtype, allDds.mtype)

    def testSingleProcessGather(self):
        dds = mango.ones(shape=self.shape, mtype="labels", origin=(5,11,7), halo=2)
        
        if (dds.mpi.comm != None):
            for rootRank in range(0, dds.mpi.comm.Get_size()):
                rootDds = mango.gather(dds, root=rootRank)
                self.assertTrue(sp.all(dds.shape  == rootDds.shape))
                self.assertTrue(sp.all(dds.origin == rootDds.origin))
                self.assertEqual(dds.dtype, rootDds.dtype)
                self.assertEqual(dds.mtype, rootDds.mtype)
    
                if (dds.mpi.comm.Get_rank() == rootRank):
                    self.assertTrue(sp.all(dds.halo   == rootDds.halo))
                    self.assertTrue(sp.all(dds.shape  == rootDds.subd.shape))
                    self.assertTrue(sp.all(dds.origin == rootDds.subd.origin))
                    self.assertTrue(sp.all(dds.halo   == rootDds.subd.halo))
                else:
                    self.assertTrue(sp.all(0          == rootDds.halo))
                    self.assertTrue(sp.all(0          == rootDds.subd.shape))
                    self.assertTrue(sp.all(dds.origin == rootDds.subd.origin))
                    self.assertTrue(sp.all(0          == rootDds.subd.halo))

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.core", "mango.coreTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
