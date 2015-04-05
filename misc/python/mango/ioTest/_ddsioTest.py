#!/usr/bin/env python

import mango.unittest
import logging
import os
import os.path

import scipy as sp
import mango
import mango.mpi as mpi
import mango.io
import mango.data

logger, rootLogger = mpi.getLoggers(__name__)

class DdsIoTest(mango.unittest.TestCase):
    """
    Test for the :obj:`mango.io._ddsio` functions.
    """
    def setUp(self):
        dSz = 17
        if mango.mpi.haveMpi4py:
            dSz = 4*mango.mpi.world.Get_size()
        self.shape = (dSz, 2*dSz, 3*dSz)

    def testSplitPath(self):
        """
        Test the :func:`mango.io.splitpath` function.
        """
        testPathList = [
            ("", "tomo_float", "", ".nc"),
            ("", "tomo_float", "", "_nc"),
            ("rel/dir/name", "tomo_float", "", "_nc"),
            ("/abs/dir/name", "tomo_float", "", "_nc"),
            ("/abs/dir/name", "tomo_float", "Suffix", "_nc"),
            ("/abs/dir/name", "tomo_float", "_Suffix_", "_nc"),
            ("/abs/dir/name", "tomo_float", "_Suffix_", ".nc"),
            ("", "tomo", "", ".nc"),
            ("", "tomo", "", "_nc"),
            ("rel/dir/name", "tomo", "", "_nc"),
            ("/abs/dir/name", "tomo", "", "_nc"),
            ("/abs/dir/name", "tomo", "Suffix", "_nc"),
            ("/abs/dir/name", "tomo", "_Suffix_", "_nc"),
            ("/abs/dir/name", "tomo", "_Suffix_", ".nc"),
            ("", "distance_map", "", ".nc"),
            ("", "distance_map", "", "_nc"),
            ("rel/dir/name", "distance_map", "", "_nc"),
            ("/abs/dir/name", "distance_map", "", "_nc"),
            ("/abs/dir/name", "distance_map", "Suffix", "_nc"),
            ("/abs/dir/name", "distance_map", "_Suffix_", "_nc"),
            ("/abs/dir/name", "distance_map", "_Suffix_", ".nc"),

        ]
        for pathTuple in testPathList:
            path = os.path.join(pathTuple[0], "".join(pathTuple[1:]))
            splt = mango.io.splitpath(path)
            self.assertEqual(pathTuple[0], splt[0])
            self.assertEqual(pathTuple[1], splt[1])
            self.assertEqual(pathTuple[2], splt[2])
            self.assertEqual(pathTuple[3], splt[3])
        
        try:
            path = "tomoShouldThrowException,nc"
            splt = mango.io.splitpath(path)
            failed = True
        except:
            failed = False
        
        if (failed):
           self.fail("Did not throw exception for mango.io.splitpath('%s')" % path)

    def testWriteDdsWithHistogram(self):
        outDir = self.createTmpDir("testWriteDdsWithHistogram")
        dds = mango.data.gaussian_noise(shape=self.shape, mtype="tomo_float", mean=0.0, stdd=0.02)
        mango.io.writeDds(os.path.join(outDir, "tomo_floatWithHistogram.nc"), dds, writehistogram=True)
        mango.io.writeDds(os.path.join(outDir, "tomo_floatWithOutHistogram.nc"), dds, writehistogram=False)

    def testDdsWriteAndRead(self):
        outDir = self.createTmpDir("testDdsWriteAndRead")
        dds = mango.data.gaussian_noise(shape=self.shape, mtype="float64", mean=0.0, stdd=20000.)
        outFileName = mango.io.writeDds(os.path.join(outDir, "float64Noise.nc"), dds)
        readDds = mango.io.readDds(outFileName, mpidims=dds.mpi.shape)
        self.assertTrue(sp.all(dds.asarray() == readDds.asarray()))

        if (mango.haveFloat16):
            dds = mango.data.gaussian_noise(shape=self.shape, mtype="float16", mean=0.0, stdd=2000.)
            outFileName = mango.io.writeDds(os.path.join(outDir, "float16Noise.nc"), dds, writehistogram=True)
            readDds = mango.io.readDds(outFileName, mpidims=dds.mpi.shape)
            self.assertTrue(sp.all(dds.asarray() == readDds.asarray()))
        
if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.io"],
        logLevel=logging.DEBUG
    )
    mango.unittest.main()

