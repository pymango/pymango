#!/usr/bin/env python

import logging
import mango.unittest
import datetime
import os
import os.path
import tempfile
import shutil

import scipy as sp

import mango.mpi as mpi
import mango.data
import mango.io
from mango.utils.getuser import lookup_username

logger, rootLogger = mpi.getLoggers(__name__)

class CompressionTest(mango.unittest.TestCase):
    """
    Test for the :obj:`mango.io.compressDir` and :obj:`mango.io.uncompressDir` functions.
    """
    def setUp(self):
        """
        Create temporary directory.
        """
        self.locTmpDir = self.createTmpDir("CompressionTest")

    def tearDown(self):
        """
        Remove temporary directory.
        """
        self.removeTmpDir(self.locTmpDir)

    def testDirectoryCompressionNonPreserve(self):
        """
        Tests :func:`mango.io.compressDir` and :func:`mango.io.uncompressDir` functions.
        """
        dds = mango.data.gaussian_noise(shape=(256,198,200), mean=128, stdd=16, mtype="segmented")
        
        for method in [mango.io.BZIP2, mango.io.GZIP]:
            ddsDestPath = \
                mango.io.writeDds(
                    os.path.join(self.locTmpDir, "segmentedGaussianNoise.nc"),
                    dds,
                    melemperfile=4
                )
            
            rootLogger.info(os.listdir(ddsDestPath))
            
            mango.io.compressDir(ddsDestPath, method=method, preserve=False)
            rootLogger.info(os.listdir(ddsDestPath))
    
            mango.io.uncompressDir(ddsDestPath, preserve=False)
            rootLogger.info(os.listdir(ddsDestPath))
            
            rDds = mango.io.readDds(ddsDestPath)
            self.assertTrue(sp.all(dds.asarray() == rDds.asarray()))

    def testDdsDataCompressionNonPreserve(self):
        """
        Tests :func:`mango.io.compressDdsData` and :func:`mango.io.uncompressDdsData` functions.
        """
        dds = mango.data.gaussian_noise(shape=(256,198,200), mean=128, stdd=16, mtype="segmented")
        
        for method in [mango.io.BZIP2, mango.io.GZIP]:
            ddsDestPath = \
                mango.io.writeDds(
                    os.path.join(self.locTmpDir, "segmentedGaussianNoise.nc"),
                    dds,
                    melemperfile=4
                )
            
            rootLogger.info(os.listdir(ddsDestPath))
            
            compressedList = mango.io.compressDdsData(ddsDestPath, method=method, preserve=False)
            rootLogger.info("Compressed = %s" % (compressedList,)) 
            rootLogger.info(os.listdir(ddsDestPath))
    
            uncompressedList = mango.io.uncompressDdsData(ddsDestPath, preserve=False)
            rootLogger.info("Uncompressed = %s" % (uncompressedList,))
            rootLogger.info(os.listdir(ddsDestPath))
            
            rDds = mango.io.readDds(ddsDestPath)
            self.assertTrue(sp.all(dds.asarray() == rDds.asarray()))

    def testDirectoryCompressionPreserve(self):
        """
        Tests :func:`mango.io.compressDir` and :func:`mango.io.uncompressDir` functions.
        """
        dds = mango.data.gaussian_noise(shape=(256,198,200), mean=128, stdd=16, mtype="segmented")

        for method in [mango.io.BZIP2, mango.io.GZIP]:
            ddsDestPath = \
                mango.io.writeDds(
                    os.path.join(self.locTmpDir, "segmentedGaussianNoise.nc"),
                    dds,
                    melemperfile=4
                )
            zipDir = os.path.join(self.locTmpDir, "segmentedGaussianNoiseZipped_nc")
            unzipDir = os.path.join(self.locTmpDir, "segmentedGaussianNoiseUnzipped_nc")
            
            if ((not mpi.haveMpi4py) or (mpi.world.Get_rank() == 0)):
                if (os.path.exists(zipDir)):
                    shutil.rmtree(zipDir)
                if (os.path.exists(unzipDir)):
                    shutil.rmtree(unzipDir)

                os.makedirs(zipDir)
                os.makedirs(unzipDir)
            if (mpi.haveMpi4py):
                mpi.world.barrier()

            rootLogger.info("orig=%s" % (os.listdir(ddsDestPath),))
            rootLogger.info("zipd=%s" % (os.listdir(zipDir),))
            rootLogger.info("unzp=%s" % (os.listdir(unzipDir),))
            
            mango.io.compressDir(ddsDestPath, zipDir, method=method)
            rootLogger.info("orig=%s" % (os.listdir(ddsDestPath),))
            rootLogger.info("zipd=%s" % (os.listdir(zipDir),))
    
            mango.io.uncompressDir(zipDir, unzipDir)
            rootLogger.info("zipd=%s" % (os.listdir(zipDir),))
            rootLogger.info("unzp=%s" % (os.listdir(unzipDir),))
            
            rDds = mango.io.readDds(unzipDir)
            self.assertTrue(sp.all(dds.asarray() == rDds.asarray()))

if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango"],
        logLevel=logging.DEBUG
    )
    mango.setLoggingVerbosityLevel("high")
    mango.unittest.main()

