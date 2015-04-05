#!/usr/bin/env python

import logging
import unittest
import datetime
import os
import os.path
import tempfile
import shutil

import mango.mpi as mpi
from mango.application import mdss
from mango.utils.getuser import lookup_username

logger, rootLogger = mpi.getLoggers(__name__)

if (mdss.haveMdss):
    class MdssTest(unittest.TestCase):
        """
        Test for the :obj:`mango.application.mdss` functions.
        """
        def setUp(self):
            """
            Create temporary directory.
            """
            self.mdssTmpDir = None
            self.locTmpDir = None
            if (mpi.rank == 0):
                timeString = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
                self.mdssTmpDir = os.path.join(os.path.join("tmp", lookup_username()), timeString)
                self.locTmpRoot = tempfile.mkdtemp()
                self.locTmpDir  = os.path.join(os.path.join(self.locTmpRoot, lookup_username()), timeString)
                os.makedirs(self.locTmpDir)

                rootLogger.info("Creating directory %s" % self.mdssTmpDir)
                mdss.makedirs(self.mdssTmpDir)

            if (mpi.haveMpi4py):
                self.mdssTmpDir = mpi.world.bcast(self.mdssTmpDir, root=0)
                self.locTmpDir = mpi.world.bcast(self.locTmpDir, root=0)             
            rootLogger.info("Temporary directory = %s" % self.mdssTmpDir)
        
        def tearDown(self):
            if (mpi.rank == 0):
                rootLogger.info("Temporary directory contents = %s" % mdss.listdir(self.mdssTmpDir))
                rootLogger.info("Removing directory %s" % self.mdssTmpDir)
                mdss.rmtree(self.mdssTmpDir)
                shutil.rmtree(self.locTmpRoot)

            if (mpi.haveMpi4py):
                mpi.world.barrier()
    
        def testMdssOperations(self):
            """
            Tests :func:`mango.application.mdss.listdir` function.
            """
            
            if (mpi.rank == 0):
                self.assertFalse(mdss.exists("a/dir/which/does/not/exist"))
                
                self.assertTrue(mdss.exists(self.mdssTmpDir))
                self.assertTrue(mdss.isdir(self.mdssTmpDir))
                self.assertFalse(mdss.isfile(self.mdssTmpDir))

                nonExistantPath = os.path.join(self.mdssTmpDir, "non_existent")
                self.assertFalse(mdss.exists(nonExistantPath))
                self.assertFalse(mdss.isfile(nonExistantPath))
                self.assertFalse(mdss.isdir(nonExistantPath))
                try:

                    mdss.listdir(nonExistantPath)
                    self.assertTrue(
                        False,
                        "mdss.listdir did not throw exception when listing non-existent directory."
                    )
                except Exception as e:
                    pass
                
                fileList = mdss.listdir(self.mdssTmpDir)
                self.assertEqual(0, len(fileList))
                
                mdss.mkdir(os.path.join(self.mdssTmpDir, "exists0"))
                fileList = mdss.listdir(self.mdssTmpDir)
                self.assertEqual(1, len(fileList))
                
                mdss.mkdir(os.path.join(self.mdssTmpDir, "exists1"))
                fileList = mdss.listdir(self.mdssTmpDir)
                self.assertEqual(2, len(fileList))
                fileList.sort()
                self.assertEqual("exists0", fileList[0])
                self.assertEqual("exists1", fileList[1])

                dstDir = os.path.join(self.locTmpDir, "mdss_get")
                os.mkdir(dstDir)
                mdss.get(os.path.join(self.mdssTmpDir,"*"), dstDir, recursive=True)
                fileList = os.listdir(dstDir)
                self.assertEqual(2, len(fileList))
                fileList.sort()
                self.assertEqual("exists0", fileList[0])
                self.assertEqual("exists1", fileList[1])
                
                srcDir = os.path.join(self.locTmpDir, "mdss_get")
                dstDir = os.path.join(self.mdssTmpDir, "mdss_put")
                mdss.mkdir(dstDir)
                mdss.put(os.path.join(srcDir,"*"), dstDir, recursive=True)
                fileList = mdss.listdir(dstDir)
                self.assertEqual(2, len(fileList))
                fileList.sort()
                self.assertEqual("exists0", fileList[0])
                self.assertEqual("exists1", fileList[1])
                srcFileName = os.path.join(srcDir, "src_file.txt")
                f = open(srcFileName, 'w')
                f.write("Some text.")
                f.close()
                dstFileName = os.path.join(dstDir, "dst_file.txt")
                mdss.put(srcFileName, dstFileName)
                self.assertTrue(mdss.exists(dstFileName))
                self.assertTrue(mdss.isfile(dstFileName))
                self.assertFalse(mdss.isdir(dstFileName))
                
                mvFileName = os.path.join(dstDir, "mv_dst_file.txt")
                mdss.mv(dstFileName, mvFileName)
                self.assertFalse(mdss.exists(dstFileName))
                self.assertTrue(mdss.exists(mvFileName))
                self.assertTrue(mdss.isfile(mvFileName))
                self.assertFalse(mdss.isdir(mvFileName))
                mvDstDir = os.path.join(dstDir, "exists0")
                self.assertTrue(mdss.isdir(mvDstDir))
                mvDirName = os.path.join(dstDir, "exists1")
                mvmvDirName = os.path.join(mvDstDir, os.path.split(mvDirName)[1])
                mvmvFileName = os.path.join(mvDstDir, os.path.split(mvFileName)[1])
                mdss.mv([mvFileName, mvDirName], mvDstDir)
                
                self.assertFalse(mdss.exists(mvDirName))
                self.assertFalse(mdss.exists(mvFileName))
                self.assertTrue(mdss.exists(mvmvDirName))
                self.assertTrue(mdss.isdir(mvmvDirName))
                self.assertTrue(mdss.exists(mvmvFileName))
                self.assertTrue(mdss.isfile(mvmvFileName))
                fileList = mdss.listdir(mvDstDir)
                self.assertEqual(2, len(fileList))
                self.assertTrue(os.path.split(mvmvFileName)[1] in fileList)
                self.assertTrue(os.path.split(mvmvDirName)[1] in fileList)
                
                duList = mdss.du(os.path.join(mvDstDir, "*"))
                rootLogger.info("du %s = %s" % (mvDstDir, duList))
                self.assertEqual(2, len(duList))


            if (mpi.haveMpi4py):
                mpi.world.barrier()

if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.application.mdss"],
        logLevel=logging.DEBUG
    )
    unittest.main()

