"""
==============================================
Unit testing utilities (:mod:`mango.unittest`)
==============================================

.. currentmodule:: mango.unittest

See the python :mod:`unittest` module for unit-testing concepts.
The :mod:`mango.unittest` module just provides MPI-aware interface for
creating temporary unit-test directory which gets cleaned up at python-exit.

Classes
=======

.. autosummary::
   :toctree: generated/

   TestCase - Extends :obj:`unittest.TestCase` with :meth:`mango.unittest.TestCase.createTmpDir` method.

Functions
==========

.. autosummary::
   :toctree: generated/

   setDoRemoveTmpDirAtExit - Set whether to remove temporary test directory on python exit.
   getDoRemoveTmpDirAtExit - Returns whether temporary test directory will be removed on python exit.
   getUnitTestTmpDir - Returns the path of the top-level unit-test temporary directory.

"""
# Use this __future__ import so we can import the python unittest module
# (otherwise :samp:`import unittest` picks up the `mango.unittest` module.
# Version 3 python does absolute_import by default. 
from __future__ import absolute_import

import os
import tempfile
import unittest as builtin_unittest
import unittest.main as main
import mango.mpi as mpi
import atexit
import shutil
from mango.utils.getuser import lookup_username

logger, rootLogger = mpi.getLoggers(__name__)

def _makedirs(dirName, mpiComm=mpi.world, rootRank=0):
    """
    Creates the specified directory. Appropriate MPI
    (:samp:`mpiComm.barrer()`) barriers to prevent attempted
    directory access before creation.
    """
    if (mpiComm != None):
        # Barrier here to make sure no other MPI process has
        # already created the directory.
        rootLogger.debug("Barrier pre %s creation..." % dirName)
        mpiComm.barrier()
    
    if ((mpiComm == None) or (mpiComm.Get_rank() == rootRank)):
        if (not os.path.exists(dirName)):
            rootLogger.info("Creating directory %s..." % dirName)
            os.makedirs(dirName)
            rootLogger.info("Finished creating directory %s." % dirName)

    if (mpiComm != None):
        # Barrier here to make sure no process starts writing
        # until the directory is created.
        rootLogger.debug("Barrier post %s creation..." % dirName)
        mpiComm.barrier()

def _rmtree(dirName, mpiComm=mpi.world, rootRank=0):
    """
    Recursively removes the specified directory. Appropriate MPI
    (:samp:`mpiComm.barrer()`) barriers to prevent attempted
    directory access after/during removal.
    """
    if (mpiComm != None):
        # Barrier here to make sure no other MPI process has
        # already created the directory.
        rootLogger.debug("Barrier pre %s removal..." % dirName)
        mpiComm.barrier()
    
    if ((mpiComm == None) or (mpiComm.Get_rank() == rootRank)):
        if (os.path.exists(dirName)):
            rootLogger.info("Removing directory %s..." % dirName)
            shutil.rmtree(dirName)
            rootLogger.info("Finished removing directory %s." % dirName)

    if (mpiComm != None):
        # Barrier here to make sure no process starts writing
        # until the directory is created.
        rootLogger.debug("Barrier post %s removal..." % dirName)
        mpiComm.barrier()

def _createRootTmpDir(prefix=None, mpiComm=mpi.world, rootRank=0):
    """
    Creates a unique temporary directory.
    Directory name is MPI-broadcasted to all :samp:`mpiComm` processes.
    :rtype: :obj:`str`
    :return: Freshly created temporary directory.
    """
    tmpRootDirName = None
    suffixedTmpRootDirName = None
    if ((mpiComm == None) or (mpiComm.Get_rank() == rootRank)):
        if (prefix != None):
            tmpRootDirName = tempfile.mkdtemp(prefix=prefix)
        else:
            tmpRootDirName = tempfile.mkdtemp()
    
    if (mpiComm != None):
        rootLogger.debug("Broadcasting tmp directory name...")
        tmpRootDirName = mpiComm.bcast(tmpRootDirName, rootRank)
    
    return tmpRootDirName

_tmpDirNameList = []
_removeTmpDirAtExit = True
def setDoRemoveTmpDirAtExit(doRemove):
    """
    Sets whether to remove temporary unit-test directory at exit.
    
    :type: :obj:`bool`
    :param doRemove: If :samp:`True`, temporary unit-test directory will be removed at exit.

    """
    _removeTmpDirAtExit = doRemove

def getDoRemoveTmpDirAtExit():
    """
    Returns whether temporary unit-test directory is removed at exit.
    :rtype: :obj:`bool`
    :return: :samp:`True` if temporary unit-test directory is to be removed at exit.
    """
    return _removeTmpDirAtExit

def _cleanUpTmpDirs(mpiComm=mpi.world, rootRank=0):
    """
    Removes directories specified in the mango.unittest._tmpDirNameList list.
    Appropriate MPI barriers.
    """
    rootLogger.debug("Entering _cleanUpTmpDirs, dirs=%s..." % (_tmpDirNameList,))
    if (_removeTmpDirAtExit):
        if (mpiComm != None):
            # Barrier here to ensure all processes have completed writing
            # to directories which are to be cleaned up/removed.
            rootLogger.debug("Barrier before unit-test tmp-dir cleanup...")
            mpiComm.barrier()
        
        for tmpDirName in _tmpDirNameList:
            if ((mpiComm == None) or (mpiComm.Get_rank() == rootRank)):
                if (os.path.exists(tmpDirName) and os.path.isdir(tmpDirName)):
                    rootLogger.info("Removing temporary directory %s..." % tmpDirName)
                    shutil.rmtree(tmpDirName)
                    rootLogger.info("Finished removing temporary directory %s." % tmpDirName)
    
        if (mpiComm != None):
            # Make sure no process gets ahead before clean-up is finisjed.
            rootLogger.debug("Barrier post unit-test tmp-dir cleanup...")
            mpiComm.barrier()
    else:
        rootLogger.debug("Skipping unit-test tmp dir cleanup, _removeTmpDirAtExit=False.")
    rootLogger.debug("Exiting _cleanUpTmpDirs.")

# Register the _cleanUpTmpDirs function to run at python exit.
atexit.register(_cleanUpTmpDirs)

_testTmpDirName = _createRootTmpDir(prefix="mangounittest_%s_" % lookup_username())
_tmpDirNameList.append(_testTmpDirName)

def getUnitTestTmpDir():
    """
    Returns the temporary unit-test directory.
    :rtype: :obj:`str`
    :return: Temporary unit-test directory name.
    """
    return _testTmpDirName

class TestCase(builtin_unittest.TestCase):
    """
    Extends :obj:`unittest.TestCase` with :meth:`mango.unittest.TestCase.getRootTmpDir`, :meth:`mango.unittest.TestCase.createTmpDir` method
    and :meth:`mango.unittest.TestCase.removeTmpDir` methods.
    """
    def getRootTmpDir(self):
        """
        Returns the temporary unit-test root directory.
        
        :rtype: :obj:`str`
        :return: Temporary unit-test directory name.
        """
        return _testTmpDirName

    def createTmpDir(self, dirName=None, mpiComm=mpi.world, rootRank = 0):
        """
        Creates unit-test temporary directory. Directory and files
        are cleaned up (removed) on python exit.
        
        :type dirName: :obj:`str`
        :param dirName: Pathname (absolute or relative to the root temporary test directory)
           to be removed. If absolute, must be a sub-directory/descendant of the root unit-test directory.
        :type mpiComm: :obj:`mpi4py.MPI.Comm`
        :param mpiComm: MPI communicator object, blocking (barrier) is performed
           using this object (to avoid an MPI process racing to access the yet-to-be-created
           directory).
        :type rootRank: int
        :param rootRank: Rank of the root process which creates the directory.
        
        :rtype: :obj:`str`
        :return: Full path of newly created directory.
        """
        dirPathName = _testTmpDirName
        if ((dirName != None) and (not os.path.isabs(dirName))):
            dirPathName = os.path.join(dirPathName, dirName)
        
        #
        # Check that the path name is in the actual unit-test tmp dir tree.
        #
        if (dirPathName.find(_testTmpDirName) == 0):
            _makedirs(dirPathName, mpiComm, rootRank)
        else:
            raise Exception("Path name %s is not in unit-test tmp directory tree %s.", (dirPathName, _testTmpDirName))

        return dirPathName

    def removeTmpDir(self, dirName, mpiComm=mpi.world, rootRank = 0):
        """
        Recursively removes specified unit-test directory.
        
        :type dirName: :obj:`str`
        :param dirName: Pathname (absolute or relative to the root temporary test directory)
           to be removed.
        :type mpiComm: :obj:`mpi4py.MPI.Comm`
        :param mpiComm: MPI communicator object, blocking (barrier) is performed
           using this object (to avoid an MPI process writing to directory while it
           is being removed).
        :type rootRank: int
        :param rootRank: Rank of the root process which removes the directory.
        
        :rtype: :obj:`str`
        :return: Full path of the removed directory.
        """
        dirPathName = _testTmpDirName

        if ((dirName != None) and (not os.path.isabs(dirName))):
            dirPathName = os.path.join(dirPathName, dirName)
        
        #
        # Check that the path name is in the actual unit-test tmp dir tree.
        #
        if (dirPathName.find(_testTmpDirName) == 0):
            _rmtree(dirPathName, mpiComm, rootRank)
        else:
            raise Exception("Path name %s is not in unit-test tmp directory tree %s.", (dirPathName, _testTmpDirName))
        
        
        return dirPathName

    ###
    ### Avoid sphinx warnings/errors about badly formed doc-strings.
    ###
    def assertItemsEqual(*args,**kwargs):
        """
        See :meth:`unittest.TestCase.assertItemsEqual`.
        """
        builtin_unittest.TestCase.assertItemsEqual(*args,**kwargs)
        
    def assertListEqual(*args,**kwargs):
        """
        See :meth:`unittest.TestCase.assertListEqual`.
        """
        builtin_unittest.TestCase.assertListEqual(*args,**kwargs)
        
    def assertRaisesRegexp(*args,**kwargs):
        """
        See :meth:`unittest.TestCase.assertRaisesRegexp`.
        """
        builtin_unittest.TestCase.assertRaisesRegexp(*args,**kwargs)
        
    def assertRaisesRegexp(*args,**kwargs):
        """
        See :meth:`unittest.TestCase.assertRaisesRegexp`.
        """
        builtin_unittest.TestCase.assertRaisesRegexp(*args,**kwargs)
        
    def assertSequenceEqual(*args,**kwargs):
        """
        See :meth:`unittest.TestCase.assertSequenceEqual`.
        """
        builtin_unittest.TestCase.assertSequenceEqual(*args,**kwargs)
        
    def assertSequenceEqual(*args,**kwargs):
        """
        See :meth:`unittest.TestCase.assertSequenceEqual`.
        """
        builtin_unittest.TestCase.assertSequenceEqual(*args,**kwargs)
        
    def assertSetEqual(*args,**kwargs):
        """
        See :meth:`unittest.TestCase.assertSetEqual`.
        """
        builtin_unittest.TestCase.assertSetEqual(*args,**kwargs)
        
    def assertTupleEqual(*args,**kwargs):
        """
        See :meth:`unittest.TestCase.assertTupleEqual`.
        """
        builtin_unittest.TestCase.assertTupleEqual(*args,**kwargs)

__all__ = [s for s in dir() if not s.startswith('_')]


