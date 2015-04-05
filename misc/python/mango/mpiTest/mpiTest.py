#!/usr/bin/env python
import logging
import unittest
import random
import scipy, numpy, numpy.random
import mango.mpi as mpi
try:
    import cPickle as pickle
except ImportError as e:
    import pickle

from io import BytesIO

logger, rootLogger = mpi.getLoggers(__name__)

class MpiTest(unittest.TestCase):

    def testAaaLogging(self):
        logger.info("\n\n")
        logger.info("Should be rank %s " % mpi.rank)
        rootLogger.info("Should only be output by root-rank process.")

    def testGetCartShape(self):
        mpiCartShape = mpi.getCartShapeForSize(dimension=2, size=64)
        rootLogger.info("shape=%s" % (mpiCartShape,))
        self.assertTrue((scipy.array((8,8),dtype="int32") == mpiCartShape).all())
        
        mpiCartShape = mpi.getCartShapeForSize(dimension=3, size=64)
        rootLogger.info("shape=%s" % (mpiCartShape,))
        self.assertTrue((scipy.array((4,4,4),dtype="int32") == mpiCartShape).all())

        mpiCartShape = mpi.getCartShapeForSize(dimension=3, size=128)
        rootLogger.info("shape=%s" % (mpiCartShape,))
        self.assertTrue((scipy.array((4,4,8),dtype="int32") == mpiCartShape).all())

        mpiCartShape = mpi.getCartShapeForSize(dimension=2, size=128)
        rootLogger.info("shape=%s" % (mpiCartShape,))
        self.assertTrue((scipy.array((8,16),dtype="int32") == mpiCartShape).all())

    def testBlockingSendReceive(self):
        if (mpi.haveMpi4py and (mpi.world.Get_size() > 1)):
            commWorld = mpi.world
            myRank = commWorld.Get_rank()
            a = bytearray("Hi to rank 1 from rank 0.", encoding='utf-8')
            if (myRank == 0):
                commWorld.send(a, dest=1)
            elif (myRank == 1):
                b = commWorld.recv(source=0)
                logger.info(str(b))

            a = bytearray("Second hi to rank 1 from rank 0.", encoding='utf-8')
            if (myRank == 0):
                commWorld.Send(a, dest=1)
            elif (myRank == 1):
                b = bytearray(2*len(a))
                commWorld.Recv(b, source=0)
                logger.info(str(b))


    def testNonBlockingSendReceive(self):
        if (mpi.haveMpi4py and (mpi.world.Get_size() > 1)):
            commWorld = mpi.world
            myRank = commWorld.Get_rank()
            if (myRank == 0):
                sndReqList = []
                recReqList = []
                rStrList = []
                sStrList = []
                for rank in range(1,commWorld.Get_size()):
                    sStrList.append(bytearray("From rank  0 to rank %2d" % rank, encoding='utf-8'))
                    sndReqList.append(commWorld.Isend(sStrList[-1], dest = rank, tag = 0))
                    rStrList.append(bytearray(128)) 
                    recReqList.append(commWorld.Irecv(rStrList[-1], source = rank, tag = 1))
                
                while (len(recReqList) > 0):
                    for rIdx in range(0, len(recReqList)):
                        if (recReqList[rIdx].Test()):
                            logger.info("Recv: " + str(rStrList[rIdx]))
                            recReqList.pop(rIdx)
                            rStrList.pop(rIdx)
                            break
                mpi.Request.Waitall(sndReqList)
            else:
                rStr = bytearray(128)
                recReq = commWorld.Irecv(rStr, source = 0, tag = 0)
                sStr = bytearray("From rank %2d to rank  0" % myRank, encoding='utf-8')
                sndReq = commWorld.Isend(sStr, dest = 0, tag = 1)
                recReq.Wait()
                logger.info("Recv: " + str(rStr))
                sndReq.Wait()
                logger.info("Send: " + str(sStr))
            commWorld.Barrier()

    def pickleToByteArray(self, obj):
        bIO = BytesIO()
        pickle.dump(obj, bIO)
        return bytearray(bIO.getvalue())
    
    def unpickleFromByteArray(self, ba):
        # bIO = BytesIO(ba.strip())
        bIO = BytesIO(ba)
        return pickle.load(bIO)
    
    def testNonBlockingSendReceiveByteArraySerialization(self):
        if (mpi.haveMpi4py and (mpi.world.Get_size() > 1)):
            commWorld = mpi.world
            myRank = commWorld.Get_rank()
            if (myRank == 0):
                sndReqList = []
                recReqList = []
                rBaList = []
                sBaList = []
                for rank in range(1,commWorld.Get_size()):
                    sBaList.append(self.pickleToByteArray([rank, "From rank  0 to rank %2d" % rank]))
                    sndReqList.append(commWorld.Isend(sBaList[-1], dest = rank, tag = 0))
                    rBaList.append(bytearray(2**16)) 
                    recReqList.append(commWorld.Irecv(rBaList[-1], source = rank, tag = 1))
                
                while (len(recReqList) > 0):
                    for rIdx in range(0, len(recReqList)):
                        if (recReqList[rIdx].Test()):
                            logger.info("Recv: " + str(self.unpickleFromByteArray(rBaList[rIdx])))
                            recReqList.pop(rIdx)
                            rBaList.pop(rIdx)
                            break
                mpi.Request.Waitall(sndReqList)
            else:
                rBa = bytearray(2**16)
                recReq = commWorld.Irecv(rBa, source = 0, tag = 0)
                sBa = self.pickleToByteArray([myRank, "From rank %2d to rank  0" % myRank])
                sndReq = commWorld.Isend(sBa, dest = 0, tag = 1)
                recReq.Wait()
                logger.info("Recv: " + str(self.unpickleFromByteArray(rBa)))
                sndReq.Wait()
                logger.info("Send: " + str(self.unpickleFromByteArray(sBa)))
            commWorld.Barrier()

if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.mpiTest"],
        logLevel=logging.INFO
    )
    random.seed(23456243)
    numpy.random.seed(23456134)
    unittest.main()
