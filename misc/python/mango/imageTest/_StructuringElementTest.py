#!/usr/bin/env python
import mango
import logging
import sys
import mango.mpi as mpi
import mango.unittest
import scipy as sp
import numpy as np

logger, rootLogger = mpi.getLoggers(__name__)

class StructuringElementTest(mango.unittest.TestCase):
    def setUp(self):
        pass
    
    def testSe(self):
        se = mango.image.se()
        self.assertEqual(0, se.getIndices().shape[0])

        se = mango.image.se(([0,0,0],[1,0,0],[-1,0,0], [0,1,0], [0,-1,0]))
        self.assertEqual(5, se.getIndices().shape[0])
        self.assertEqual(3, se.getIndices().shape[1])

    def testBoxSe(self):
        se = mango.image.box_se((3,4,5))
        self.assertEqual(sp.prod((3,4,5)), se.getIndices().shape[0])
        self.assertEqual(3, se.getIndices().shape[1])

    def testSphereSe(self):
        se = mango.image.sphere_se(radius=1)
        self.assertEqual(7, se.getIndices().shape[0])
        self.assertEqual(3, se.getIndices().shape[1])

    def testOrderSe(self):
        se = mango.image.order_se(1)
        self.assertEqual(7, se.getIndices().shape[0])
        self.assertEqual(3, se.getIndices().shape[1])

if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.setLoggingVerbosityLevel("high")

    mango.unittest.main()
