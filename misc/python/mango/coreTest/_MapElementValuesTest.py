#!/usr/bin/env python
import logging
import mango.unittest
import scipy as sp
import numpy as np
import numpy.random
import mango.mpi as mpi
import mango
import mango.data
import mango.io
import os
import os.path

logger, rootLogger = mpi.getLoggers(__name__)

class MapElementValuesTest(mango.unittest.TestCase):
    def setUp(self):
        dSz = 13
        if mango.mpi.haveMpi4py:
            dSz = 4*mango.mpi.world.Get_size()
        self.shape = (dSz, 2*dSz, 3*dSz)

    def doTestMapElements(self, use1to1cache):
        dds = mango.data.gaussian_noise(self.shape, mean=1000, stdd=2, dtype="int32")
        mapDds = mango.map_element_values(dds, lambda x: x*x, use1to1cache=use1to1cache)
        self.assertTrue(sp.all(dds.asarray()**2 == mapDds.asarray()))

        mapDds = mango.map_element_values(dds, lambda x: sp.sqrt(x), dtype=sp.dtype("float32"), use1to1cache=use1to1cache)
        self.assertEqual(sp.dtype("float32"), mapDds.dtype)
        self.assertTrue(not hasattr(mapDds, "mtype"))
        diff = sp.sqrt(dds.asarray()) - mapDds.asarray()
        rootLogger.info("min, max diff = %s,%s" % (np.min(diff), np.max(diff)))
        self.assertTrue(sp.all(sp.absolute(diff) <= 1.0e-6))

        mapDds = mango.map_element_values(dds, lambda x: sp.sqrt(x), mtype="tomo_float", use1to1cache=use1to1cache)
        self.assertEqual(sp.dtype("float32"), mapDds.dtype)
        self.assertTrue(hasattr(mapDds, "mtype"))
        self.assertEqual(mapDds.mtype, mango.mtype("tomo_float"))
        diff = sp.sqrt(dds.asarray()) - mapDds.asarray()
        rootLogger.info("min, max diff = %s,%s" % (np.min(diff), np.max(diff)))
        self.assertTrue(sp.all(sp.absolute(diff) <= 1.0e-6))

    def testMapWith121Cache(self):
        self.doTestMapElements(True)

    def testMapWithout121Cache(self):
        self.doTestMapElements(False)

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.core", "mango.coreTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
