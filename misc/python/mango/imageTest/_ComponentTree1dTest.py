#!/usr/bin/env python
import mango
import logging
import sys
import mango.mpi as mpi
import mango.unittest
import scipy as sp
import numpy as np

logger, rootLogger = mpi.getLoggers(__name__)

class ComponentTree1dTest(mango.unittest.TestCase):
    def setUp(self):
        pass
    
    def testLinearIncrease(self):

        x = sp.linspace(-10, 10, 20)
        f = x
        lbls, idxs = mango.image.component_tree_1d_leaf_labels(f)
        rootLogger.info("lbls=%s" % (lbls,))
        rootLogger.info("idxs=%s" % (idxs,))
        uLbls = sp.unique(lbls)
        self.assertEqual(1, uLbls.size)
        self.assertEqual(1, uLbls[0])
        self.assertEqual(1, idxs.size)
        self.assertEqual(f.size-1, idxs[0])

    def testLinearDecrease(self):
        x = sp.linspace(-10, 10, 20)
        f = -x
        lbls, idxs = mango.image.component_tree_1d_leaf_labels(f)
        rootLogger.info("lbls=%s" % (lbls,))
        rootLogger.info("idxs=%s" % (idxs,))
        uLbls = sp.unique(lbls)
        self.assertEqual(1, uLbls.size)
        self.assertEqual(1, uLbls[0])
        self.assertEqual(1, idxs.size)
        self.assertEqual(0, idxs[0])

    def testParabolaTop(self):

        x = sp.linspace(-10,10,25)
        f = -(x*x)
        lbls, idxs = mango.image.component_tree_1d_leaf_labels(f)
        rootLogger.info("f(x)=%s" % (f,))
        rootLogger.info("lbls=%s" % (lbls,))
        rootLogger.info("idxs=%s" % (idxs,))
        uLbls = sp.unique(lbls)
        self.assertEqual(1, uLbls.size)
        self.assertEqual(1, uLbls[0])
        self.assertEqual(1, idxs.size)
        self.assertEqual(np.argmax(f), idxs[0])

    def testParabolaBottom(self):
        x = sp.linspace(-10,10,25)
        f = (x*x)
        lbls, idxs = mango.image.component_tree_1d_leaf_labels(f)
        rootLogger.info("f(x)=%s" % (f,))
        rootLogger.info("lbls=%s" % (lbls,))
        rootLogger.info("idxs=%s" % (idxs,))
        uLbls = sp.unique(lbls)
        self.assertEqual(3, uLbls.size)
        self.assertEqual(0, uLbls[0])
        self.assertEqual(1, uLbls[1])
        self.assertEqual(2, uLbls[2])
        self.assertEqual(sp.unique(lbls[0:np.argmin(f)-1]).size, 1)
        self.assertEqual(sp.unique(lbls[np.argmin(f)+1:]).size, 1)
        self.assertEqual(2, idxs.size)
        self.assertEqual(0, idxs[0])
        self.assertEqual(f.size-1, idxs[1])

    def testCosine(self):
        import matplotlib
        import matplotlib.pyplot as plt
        
        x = sp.linspace(-3*sp.pi, 3*sp.pi, 100)
        f = x*sp.cos(x)
        lbls, idxs = mango.image.component_tree_1d_leaf_labels(f)
        rootLogger.info("f(x)=%s" % (f,))
        rootLogger.info("lbls=%s" % (lbls,))
        rootLogger.info("idxs=%s" % (idxs,))
        uLbls = sp.unique(lbls)
#         self.assertEqual(4, uLbls.size)
#         self.assertEqual(4, idxs.size)
#         self.assertEqual(0, idxs[0])

        for u in uLbls:
            msk = sp.where(lbls == u)
            plt.plot(x[msk], f[msk], label="%s" % u)

        plt.xlim(np.min(x), np.max(x))
        plt.ylim(np.min(f), np.max(f))
        plt.legend()
        #plt.show()
        
if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.INFO
    )
    mango.setLoggingVerbosityLevel("high")

    mango.unittest.main()
