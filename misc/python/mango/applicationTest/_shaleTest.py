#!/usr/bin/env python

import logging
import mango.unittest
import scipy as sp
import numpy as np
import mango
import mango.mpi as mpi
from mango.application.shale    import *
from mango.application.plot     import *
from mango.application.io       import *

logger, rootLogger = mpi.getLoggers(__name__)

def mockHistData(xx, yy):
    """
    Returns evaluates a sum of 2D Gaussians.
    """
    sx = 10.0
    sy = 20.0
    invSxSqrd = 1.0/(2*sx*sx)
    invSySqrd = 1.0/(2*sy*sy)
    dx = xx-90.0
    dy = yy-30.0
    f = sp.exp(-((dx*dx)*invSxSqrd + (dy*dy)*invSySqrd))
    
    sx = 35.0
    sy =  5.0
    invSxSqrd = 1.0/(2*sx*sx)
    invSySqrd = 1.0/(2*sy*sy)
    dx = xx-25.0
    dy = yy-95.0
    f += sp.exp(-((dx*dx)*invSxSqrd + (dy*dy)*invSySqrd))
    
    return f

class ShaleTernaryPlotTest(mango.unittest.TestCase):
    """
    Test for the :func:`mango.application.shale.generateShaleTernaryPlot`
    function.
    """
    def setUp(self):
        """
        Creates some dummy 2D histogram data.
        """
        import matplotlib
        from matplotlib import pyplot as plt
        plt.ioff()
        hd = HistData()
        hd.edges = [sp.arange(0, 104, 1), sp.arange(0, 104, 1)]
        hd.x = (hd.edges[0][1:] + hd.edges[0][0:-1])*0.5
        hd.y = (hd.edges[1][1:] + hd.edges[1][0:-1])*0.5
        xx,yy = np.meshgrid(hd.x, hd.y)
        hd.hist2dData = mockHistData(xx,yy)
        hd.hist1dData0 = sp.sum(hd.hist2dData, axis=0)
        hd.hist1dData1 = sp.sum(hd.hist2dData, axis=1)
        
        plt.pcolor(xx,yy,hd.hist2dData)
        self.histData = hd
        
    def testTernaryPlot(self):
        """
        Tests ternary plotting of 2D histogram data.
        """
        figList = mango.application.shale.generateShaleTernaryPlot(self.histData)
        self.assertLess(2, len(figList))
        # plt.show()


if __name__ == "__main__":
    mpi.initialiseLoggers(
        [__name__, "mango.application"],
        logLevel=logging.DEBUG
    )
    unittest.main()

