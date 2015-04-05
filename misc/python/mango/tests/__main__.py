import logging
import mango.unittest
import mango.mpi
from mango.tests import *

if __name__ == "__main__":
    mango.mpi.initialiseLoggers(
        [__name__, "mango"],
        logLevel=logging.WARNING
    )
    # Make sure we use rendering that
    # doesn required a $DISPLAY.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    mango.unittest.main()
