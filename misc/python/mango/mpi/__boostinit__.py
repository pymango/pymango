
haveBoostMpi = False
try:
    import ctypes
except:
    import dl as ctypes
import sys
import math
try:
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL) 

    from boost.mpi import *
    haveBoostMpi = True
except:
    try:
        from mpi import *
        haveBoostMpi = True
    except:
        haveBoostMpi = False

sys.setdlopenflags(flags) 

if (not haveBoostMpi):
    global rank, size
    rank = 0
    size = 1

import logging

def getFormattedRankString():
    global rank, size
    if (haveBoostMpi):
        rnkFormatStr = ("%0" + str(len(str(size))) + "d") % rank
    else:
        rnkFormatStr = "0"
    return rnkFormatStr

def getFormatter(mpiRnkLabel="MpiRnk"):
    if haveBoostMpi:
        formatter = \
            logging.Formatter(
                "%(asctime)s:" + mpiRnkLabel + getFormattedRankString() + ": " + "%(message)s",
                "%H:%M:%S"
            )
    else:
        formatter = \
            logging.Formatter(
                "%(asctime)s:" + mpiRnkLabel + ": %(message)s",
                "%H:%M:%S"
            )
    return formatter

def getRootFormatter():
    if (not haveBoostMpi):
        formatter = getFormatter("Root")
    else:
        formatter = getFormatter("MpiRoot")
    return formatter

def getRootLogger(logrNamePrefix, handlerList=[], formatter=None, rootRank=0):
    """
    Returns logger for logging events from only the MPI process with
    rank rootRank.
    :Parameters:
      logrNamePrefix : str
        Name of returned logger is logrNamePrefix + rank-number suffix.
      handlerList : list
        List of logging.Handler objects, these get added to the logger on the rootRank process.
      formatter : logging.Formatter
        Gets set as the formatter on the rootRank process.
      rootRank : int
        MPI rank of the root process on which logging events are to be recorded.
    """
    global rank, size
    if (not haveBoostMpi):
        rank = rootRank
        logr = logging.getLogger(logrNamePrefix + "Root")
        if (formatter == None):
            formatter = getRootFormatter()
    else:
        if (formatter == None):
            formatter = getRootFormatter()
        logr = logging.getLogger(logrNamePrefix + "MpiRoot" + getFormattedRankString())
    if (rank == rootRank):
        for handler in handlerList:
            handler.setFormatter(formatter)
            logr.addHandler(handler)

    return logr

def getLoggers(name, rootRank=0):
    logr = logging.getLogger(name)
    for handler in logr.handlers:
        handler.setFormatter(getFormatter())
    rootLogr = getRootLogger(name, rootRank=rootRank)
    return logr, rootLogr

def initialiseLoggers(logrNameList, handlerClass, logLevel=logging.WARN, rootRank=0):
    frmttr = getFormatter()
    rootFrmttr = getRootFormatter()
    for name in logrNameList:
        logr,rootLogr = getLoggers(name, rootRank=rootRank)
        handler = handlerClass()
        handler.setFormatter(frmttr)
        logr.addHandler(handler)
        logr.setLevel(logLevel)
        if (rank == rootRank):
            handler = handlerClass()
            handler.setFormatter(rootFrmttr)
            rootLogr.addHandler(handler)
            rootLogr.setLevel(logLevel)
        elif hasattr(logging, "NullHandler"):
            rootLogr.addHandler(logging.NullHandler())


def getCartShapeForSize(dimension, size):
    shape = [int(1) for i in range(dimension)]
    numRemaining = size
    d = 0 
    while (d < dimension) and (numRemaining > 0):
        f = int(round(math.pow(float(numRemaining), 1.0/float(dimension-d))))
        while ((f > 1) and ((numRemaining % f) != 0)):
            f -= 1
        shape[d] = f
        numRemaining = numRemaining/f
        d += 1

    shape.sort()
    return shape

def getCartShape(dimension, communicator=None):
    if communicator == None:
        communicator = world
    return getCartShapeForSize(dimension, communicator.size)

logger,rootLogger = getLoggers(__name__)
