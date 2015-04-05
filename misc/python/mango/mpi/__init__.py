__doc__ = \
"""
======================================================
Message Passing Interface Utilities (:mod:`mango.mpi`)
======================================================
.. _mpi4py: http://mpi4py.scipy.org
.. currentmodule:: mango.mpi

Convenience functions for MPI (`mpi4py`_). This module imports
everything from the `mpi4py`_ namespace (if available).


Variables
=========

.. autodata:: haveMpi4py

.. autodata:: world
   
.. autodata:: size
   
.. autodata:: rank


Functions
=========

.. autosummary::
   :toctree: generated/

   getLoggers - Returns :samp:`(logger,rootLogger)` pair for MPI logging.
   initialiseLoggers - Initialises loggers to generate output at a specified logging-level.
   getCartShape - Calculates an MPI cartesian layout for a given dimension and MPI communicator.
   getCartShapeForSize - Calculates an MPI cartesian layout for a given dimension and number of MPI processes.
   
Classes
=======
.. autosummary::
   :toctree: generated/

   SplitStreamHandler - A :obj:`logging.Handler` class for splitting messages between a pair of files/streams.

"""

#    
#    :annotation: The `COMM_WORLD` MPI communicator.
#    :annotation: Number of MPI processes in world (world.Get_size()).
#    :annotation: Rank of this MPI process (world.Get_rank()).

#global rank, size, world, haveMpi4py

haveMpi4py = False #: Boolean indicating the availability of :samp:`mpi4py`.

import sys
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_open_mpi
    sys.setdlopenflags(_flags)
else:
    from . import _mango_open_mpi

from ._mango_open_mpi import usingMpi as _mangoUsingMpi

try:
    if (_mangoUsingMpi()):
        from mpi4py.MPI import *
        haveMpi4py = True
except:
    haveMpi4py = False

import atexit
_mango_open_mpi._initialiseMpeTiming()
atexit.register(_mango_open_mpi._finaliseMpeTiming)

rank = 0 #: Rank of this MPI process (:samp:`world.Get_rank()`).
size = 1 #: Number of MPI processes in world (:samp:`world.Get_size()`).
world = None #: The :samp:`COMM_WORLD` MPI communicator.

if (haveMpi4py):
    world = COMM_WORLD
    rank  = COMM_WORLD.Get_rank()
    size  = COMM_WORLD.Get_size()

import logging, math, sys

def getFormattedRankString():
    """
    Returns the logging prefix containing MPI-rank info.
    """
    global rank, size
    if (haveMpi4py):
        rnkFormatStr = ("%0" + str(len(str(size))) + "d") % rank
    else:
        rnkFormatStr = "0"
    return rnkFormatStr

def getFormatter(mpiRnkLabel="MpiRnk"):
    """
    :rtype: logging.Formatter
    :return: Regular formatter for non-root-rank logging.
    """
    if haveMpi4py:
        rnkStr = mpiRnkLabel + getFormattedRankString()
        paddingLen = max([0, len("MpiRoot" + getFormattedRankString()) - len(rnkStr)])
        formatter = \
            logging.Formatter(
                "%(asctime)s:" + rnkStr + ": " + (" "*paddingLen) + "%(message)s",
                "%H:%M:%S"
            )
    else:
        rnkStr = mpiRnkLabel
        paddingLen = max([0, len("Root") - len(rnkStr)])
        formatter = \
            logging.Formatter(
                "%(asctime)s:" + mpiRnkLabel + ": " + (" "*paddingLen) + "%(message)s",
                "%H:%M:%S"
            )
    return formatter

def getRootFormatter():
    """
    :rtype: logging.Formatter
    :return: Formatter for root-rank logging.
    """
    if (not haveMpi4py):
        formatter = getFormatter("Root")
    else:
        formatter = getFormatter("MpiRoot")
    return formatter

def getRootLogger(logrNameSuffix, handlerList=[], formatter=None, rootRank=0):
    """
    Returns logger for logging events from only the MPI process with
    rank rootRank.

    :type logrNameSuffix: str
    :param logrNameSuffix: Name of returned logger is "root." + logrNameSuffix.
    :type handlerList: list
    :param handlerList: List of logging.Handler objects, these get added to the logger on the rootRank process.
    :type formatter: logging.Formatter
    :param formatter: Gets set as the formatter on the rootRank process.
    :type rootRank: int
    :param rootRank: MPI rank of the root process on which logging events are to be recorded.
    :rtype: logging.Logger
    :return: A logger object for logging from the root-rank MPI process only.
    """
    global rank, size
    if (not haveMpi4py):
        rank = rootRank
        logr = logging.getLogger("root." + logrNameSuffix)
        if (formatter == None):
            formatter = getFormatter()
    else:
        if (formatter == None):
            formatter = getFormatter()
        logr = logging.getLogger("root." + logrNameSuffix)
    if (rank == rootRank):
        for handler in handlerList:
            handler.setFormatter(formatter)
            logr.addHandler(handler)

    return logr

def getNonRootLogger(logrNameSuffix, handlerList=[], formatter=None, rootRank=0):
    """
    Returns logger for logging events from all the MPI processes.

    :type logrNameSuffix: str
    :param logrNameSuffix: Name of returned logger is "rank." + logrNameSuffix.
    :type handlerList: list
    :param handlerList: List of logging.Handler objects, these get added to the logger on this MPI process.
    :type formatter: logging.Formatter
    :param formatter: Gets set as the formatter on this MPI process.
    :type rootRank: int
    :param rootRank:MPI rank of the root process.
    :rtype: logging.Logger
    :return: A logger object for logging from the any-rank MPI process.
    """
    global rank, size
    if (not haveMpi4py):
        rank = rootRank
        logr = logging.getLogger("rank." + logrNameSuffix)
        if (formatter == None):
            formatter = getFormatter()
    else:
        if (formatter == None):
            formatter = getFormatter()
        logr = logging.getLogger("rank." + logrNameSuffix)
    for handler in handlerList:
        handler.setFormatter(formatter)
        logr.addHandler(handler)

    return logr

def getLoggers(name, rootRank=0):
    """
    Returns :samp:`(logger,rootLogger)` pair of :obj:`logging.Logger`
    objects. The objects will prefix messages with time and rank strings.
    
    :type name: str
    :param name: The name suffix for the loggers.
    :type rootRank: int
    :param rootRank: Rank of the root-logger MPI process.
    :rtype: :obj:`logging.Logger` pair
    :return: :samp:`(logger,rootLogger)` logging object pair.
    
    For example::
       >>> import mango.mpi
       >>> import logging
       >>> logger, rootLogger = mangp.mpi.getLoggers("my.module.name")
       >>> mango.mpi.initialise()
       >>> mpi.initialiseLoggers(["my.module.name",], logLevel=logging.INFO)
       >>> logger.info("This message appears for all MPI process ranks (including root rank).")
       >>> rootLogger.info("This message only appears for the root rank process.")
    
    """
    logr = getNonRootLogger(name)
    for handler in logr.handlers:
        handler.setFormatter(getFormatter())
    rootLogr = getRootLogger(name, rootRank=rootRank)
    return logr, rootLogr

class _Python2SplitStreamHandler(logging.Handler):
    """
    A python :obj:`logging.Handler` for splitting logging
    messages to different streams depending on the logging-level. 
    """
    def __init__(self, outstr=sys.stdout, errstr=sys.stderr, splitlevel=logging.WARNING):
        """
        Initialise with a pair of streams and a threshold level which determines
        the stream where the messages are writting.
        
        :type outstr: file-like
        :param outstr: Logging messages are written to this stream if
           the message level is less than :samp:`self.splitLevel`.
        :type errstr: stream
        :param errstr: Logging messages are written to this stream if
           the message level is greater-than-or-equal-to :samp:`self.splitLevel`.
        :type splitlevel: int
        :param splitlevel: Logging level threshold determining split streams for log messages. 
        """
        self.outStream = outstr
        self.errStream = errstr
        self.splitLevel = splitlevel
        logging.Handler.__init__(self)

    def emit(self, record):
        # mostly copy-paste from logging.StreamHandler
        try:
            msg = self.format(record)
            if record.levelno < self.splitLevel:
                stream = self.outStream
            else:
                stream = self.errStream
            fs = "%s\n"

            try:
                if (isinstance(msg, unicode) and
                    getattr(stream, 'encoding', None)):
                    ufs = fs.decode(stream.encoding)
                    try:
                        stream.write(ufs % msg)
                    except UnicodeEncodeError:
                        stream.write((ufs % msg).encode(stream.encoding))
                else:
                    stream.write(fs % msg)
            except UnicodeError:
                stream.write(fs % msg.encode("UTF-8"))

            stream.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class _Python3SplitStreamHandler(logging.Handler):
    """
    A python :obj:`logging.Handler` for splitting logging
    messages to different streams depending on the logging-level. 
    """
    
    terminator = '\n'
    
    def __init__(self, outstr=sys.stdout, errstr=sys.stderr, splitlevel=logging.WARNING):
        """
        Initialise with a pair of streams and a threshold level which determines
        the stream where the messages are writting.
        
        :type outstr: file-like
        :param outstr: Logging messages are written to this stream if
           the message level is less than :samp:`self.splitLevel`.
        :type errstr: stream
        :param errstr: Logging messages are written to this stream if
           the message level is greater-than-or-equal-to :samp:`self.splitLevel`.
        :type splitlevel: int
        :param splitlevel: Logging level threshold determining split streams for log messages. 
        """
        self.outStream = outstr
        self.errStream = errstr
        self.splitLevel = splitlevel
        logging.Handler.__init__(self)

    def flush(self):
        """
        Flushes the stream.
        """
        self.acquire()
        try:
            if self.outStream and hasattr(self.outStream, "flush"):
                self.outStream.flush()
            if self.errStream and hasattr(self.errStream, "flush"):
                self.errStream.flush()
        finally:
            self.release()

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            if record.levelno < self.splitLevel:
                stream = self.outStream
            else:
                stream = self.errStream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except (KeyboardInterrupt, SystemExit): #pragma: no cover
            raise
        except:
            self.handleError(record)

if (sys.version_info[0] <= 2):
    class SplitStreamHandler(_Python2SplitStreamHandler):
        pass
else:
    class SplitStreamHandler(_Python3SplitStreamHandler):
        pass

def initialiseLoggers(logrNameList, handlerClass=SplitStreamHandler, logLevel=logging.WARN, rootRank=0):
    """
    Initialises specified (by name suffix) loggers to generate output at the
    specified logging level. If the specified named loggers do not exist,
    they are created.
    
    :type logrNameList: :obj:`list` of :obj:`str`
    :param logrNameList: List of logger names.
    :type handlerClass: :obj:`logging.Handler` type.
    :param handlerClass: The handler class for output of log messages,
       for example :obj:`SplitStreamHandler` or :obj:`logging.StreamHandler`.
    :type logLevel: int
    :param logLevel: Log level for messages, typically
       one of :obj:`logging.DEBUG`, :obj:`logging.INFO`, :obj:`logging.WARN`, :obj:`logging.ERROR`
       or :obj:`logging.CRITICAL`.
       See :ref:`levels`.
    :type rootRank: int
    :param rootRank: MPI rank of the root-logger process.
    
    See :func:`getLoggers` for example usage.
    
    """
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
    """
    Returns an MPI cartesian layout for the given
    cartesian-dimension (:samp:`dimension`) and number of
    MPI processes (:samp:`size`).
    
    :type dimension: int
    :param dimension: Spatial dimension for returned cartesian layout.
    :type size: int
    :param size: Number of MPI processes to use in layout.
    :rtype: list
    :return: list of dimension elements, such that
       the product (multiplication) of the elements equals :samp:`size`.
    """
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
    """
    Returns :samp:`getCartShapeForSize(dimension, communicator.Get_size())`.
    
    :type dimension: int
    :param dimension: Spatial dimension for returned cartesian layout.
    :type communicator: :obj:`mpi4py.MPI.Comm`
    :param communicator: If :samp:`None`, uses :obj:`world`.
    :rtype: list
    :return: :samp:`getCartShapeForSize(dimension, communicator.Get_size())`
    """
    if communicator == None:
        communicator = world
    if communicator != None:
        commSz = communicator.Get_size()
    else:
        commSz = 1
    return getCartShapeForSize(dimension, commSz)

__all__ = [s for s in dir() if not s.startswith('_')]

logger,rootLogger = getLoggers(__name__)

