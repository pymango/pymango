__doc__ = \
"""
===============================================================================================
Mean vs Standard Deviation 2D Histogram Analysis (:mod:`mango.application.line_intensity_plot`)
===============================================================================================

.. currentmodule:: mango.application.line_intensity_plot

Functions
=========

.. autosummary::
   :toctree: generated/
   
"""

import mango
import mango.data
import mango.io
import mango.math
import mango.mpi as mpi

import scipy as sp
import numpy as np
import math

import os
import os.path
import json
import pickle
import logging

from mango.utils import ModuleObjectFactory as _ModuleObjectFactory

logger, rootLogger = mpi.getLoggers(__name__)


haveArgParse = False
try:
    import argparse
    haveArgParse = True
except:
    import optparse

def getArgumentParser():
    """
    Returns object for parsing command line options.
    :rtype: argparse.ArgumentParser
    :return: Object to parse command line options.
    """
    descStr = \
        (
            "Generates plot of intensity vs voxel position for specified images."
        )
    
    argList = []

    argList.append(
        {
             'cmdLine':['-l', '--logging-level'],
             'dest':'loggingLevel',
             'type':str,
             'metavar':'LVL',
             'default':"INFO",
             'action':'store',
             'help':"Level of logging output (one of 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')."
        }
    )

    argList.append(
        {
             'cmdLine':['-v', '--mango-verbosity'],
             'dest':'mangoVerbosity',
             'type':str,
             'metavar':'LVL',
             'default':"low",
             'action':'store',
             'help':"Level of mango C++ logging verbosity level (one of 'none', 'low', 'medium', 'high', 'extreme')."
        }
    )

    argList.append(
        {
             'cmdLine':['-s', '--start-index'],
             'dest':'startIndex',
             'type':str,
             'metavar':'(Z,Y,X)',
             'default':"(0,0,0)",
             'action':'store',
             'help':"Relative start index of line."
        }
    )

    argList.append(
        {
             'cmdLine':['-e', '--length'],
             'dest':'length',
             'type':int,
             'metavar':'N',
             'default':64,
             'action':'store',
             'help':"Number of voxels traversed."
        }
    )

    argList.append(
        {
             'cmdLine':['--dpi'],
             'dest':'dpi',
             'type':int,
             'metavar':'N',
             'default':96,
             'action':'store',
             'help':"Dots-per-inch value for plot image generation."
        }
    )

    argList.append(
        {
             'cmdLine':['--labels'],
             'dest':'labels',
             'type':str,
             'metavar':'L',
             'default':None,
             'action':'store',
             'help':"Labels for line legend."
        }
    )

    argList.append(
        {
             'cmdLine':['-a', '--axis'],
             'dest':'axis',
             'type':int,
             'metavar':'N',
             'default':0,
             'action':'store',
             'help':"Axis direction of traversal."
        }
    )

    argList.append(
        {
             'cmdLine':['-o', '--output-file-name'],
             'dest':'outputFileName',
             'type':str,
             'metavar':'F',
             'default':None,
             'action':'store',
             'help':"Name of output plot image file."
        }
    )

    if (haveArgParse):
        parser = argparse.ArgumentParser(description=descStr)

        parser.add_argument(
            'fileName',
            metavar='F',
            type=str,
            nargs='+',
            help='Mango NetCDF image files/dirs. Plot intensities for all images.'
        )

        for arg in argList:
            addArgumentDict = dict(arg)
            del addArgumentDict['cmdLine']
            parser.add_argument(*arg['cmdLine'], **addArgumentDict)
    else:
        parser = optparse.OptionParser(description=descStr)
        for arg in argList:
            addOptionDict = dict(arg)
            del addOptionDict['cmdLine']
            parser.add_option(*arg['cmdLine'], **addOptionDict)

    return parser

def genPlot(dds, startIdx, length, axis, label=None, root=0):
    import matplotlib
    import matplotlib.pyplot as plt
    
    glbStart = dds.origin + startIdx
    glbEnd   = sp.array(glbStart)+1
    glbEnd[axis] += length-1
    
    subdStart = np.maximum(dds.subd.origin,  glbStart)
    subdStart = np.minimum(dds.subd.origin+dds.subd.shape,  subdStart)
    subdEnd = np.maximum(dds.subd.origin,  glbEnd)
    subdEnd = np.minimum(dds.subd.origin+dds.subd.shape,  glbEnd)
    
    y = sp.zeros((length,), dtype = dds.dtype)
    if (sp.all(subdStart < subdEnd)):
        arrStart = subdStart - dds.subd.origin
        arrEnd = subdEnd - dds.subd.origin
        yStart = subdStart[axis] - glbStart[axis]
        yEnd   = subdEnd[axis] - glbStart[axis]
        rootLogger.info("arrStart = %s, arrEnd = %s" % (arrStart, arrEnd))
        y[yStart:yEnd] = \
            dds.subd.asarray()[arrStart[0]:arrEnd[0],arrStart[1]:arrEnd[1],arrStart[2]:arrEnd[2]]

    if (not (dds.mpi.comm is None)):
        y = dds.mpi.comm.allreduce(y, mpi.SUM)

    if (dds.mpi.comm is None) or (dds.mpi.comm.Get_rank() == root):
        if (label is None):
            plt.plot(range(glbStart[axis], glbEnd[axis]), y, linewidth=3)
        else:
            plt.plot(range(glbStart[axis], glbEnd[axis]), y, label=label, linewidth=3)

if (__name__ == "__main__"):
    import matplotlib
    matplotlib.use("Agg")
    argParser = getArgumentParser()
    if haveArgParse:
        args = argParser.parse_args()
        argv = args.fileName
    else:
        (args, argv) = argParser.parse_args()

    mpi.initialiseLoggers(
        [__name__, "mango.fmm", "mango.core", "mango.image", "mango.io", "mango.application"],
        logLevel=getattr(logging, args.loggingLevel)
    )
    
    args.startIndex = eval(args.startIndex)

    mango.setLoggingVerbosityLevel(args.mangoVerbosity)
    
    import matplotlib
    import matplotlib.pyplot as plt

    fig = None
    i = 0
    if (args.labels is None):
        args.labels = ["",]*len(argv)
    else:
        args.labels = eval(args.labels)
    for ddsFileName in argv:
        ddsSplitFileName = mango.io.splitpath(ddsFileName)
        imgDds = mango.io.readDds(ddsFileName)
        if (fig == None) and ((imgDds.mpi.comm == None) or (imgDds.mpi.comm.Get_rank() == 0)):
            fig = plt.figure()
        genPlot(imgDds, args.startIndex, args.length, args.axis, label=args.labels[i])
        i += 1

    if (imgDds.mpi.comm == None) or (imgDds.mpi.comm.Get_rank() == 0):
        plt.legend()
        if (args.outputFileName == None):
            plotFileName = "line_intensity_plot.png"
        else:
            plotFileName = args.outputFileName
        plt.savefig(plotFileName, dpi=args.dpi)

