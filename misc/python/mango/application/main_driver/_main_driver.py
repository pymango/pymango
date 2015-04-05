import mango
import mango.mpi as mpi

import os
import os.path
import sys
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_main_driver as _mango_main_driver_so
    sys.setdlopenflags(_flags)
else:
    from . import _mango_main_driver as _mango_main_driver_so

from ._mango_main_driver import InputParmManager, InputParmSection
from ._mango_main_driver import initialiseGlobalWithPaths, finaliseGlobal
from ._mango_main_driver import doCompletionBannerLogging, doStartBannerLogging
from ._mango_main_driver import doFilterRunBannerLogging, doRunParmSectionHacks
from ._mango_main_driver import findMainDriverFilter, registerMainDriverFilter, clearMainDriverFilterRegistry
from ._mango_main_driver import getAllRegisteredMainDriverFilterClasses
from ._mango_main_driver import MainDriverFilter

from . import filters
from . import logstream
import mango.mpe as mpe

class MainDriver:
    """
    Simple driver for executing a series of *Run* sections
    as specified by a :obj:`InputParmManager` object. 
    """
    def __init__(self, prmMngr):
        """
        Initialise with a :obj:`InputParmManager`. 
        """
        self.prmMngr = prmMngr
    
    def executeRuns(self):
        """
        Executes *Run* sections from :samp:`self.prmMngr`. 
        """
        self.prmMngr.resetUsedParmsList()
        logstream.mstOut.resetStreamContent();
        sectionNameList = self.prmMngr.getSectionNames()
        for name in sectionNameList:
            if (name.find("Run_") == 0):
                runSect = self.prmMngr.getSection(name)
                if (runSect.getParameter("enabled").lower() == "true"):
                    doRunParmSectionHacks(runSect)
                    for filtSectName in runSect.getSectionNames():
                        if ((mango.mpi.world == None) or (mango.mpi.world.Get_rank() == 0)):
                            doFilterRunBannerLogging(filtSectName)
                        if (mango.mpi.world != None):
                            mango.mpi.world.barrier()
                        filtSect = runSect.getSection(filtSectName)
                        filtDataType = mango.makeMType(runSect.getParameter("input_data_type"))
                        filtCls = findMainDriverFilter(filtSectName, filtDataType)
                        filt = filtCls()
                        filt.setFilterParms(filtSect)
                        filt.setRunParms(runSect)
                        filt.loadExecuteWrite()

                        self.prmMngr.resetUsedParmsList()
                        logstream.mstOut.resetStreamContent();


def execute_run_sections(prmMngr):
    """
    Executes a series of filters as specified in :samp:`{prmMngr}` argument.
    
    :type prmMngr: :obj:`InputParmManager`
    :param prmMngr: A dictionary which specifies input/output data and filters to
        execute in a series of *Run* sections.
    """
    md = MainDriver(prmMngr)
    md.executeRuns()

def run_main_driver(parameter_file_name, input_directory=None, exe_name="", script_name=""):
    """
    Executes a series of *filters* as specified in the file :samp:`{parameter_file_name}`.
    
    :type parameter_file_name: :obj:`str`
    :param parameter_file_name: Name of '.in' file. 
    """
    parameter_file_name = os.path.splitext(parameter_file_name)[0]
    if (input_directory == None):
        input_directory = ""
    prmMngr = initialiseGlobalWithPaths(parameter_file_name, input_directory)
    
    if ((mango.mpi.world == None) or (mango.mpi.world.Get_rank() == 0)):
        doStartBannerLogging(exe_name, os.path.dirname(os.path.realpath(__file__)), script_name)
    if (mango.mpi.world != None):
        mango.mpi.world.barrier()
    
    execute_run_sections(prmMngr)
    
    if (mango.mpi.world != None):
        mango.mpi.world.barrier()
   
    mpe.stop_log()
    mpe.finish_log()

    if ((mango.mpi.world == None) or (mango.mpi.world.Get_rank() == 0)):
        doCompletionBannerLogging()

    finaliseGlobal()


haveArgParse = False
try:
    import argparse
    haveArgParse = True
except:
    import optparse

def getArgumentParser():
    """
    Returns object for parsing command line options.
    
    :rtype: :obj:`argparse.ArgumentParser`
    :return: Object to parse command line options.
    """
    descStr = \
        (
            "Executes sequence of filters as specified in a parameter file."
        )
    
    argList = []

    argList.append(
        {
             'cmdLine':['-b', '-base'],
             'dest':'base',
             'type':str,
             'metavar':'F',
             'default':None,
             'required':True,
             'action':'store',
             'help':"Parameter file name (e.g. '-base prm.in' "
                    +
                    "or '-base prm')."
        }
    )

    argList.append(
        {
             'cmdLine':['-i', '-input_directory'],
             'dest':'inputDirectory',
             'type':str,
             'metavar':'DIR',
             'default':None,
             'action':'store',
             'help':"Directory where the parameter file is located."
        }
    )

    if (haveArgParse):
        parser = argparse.ArgumentParser(description=descStr)

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

__all__ = [s for s in dir() if not s.startswith('_')]