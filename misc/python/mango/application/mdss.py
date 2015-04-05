__doc__ = \
"""
=========================================================
Mass Data Storage System (:mod:`mango.application.mdss`)
=========================================================

.. currentmodule:: mango.application.mdss

National Computational Infrastructure (NCI) Mass Data Storage System (MDSS) command module. 

Message Passing Interface Issues
================================

This module uses :obj:`subprocess.Popen` objects to execute
MDSS commands. Forking processes inside MPI can be problematic, for example
see  `OpenMPI FAQ 38 <http://www.open-mpi.org/faq/?category=openfabrics#ofa-fork>`_
and `OpenMPI FAQ 19 <http://www.open-mpi.org/faq/?category=tuning#fork-warning>`_
for OpenMPI related issues. If segmentation-faults occur during the fork of
the MDSS processes, one can try to exclude the Infini-band transport layer (openib)
by supplying the following mpirun (OpenMPI) argument::

   mpirun -np 8 --mca btl self,sm,tcp python script.py

Forking is safe under recent linux kernels, recent OpenMPI versions
and using only the *self*, *shared-memory* and *TCP* Byte-Transport-Layers (BTLs). 

Data
====

.. autodata:: haveMdss

Functions
=========

.. autosummary::
   :toctree: generated/

   du - Returns MDSS disk usage info for specified files/directories. 
   exists - Returns whether a file/directory exists on MDSS.
   get - Copy data from MDSS.
   isfile - Returns whether a specified path exists and is a file or link to a file.
   isdir - Returns whether a specified path exists and is a directory or link to a directory.
   listdir - List contents of a specified MDSS directory.
   mkdir - Create a specified MDSS directory.
   makedirs - Creates MDSS directory with parent directories as necessary.
   mv - Rename/move files/directories within MDSS.
   put - Copy data to MDSS.
   remove - Remove files from MDSS.
   rmdir - Remove empty directory from MDSS.
   rmtree - Recursively remove directories from MDSS.
   stage - Notify MDSS of files which are soon to be copied to the local file-system. 
   
   setDefaultProject - set the default project identifier.
   getDefaultProject - returns the default project identifier.
"""

import mango.mpi as mpi
logger, rootLogger = mpi.getLoggers(__name__)
import subprocess
import sys
import shutil
import os
import os.path
import glob
import re

def _locWhich(cmd, mode=os.F_OK | os.X_OK, path=None):
    """Given a command, mode, and a PATH string, return the path which
    conforms to the given mode on the PATH, or None if there is no such
    file.

    `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
    of os.environ.get("PATH"), or can be overridden with a custom search
    path.

    """
    # Check that a given file can be accessed with the correct mode.
    # Additionally check that `file` is not a directory, as on Windows
    # directories pass the os.access check.
    def _access_check(fn, mode):
        return (os.path.exists(fn) and os.access(fn, mode)
                and not os.path.isdir(fn))

    # If we're given a path with a directory part, look it up directly rather
    # than referring to PATH directories. This includes checking relative to the
    # current directory, e.g. ./script
    if os.path.dirname(cmd):
        if _access_check(cmd, mode):
            return cmd
        return None

    if path is None:
        path = os.environ.get("PATH", os.defpath)
    if not path:
        return None
    path = path.split(os.pathsep)

    if sys.platform == "win32":
        # The current directory takes precedence on Windows.
        if not os.curdir in path:
            path.insert(0, os.curdir)

        # PATHEXT is necessary to check on Windows.
        pathext = os.environ.get("PATHEXT", "").split(os.pathsep)
        # See if the given file matches any of the expected path extensions.
        # This will allow us to short circuit when given "python.exe".
        # If it does match, only test that one, otherwise we have to try
        # others.
        if any(cmd.lower().endswith(ext.lower()) for ext in pathext):
            files = [cmd]
        else:
            files = [cmd + ext for ext in pathext]
    else:
        # On other platforms you don't have things like PATHEXT to tell you
        # what file suffixes are executable, so just pass on cmd as-is.
        files = [cmd]

    seen = set()
    for dir in path:
        normdir = os.path.normcase(dir)
        if not normdir in seen:
            seen.add(normdir)
            for thefile in files:
                name = os.path.join(dir, thefile)
                if _access_check(name, mode):
                    return name
    return None


if (hasattr(shutil, 'which')):
    _which = shutil.which
else:
    _which = _locWhich


_defaultProject=None

def _getProjectArg(project):
    if (project == None):
        if (_defaultProject != None):
            project = _defaultProject
        else:
            if ("PROJECT" in os.environ.keys()):
                project = os.environ["PROJECT"]
    if (project != None):
        arg = ["-P" + str(project),]
    else:
        arg = []
    
    return arg

def _getListDirAllArg(all):
    arg = []
    if (all != None):
        if (isinstance(all, bool) and all):
            arg += ["-a"]
        elif (isinstance(all, str) and ((all == "a") or (all.lower() == "all"))):
            arg += ["-a"]
        elif (isinstance(all, str) and ((all == "A") or (all.lower() == "almost-all"))):
            arg += ["-A"]
    return arg

def _getListDirAppendTypeArg(appendType):
    arg = []
    if (appendType):
        arg += ["-F",]

    return arg

def _getListDirDereferenceArg(dereference):
    arg = []
    if (dereference):
        arg += ["-L",]

    return arg

def _getListDirDirectoryOnlyArg(directoryOnly):
    arg = []
    if (directoryOnly):
        arg += ["-d",]

    return arg

def _getRecursiveArg(recursive):
    if (recursive):
        arg = ["-r",]
    else:
        arg = []
    
    return arg

def _getWaitArg(wait):
    if (wait):
        arg = ["-w",]
    else:
        arg = []
    
    return arg

def _getModeArg(mode):
    if (mode != None):
        arg = ["-m " + str(mode),]
    else:
        arg = []
    
    return arg

def _getMkDirCreateParentsArg():
    return ["-p",]

def _getRmTreeRecursiveArg():
    return ["-r",]

def _getFileListArgs(files):
    if (isinstance(files, str)):
        fileList = [files,]
    else:
        fileList = list(files)
    
    return fileList

_mdssExecutablePath = _which("mdss")
haveMdss = (_mdssExecutablePath != None) #: Indicates whether the "mdss" executable could be located.

class MdssCommand(object):
    """
    Class for executing MDSS commands.
    """
    def __init__(self, commandStr, project=None, args=[]):
        self.executableName = _mdssExecutablePath
        self.commandStr     = commandStr
        self.project        = project
        self.args           = args
    
    def execute(self, wait=True):
        """
        Uses :obj:`subprocess.Popen` to execute the relevant MDSS command.
        """
        stderr = subprocess.PIPE
        stdout = subprocess.PIPE
        cmdList = [self.executableName,] + _getProjectArg(self.project) + [self.commandStr, ] + self.args
        logger.debug("Executing subprocess with args=%s" % (cmdList,))
        p = \
            subprocess.Popen(
                args=cmdList,
                stderr = stderr,
                stdout = stdout
            )
        if (wait):
            p.wait()
            if (p.returncode != 0):
                stdoutdata,stderrdata = p.communicate()
                if (len(stderrdata.strip()) > 0):
                    errData = stderrdata
                else:
                    errData = stdoutdata
                raise Exception(errData)
        
        return p

def setDefaultProject(project=None):
    """
    Sets the default project string for MDSS commands.
    
    :type project: :obj:`str`
    :param project: NCI project identifier.
    """
    _defaultProject = project

def getDefaultProject():
    """
    Returns the default project string for MDSS commands.
    
    :rtype: :obj:`str`
    :return: NCI project identifier.
    """
    p = _defaultProject
    if (p == None):
        p = os.environ["PROJECT"]
    return p

def _getCanonicalPath(pathName):
    """
    Remove trailing *slash* from path name.
    """
    parent,leaf = os.path.split(pathName)
    if (len(leaf) <= 0):
        pathName = parent
    return pathName

def exists(file, project=None):
    """
    Returns whether specified file/directory exists
    on the MDSS file system.
    
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).
    :rtype: :obj:`bool`
    :return: :samp:`True` if :samp:`file` is a file or directory
       (or link to a file or directory) which exists in MDSS.
    """
    doesExist = False
    file = _getCanonicalPath(file)
    try:
        lsList = _ls(file, project=project, all="all", appendType=True, dereference=True, directoryOnly=True)
        doesExist = True
    except Exception as e:
        regEx = re.compile("(.*)\\s*no\\s+such\\s+file\\s+or\\s+directory\\s*(.*)")
        if (regEx.match(str(e).lower()) != None):
            doesExist = False
        else:
            raise
    return doesExist

def isfile(file, project=None):
    """
    Returns whether specified file is a plain file or link to a file.

    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).
    :rtype: :obj:`bool`
    :return: :samp:`True` if :samp:`file` is a plain file
        (or link to a plain file) which exists in MDSS.
    """
    isFile = False
    file = _getCanonicalPath(file)
    try:
        lsList = _ls(file, project=project, all="all", appendType=True, dereference=True, directoryOnly=True)
        if ((len(lsList) == 1) and ((file in lsList) or ((file+'*') in lsList))):
            isFile = True
    except Exception as e:
        regEx = re.compile("(.*)\\s*no\\s+such\\s+file\\s+or\\s+directory\\s*(.*)")
        if (regEx.match(str(e).lower()) != None):
            isFile = False
        else:
            raise
    return isFile

def isdir(file, project=None):
    """
    Returns whether specified file is a directory or link to a directory.

    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).
    :rtype: :obj:`bool`
    :return: :samp:`True` if :samp:`file` is a directory (or link to a directory) which exists in MDSS.
    """
    isDir = False
    file = _getCanonicalPath(file)
    try:
        lsList = _ls(file, project=project, all="all", appendType=True, dereference=True, directoryOnly=True)
        if ((len(lsList) == 1) and (os.path.join(file, "") in lsList)):
            isDir = True
    except Exception as e:
        regEx = re.compile("(.*)\\s*no\\s+such\\s+file\\s+or\\s+directory\\s*(.*)")
        if (regEx.match(str(e).lower()) != None):
            isDir = False
        else:
            raise
    return isDir

def _ls(dir=None, project=None, all=False, appendType=False, dereference=False, directoryOnly=False):
    """
    Lists file(s) in specified MDSS directory.
    
    :type dir: :obj:`str`
    :param dir: MDSS directory path for which files are listed.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).
    :type all: :obj:`bool` or :obj:`str`
    :param all: If :samp:`True` or :samp:`"all"` lists files/directories whose names begin with '.'.
       If :samp:`almost-all` lists files/directories whose names begin with '.' but not
       the :samp:`"."` and :samp:`".."` entries.
    :type appendType: :obj:`bool`
    :param appendType: If :samp:`True` each name in the listing will have a character appended
       which indicates the type of *file*.
    :type dereference: :obj:`bool`
    :param dereference: If :samp:`True` symbolic links are dereferenced in the listing.
    :type directoryOnly: :obj:`bool`
    :param directoryOnly: If :samp:`True` only list directory name and not directory contents.
    
    :rtype: :obj:`list` of :obj:`str`
    :return: MDSS directory listing.
    """
    args = ["-1"] # Separate listed entries with newline, one entry per line.
    args += _getListDirAllArg(all)
    args += _getListDirDirectoryOnlyArg(directoryOnly)
    args += _getListDirAppendTypeArg(appendType)
    args += _getListDirDereferenceArg(dereference)
    if (dir != None):
        args += [dir,]
    else:
        args = []
    p = MdssCommand(commandStr="ls", project=project, args=args).execute()
    return p.communicate()[0].split("\n")[0:-1] # exclude the last newline
    
def listdir(dir=None, project=None, all=False, appendType=False, dereference=False):
    """
    Lists file(s) in specified MDSS directory.
    
    :type dir: :obj:`str`
    :param dir: MDSS directory path for which files are listed.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).
    :type all: :obj:`bool` or :obj:`str`
    :param all: If :samp:`True` or :samp:`"all"` lists files/directories whose names begin with '.'.
       If :samp:`almost-all` lists files/directories whose names begin with '.' but not
       the :samp:`"."` and :samp:`".."` entries.
    :type appendType: :obj:`bool`
    :param appendType: If :samp:`True` each name in the listing will have a character appended
       which indicates the type of *file*.
    :type dereference: :obj:`bool`
    :param dereference: If :samp:`True` symbolic links are dereferenced in the listing.
    
    :rtype: :obj:`list` of :obj:`str`
    :return: MDSS directory listing.
    """
    return _ls(dir=dir, project=project, all=all, appendType=appendType, dereference=dereference)

def mkdir(dir, mode=None, project=None):
    """
    Creates MDSS directory.
    
    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: Directory path or sequence of directory
       paths to be created on MDSS. Parent directories must
       exist otherwise exception is raised.
    :type mode: :obj:`str`
    :param mode: Unix *mode* string for the permissions given to created directories.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).

    """
    args = []
    args += _getModeArg(mode) + _getFileListArgs(dir)
    p = MdssCommand(commandStr="mkdir", project=project, args=args).execute()


def makedirs(dirs, mode=None, project=None):
    """
    Creates MDSS directory with parent directories as necessary.
    
    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: Directory path or sequence of directory
       paths to be created on MDSS. Parent directories created
       as necessary.
    :type mode: :obj:`str`
    :param mode: Unix *mode* string for the permissions given to created directories.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).

    """
    args = []
    args += _getModeArg(mode)
    args += _getMkDirCreateParentsArg()
    args += _getFileListArgs(dirs)
    p = MdssCommand(commandStr="mkdir", project=project, args=args).execute()

def remove(files, project=None):
    """
    Removes specified files (not directories) from MDSS.
    
    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: File path or sequence of file
       paths to be removed from MDSS. Non-directories only,
       otherwise exception raised.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).

    """
    args = []
    args += _getFileListArgs(files)    
    p = MdssCommand(commandStr="rm", project=project, args=args).execute()

def rmdir(files, project=None):
    """
    Removes specified directories from MDSS. Directories must be empty.
    
    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: Directory path or sequence of directory
       paths to be removed from MDSS. Directories must be empty
       otherwise exception is raised.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).

    """
    args = []
    args += _getFileListArgs(files)  
    p = MdssCommand(commandStr="rmdir", project=project, args=args).execute()

def rmtree(files, project=None):
    """
    Recursively removes specified files/directories from MDSS.

    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: File/directory path or sequence of file/directory
       paths to be removed from MDSS.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).
    """
    args = _getRmTreeRecursiveArg()
    args += _getFileListArgs(files)
    p = MdssCommand(commandStr="rm", project=project, args=args).execute()

def stage(files, recursive=False, wait=False, project=None):
    """
    Tell the mass-data filesystem that the nominated files will be  accessed
    soon. mdss get automatically issues a stage comamnd before retrieving
    the files, so the main use for this command is for those cases where you
    know  you will need some files shortly (say, within the next half hour),
    but have some other processing to do first.
    
    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: MDSS file/directory path or sequence of file/directory
       paths to be staged.
    :type recursive: :obj:`bool`
    :param recursive: Whether directories are staged recursively from MDSS.
    :type wait: :obj:`bool`
    :param wait: Whether to wait for the files to be completely staged before returning.
       If :samp:`False` the :meth:`stage` method returns immediately.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).
    """
    args = _getRecursiveArg(recursive)
    args += _getWaitArg(wait)
    args += _getFileListArgs(files)
    p = MdssCommand(commandStr="stage", project=project, args=args).execute()

def get(files, dest=None, recursive=False, project=None):
    """
    Copies a file (or files) from MDSS to a specified location.
    
    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: File/directory path or sequence of file/directory
       paths to be copied from MDSS.
    :type dest: :obj:`str`
    :param dest: Destination file/directory path where files MDSS files are to be copied.
        If :samp:`None` uses current working directory as destination.
    :type recursive: :obj:`bool`
    :param recursive: Whether directories are copied recursively from MDSS.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).


    """
    args = _getRecursiveArg(recursive)
    args += _getFileListArgs(files)
    if (dest == None):
        dest = "."

    args += glob.glob(dest)
    p = MdssCommand(commandStr="get", project=project, args=args).execute()


def put(files, dest=None, recursive=False, wait=False, project=None):
    """
    Copies a file (or files) from a specified location to MDSS.
    
    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: File/directory path or sequence of file/directory
       paths to be copied to MDSS.
    :type dest: :obj:`str`
    :param dest: File/directory path where files are copied on MDSS.
       If :samp:`None` uses MDSS default project working directory as destination.
    :type recursive: :obj:`bool`
    :param recursive: Whether directories are copied recursively to MDSS.
    :type wait: :obj:`bool`
    :param wait: Whether to wait for the files to be copied from staging
       disk to tape.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).

    """
    args = _getRecursiveArg(recursive)
    args += _getWaitArg(wait)
    files = _getFileListArgs(files)
    fileList = []
    for fileName in files:
        fileList += glob.glob(fileName)
    args += fileList
    if (dest == None):
        dest = "."
    args += [dest,]
    p = MdssCommand(commandStr="put", project=project, args=args).execute()

def mv(files, dest, project=None):
    """
    Move/rename files/directories in MDSS.
    
    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: File/directory path or sequence of file/directory
       paths to be moved within MDSS.
    :type dest: :obj:`str`
    :param dest: File/directory path where files are copied on MDSS.
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).

    """
    args = []
    args += _getFileListArgs(files)
    args += [dest,]
    p = MdssCommand(commandStr="mv", project=project, args=args).execute()

def du(files=None, unit=None, project=None):
    """
    Disk usage for files/directories in MDSS.
    
    :type files: :obj:`str` or sequence of :obj:`str`
    :param files: File/directory path or sequence of file/directory
       paths whose MDSS sizes are to be calculated.
    :type unit: :obj:`str`
    :param unit: One of :samp:`"b"` byte, :samp:`"k"` kilo-byte, :samp:`"m"` mega-byte. 
    :type project: :obj:`str`
    :param project: NCI project identifier string, if :samp:`None`, uses default
       project (as returned from the :func:`getDefaultProject` function).
    :rtype: sequence of tuples
    :return: Sequence of :samp:`(path,size,sizeUnit)` tuples.
    """
    args = ["-s",]
    unitStr = ""
    if (unit == None):
        unit = ["-h"]
    elif (unit in ["b", "k", "m"]):
        unitStr = unit.upper()
        unit = ["-" + unit,]
    else:
        raise ValueError("Invalid unit=%s argument value." % str(unit))
    args += unit
    args += _getFileListArgs(files)
    p = MdssCommand(commandStr="du", project=project, args=args).execute()
    stdoutdata, stderrdata = p.communicate()
    entries = stdoutdata.split("\n")[0:-1]
    regEx = re.compile("([0-9.]+)(\\s*[^ \t]*\\s+)(.*)")
    rList = []
    for entry in entries:
        m = regEx.match(entry)
        if (m != None):
            unit = m.group(2).strip()
            if (len(unit) <= 0):
                unit = unitStr
            rList.append([m.group(3), float(m.group(1).strip()), unit])
    return rList

__all__ = [s for s in dir() if not s.startswith('_')]

