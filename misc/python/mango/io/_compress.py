import os
import os.path
import zipfile
import bz2
import gzip
import mimetypes

if (not mimetypes.inited):
    mimetypes.init()

if (not (".gz" in mimetypes.types_map.keys())):
    mimetypes.add_type("application/gzip", ".gz")

if (not (".bz2" in mimetypes.types_map.keys())):
    mimetypes.add_type("application/x-bzip2", ".bz2")


from mango import mpi

logger, rootLogger = mpi.getLoggers(__name__)

BZIP2=mimetypes.types_map[".bz2"] #: The bzip2 compression method.
GZIP=mimetypes.types_map[".gz"] #: The gzip (GNU zip) compression method.
#ZIP=mimetypes.types_map[".zip"]
_supportedCompressionTypes = (BZIP2, GZIP)

def _doUncompress(zipFilePath, cls, dest=None, preserve=True, readSz=16*1024*1024):
    uncompressor = cls(zipFilePath, mode="rb")
    if (dest == None):
        dest = os.path.splitext(zipFilePath)[0]
    if (os.path.isdir(dest)):
        dest = os.path.join(dest, os.path.splitext(os.path.split(zipFilePath)[1])[0])
    
    i = 0
    while (os.path.exists(dest + (".tmp%04d" % i))):
        i += 1
    tmpDest = dest + (".tmp%04d" % i)
    try:
        tmpDestFile = open(tmpDest, mode='wb')
        while (True):
            data = uncompressor.read(readSz)
            if (len(data) > 0):
                tmpDestFile.write(data)
            else:
                break
        tmpDestFile.close()
        os.rename(tmpDest, dest)
        if (not preserve):
            os.remove(zipFilePath)
    except:
        if (os.path.exists(tmpDest)):
            os.remove(tmpDest)
        raise
    
    return dest

def _doCompress(filePath, cls, zipExt, dest=None, preserve=True, readSz=16*1024*1024):
    if (dest == None):
        dest = filePath + zipExt
    if (os.path.isdir(dest)):
        dest = os.path.join(dest, os.path.split(filePath)[1] + zipExt)
    compressor = cls(dest, mode="wb")

    try:
        readFile = open(filePath, mode='rb')
        while (True):
            data = readFile.read(readSz)
            if (len(data) > 0):
                compressor.write(data)
            else:
                break
        compressor.close()
        readFile.close()
        if (not preserve):
            os.remove(filePath)
    except:
        del compressor
        if (os.path.exists(dest)):
            os.remove(dest)
        raise
    
    return dest

def _bunzip2(zipFilePath, dest=None, preserve=True):
    return _doUncompress(zipFilePath, dest=dest, cls=bz2.BZ2File, preserve=preserve)

def _gunzip(zipFilePath, dest=None, preserve=True):
    return _doUncompress(zipFilePath, dest=dest, cls=gzip.GzipFile, preserve=preserve)

def uncompress(zipFilePath, dest=None, preserve=True):
    """
    Uncompress a specified file. 
    
    :type zipFilePath: :obj:`str`
    :param zipFilePath: File to be uncompressed. Must have extension which
       indicates the compression method (i.e. extension is :samp:`'.bz2'` or :samp:`'.gz'`).
    :type dest: :obj:`str`
    :param dest: Name of uncompressed file
       If :samp:`dest==None`, file name is :samp:`zipFilePath` minus the
       extension related to the compression method.
    :type preserve: :obj:`bool`
    :param preserve: If :samp:`False`, original uncompressed file is deleted
       after compressed file is created.
    """
    zipFileExt = os.path.splitext(zipFilePath)[1].lower()
    uncompressedFileName = None
    if (len(zipFileExt) > 0) and (zipFileExt in mimetypes.types_map.keys()):
        zipMimeType = mimetypes.types_map[zipFileExt]
        if (zipMimeType == BZIP2):
            uncompressedFileName = _bunzip2(zipFilePath, dest, preserve=preserve)
        elif (zipMimeType == GZIP):
            uncompressedFileName = _gunzip(zipFilePath, dest, preserve=preserve)
        else:
            raise Exception(
                "Unsupported compression mime-type '%s' for file extension '%s'\n, supported types are %s"
                %
                (zipMimeType, zipFileExt, _supportedCompressionTypes)
            )
    else:
        raise Exception("Unknown compression-type/mime-type for file %s, try adding an extension." % ())
    
    return uncompressedFileName

def _bzip2(filePath, dest=None, preserve=True):
    return \
        _doCompress(
            filePath,
            dest=dest,
            cls=bz2.BZ2File,
            zipExt=".bz2",
            preserve=preserve
        )

def _gzip(filePath, dest=None, preserve=True):
    return \
        _doCompress(
            filePath,
            dest=dest,
            cls=gzip.GzipFile,
            zipExt=".gz",
            preserve=preserve
        )

def compress(filePath, dest=None, method=BZIP2, preserve=True):
    """
    Compress specified file. 
    
    :type filePath: :obj:`str`
    :param filePath: File to be compressed.
    :type dest: :obj:`str`
    :param dest: Name of compressed file
       If :samp:`dest==None`, file name is :samp:`filePath + ext` where :samp:`ext`
       is the extension related to the compression :samp:`method`.
    :type method: :obj:`str`
    :param method: Compression method used (e.g. :data:`BZIP2`, :data:`GZIP`).
    :type preserve: :obj:`bool`
    :param preserve: If :samp:`False`, original uncompressed file is deleted
       after compressed file is created.

    :rtype: :obj:`str`
    :return: Compressed file name.
    """

    zipMimeType = method
    compressedFileName = None
    if (zipMimeType == BZIP2):
        compressedFileName = _bzip2(filePath, dest, preserve=preserve)
    elif (zipMimeType == GZIP):
        compressedFileName = _gzip(filePath, dest, preserve=preserve)
    else:
        raise Exception(
            "Unsupported compression mime-type '%s', supported types are %s"
            %
            (zipMimeType, _supportedCompressionTypes)
        )
    return compressedFileName

class CompressJob(object):
    def __init__(self, fileName, dest=None, method=BZIP2, preserve=True):
        self.fileName = fileName
        self.dest = dest
        self.method = method
        self.preserve = preserve
    
    def __call__(self):
        logger.info("Compressing %s" % self.fileName)
        compressedFileName = compress(self.fileName, self.dest, method=self.method, preserve=self.preserve)
        logger.info("Compressed %s -> %s" % (self.fileName, compressedFileName))
        return compressedFileName

def _getFileList(dirPath, mpiComm=mpi.world):
    """
    Returns leaf-file-name listing for a specified directory.
    Under MPI executes blocking (barrier) and broadcast of
    file name list.
    
    :rtype: :obj:`list`
    :return: A :obj:`list` of file-name strings in directory :samp:`dirPath`. 
    """
    rootRank = 0
    if (mpi.haveMpi4py):
        mpiComm.barrier()
    
    fileNameList = None
    if (mpi.rank == rootRank):
        fileNameList = os.listdir(dirPath)
        fileNameList.sort()

    if (mpi.haveMpi4py):
        fileNameList = mpiComm.bcast(fileNameList, rootRank)
    
    return fileNameList

def compressDir(dirPath, dest=None, method=BZIP2, preserve=True, mpiComm=mpi.world):
    """
    Compress files in a specified directory. Can take advantage of
    MPI by compressing files per MPI process. 
    
    :type dirPath: :obj:`str`
    :param dirPath: Directory containing files which are to be compressed.
    :type dest: :obj:`str`
    :param dest: Directory where compressed files are written (must exist).
       If :samp:`dest==None`, files are compressed in directory :samp:`dirPath`.
    :type preserve: :obj:`bool`
    :param preserve: If :samp:`False`, original uncompressed file is deleted
       after compressed file is created.
    :type mpiComm: :obj:`mpi4py.MPI.Comm`
    :param mpiComm: Communicator for MPI parallelism.

    :rtype: :obj:`list` of :obj:`str`
    :return: List of the compressed file names.
    """

    fileNameList = _getFileList(dirPath, mpiComm)
    jobList = \
        [
             CompressJob(os.path.join(dirPath, fileName), dest, method, preserve)
             for
             fileName in fileNameList
        ]

    rList = [pair[0] for pair in executeJobs(jobList, mpiComm)]
    rList.sort()
    
    return rList


class UncompressJob(object):
    def __init__(self, zipFileName, dest=None, preserve=True):
        self.zipFileName = zipFileName
        self.dest = dest
        self.preserve = preserve
    
    def __call__(self):
        logger.info("Uncompressing %s" % self.zipFileName)
        uncompressedFileName = uncompress(self.zipFileName, self.dest, preserve=self.preserve)
        logger.info("Uncompressed %s -> %s" % (self.zipFileName, uncompressedFileName))
        
        return uncompressedFileName

def uncompressDir(zipDirPath, dest=None, preserve=True, mpiComm=mpi.world):
    """
    Uncompress files in a specified directory. Can take advantage of
    MPI by uncompressing files per MPI process. 
    
    :type zipDirPath: :obj:`str`
    :param zipDirPath: Directory containing compressed files.
    :type dest: :obj:`str`
    :param dest: Directory where uncompressed files are written (must exist).
       If :samp:`dest==None`, files are uncompressed in directory :samp:`zipDirPath`.
    :type preserve: :obj:`bool`
    :param preserve: If :samp:`False`, original compressed files are deleted
       after uncompressed file is created.
    :type mpiComm: :obj:`mpi4py.MPI.Comm`
    :param mpiComm: Communicator for MPI parallelism.

    :rtype: :obj:`list` of :obj:`str`
    :return: List of the uncompressed file names.
    """
    fileNameList = _getFileList(zipDirPath, mpiComm)
    jobList = \
        [
             UncompressJob(os.path.join(zipDirPath, fileName), dest, preserve)
             for
             fileName in fileNameList 
        ]

    rList = [pair[0] for pair in executeJobs(jobList, mpiComm)]
    rList.sort()
    
    return rList


def executeJobs(jobList, mpiComm=mpi.world):
    """
    MPI parallel execution of :samp:`jobList`. Assumes jobList
    is identical on all processes. Job elements are split equally
    among MPI processes.
    """
    if (mpi.haveMpi4py):
        mpiComm.barrier()
        rank = mpiComm.Get_rank()
        size = mpiComm.Get_size()
    else:
        rank = 0
        size = 1

    resultList = []
    for i in range(rank, len(jobList), size):
        resultList.append([jobList[i](), rank]) 
    
    if (mpi.haveMpi4py):
        results = mpiComm.allgather(resultList)
    else:
        results = [resultList]
    resultList = []
    for rList in results:
        resultList += rList
    
    return resultList

def _canUncompress(fileName):
    canUncompress = False
    try:
        ext = os.path.splitext(fileName)[1].lower()
        canUncompress = \
            (
                (ext in mimetypes.types_map.keys())
                and
                (mimetypes.types_map[ext] in _supportedCompressionTypes)
            )
    except:
        pass
    
    return canUncompress

def _splitCanUncompress(fileList):
    canUncompress = []
    canNotUncompress = []
    for fileName in fileList:
        if (_canUncompress(fileName)):
            canUncompress.append(fileName)
        else:
            canNotUncompress.append(fileName)
    return canUncompress, canNotUncompress

def _splitCanCompress(fileList):
    canNotCompress, canCompress = _splitCanUncompress(fileList)
    return canCompress, canNotCompress

def uncompressDdsData(ddsFilePath, dest=None, preserve=True, mpiComm=mpi.world):
    """
    Uncompresses :obj:`mango.Dds` files. A more forgiving uncompression function
    which ignores files which are already uncompressed.
    
    :type ddsFilePath: :obj:`str`
    :param ddsFilePath: A compressed netCDF (e.g. :samp:`.nc.bz2`) file or a
      directory containing compressed netCDF block/tile files.
    :type dest: :obj:`str`
    :param dest: Directory where uncompressed files are written (must exist).
       If :samp:`dest==None`, files are uncompressed in directory :samp:`ddsFilePath`
       (or in the same directory as :samp:`ddsFilePath` if :samp:`ddsFilePath`
       is a netCDF file).
    :type preserve: :obj:`bool`
    :param preserve: If :samp:`False`, original compressed files are deleted
       after uncompressed file is created.
    :type mpiComm: :obj:`mpi4py.MPI.Comm`
    :param mpiComm: Communicator for MPI parallelism.

    :rtype: :obj:`list` of :obj:`str`
    :return: List of the uncompressed file names
        (including any which were not compressed before the call to this function).
    """
    rootRank = 0
    if (mpi.haveMpi4py and (mpiComm != None)):
        mpiComm.barrier()

    if (not os.path.exists(ddsFilePath)):
        raise Exception("File/directory %s does not exist." % ddsFilePath)
    
    allFilesList = None
    uncompressibleList = None
    remainingList = None
    if (mpi.rank == rootRank):
        if (os.path.isdir(ddsFilePath)):
            allFilesList = [os.path.join(ddsFilePath, fileName) for fileName in os.listdir(ddsFilePath)]
        else:
            allFilesList = [ddsFilePath,]
        
        allFilesList.sort()
        uncompressibleList, remainingList = _splitCanUncompress(allFilesList)
        uncompressibleList.sort()
        remainingList.sort()
    
    if (mpi.haveMpi4py and (mpiComm != None)):
        allFilesList, uncompressibleList, remainingList = \
            mpiComm.bcast((allFilesList, uncompressibleList, remainingList), root=rootRank)

    jobList = \
        [
             UncompressJob(fileName, dest, preserve)
             for
             fileName in uncompressibleList
        ]

    uncompressedList = [pair[0] for pair in executeJobs(jobList, mpiComm)]
    allFilesList = uncompressedList + remainingList
    allFilesList.sort()

    return allFilesList

def compressDdsData(ddsFilePath, dest=None, method=BZIP2, preserve=True, mpiComm=mpi.world):
    """
    Compresses :obj:`mango.Dds` files. A more forgiving compression function
    which ignores files which are already compressed.
    
    :type ddsFilePath: :obj:`str`
    :param ddsFilePath: A netCDF (e.g. :samp:`.nc`) file or a
      directory containing netCDF block/tile files.
    :type dest: :obj:`str`
    :param dest: Directory where compressed files are written (must exist).
       If :samp:`dest==None`, files are compressed in directory :samp:`ddsFilePath`
       (or in the same directory as :samp:`ddsFilePath` if :samp:`ddsFilePath`
       is a netCDF file).
    :type preserve: :obj:`bool`
    :param preserve: If :samp:`False`, original uncompressed files are deleted
       after compressed file is created.
    :type mpiComm: :obj:`mpi4py.MPI.Comm`
    :param mpiComm: Communicator for MPI parallelism.

    :rtype: :obj:`list` of :obj:`str`
    :return: List of the compressed file names
        (including any which were already compressed before the call to this function).
    """
    rootRank = 0
    if (mpi.haveMpi4py and (mpiComm != None)):
        mpiComm.barrier()

    if (not os.path.exists(ddsFilePath)):
        raise Exception("File/directory %s does not exist." % ddsFilePath)
    
    allFilesList = None
    compressibleList = None
    remainingList = None
    if (mpi.rank == rootRank):
        if (os.path.isdir(ddsFilePath)):
            allFilesList = [os.path.join(ddsFilePath, fileName) for fileName in os.listdir(ddsFilePath)]
        else:
            allFilesList = [ddsFilePath,]
        
        allFilesList.sort()
        compressibleList, remainingList = _splitCanCompress(allFilesList)
        compressibleList.sort()
        remainingList.sort()
    
    if (mpi.haveMpi4py and (mpiComm != None)):
        allFilesList, compressibleList, remainingList = \
            mpiComm.bcast((allFilesList, compressibleList, remainingList), root=rootRank)

    jobList = \
        [
             CompressJob(fileName, dest, method, preserve)
             for
             fileName in compressibleList
        ]

    compressedList = [pair[0] for pair in executeJobs(jobList, mpiComm)]
    allFilesList = compressedList + remainingList
    allFilesList.sort()

    return allFilesList

__all__ = [s for s in dir() if not s.startswith('_')]
