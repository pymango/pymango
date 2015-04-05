"""
==============================
Input/Output (:mod:`mango.io`)
==============================

.. currentmodule:: mango.io

This module contains functions for input/output
of :obj:`mango.Dds` objects.

Variables
=========

.. autodata:: BZIP2

.. autodata:: GZIP

Functions
=========

.. autosummary::
   :toctree: generated/

   compress - Compress a specified file.
   compressDir - Compress all files in a specified directory.
   compressDdsData - Compress a '*.nc' file or compress all '*.nc' files in a directory.
   readDds - Read :obj:`mango.Dds` from file
   writeDds - Write :obj:`mango.Dds` to file
   uncompress - Compress a specified file.
   uncompressDir - Uncompress all files in a specified directory.
   uncompressDdsData - Uncompress a compressed '*.nc' file or uncompress all compressed '*.nc' files in a directory.
   splitext - Splits a :obj:`mango.Dds` netCDF file path into :samp:`(basename, extension)` string  pair.
   splitpath - Splits a :obj:`mango.Dds` netCDF file path into :samp:`(dir, prefix, suffix, extension)` string  tuple.     
"""

import mango
from ._compress import *
from ._ddsio    import *

BZIP2=_compress.BZIP2 #: The bzip2 compression method.
GZIP=_compress.GZIP #: The gzip (GNU zip) compression method.

__all__ = [s for s in dir() if not s.startswith('_')]

if (mango.haveRestricted):
    from ._io       import *
