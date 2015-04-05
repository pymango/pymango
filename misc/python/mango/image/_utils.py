
import mango.mpi as mpi
import numpy as np
import scipy as sp

import sys
if sys.platform.startswith('linux'):
    import DLFCN as dl
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
    from . import _mango_open_filters
    sys.setdlopenflags(_flags)
else:
    from . import _mango_open_filters


from mango.utils import ModuleObjectFactory as _ModuleObjectFactory

logger, rootLogger = mpi.getLoggers(__name__)

_moduleObjectFactory = _ModuleObjectFactory(_mango_open_filters)

def component_tree_1d(dtype):
    return _moduleObjectFactory.create("_MenottiComponentTree1d_", dtype)

def component_tree_1d_leaf_labels(f, cmp="greater"):
    ct = component_tree_1d(dtype=f.dtype)
    ct.generateComponentTree(f)
    ct.generateLeafLabels()
    
    return ct.getLeafLabels(), ct.getLeafIndices()
