import mango
from ._CropTest                       import *
from ._DiscreteGaussianTest           import *
from ._DiscreteGaussianDownsampleTest import *
from ._GatherSliceTest                import *
from ._ResampleGridTest               import *
from ._SobelTest                      import *
from ._SubSampleTest                  import *
from ._NeighbourhoodFilterTest        import *
from ._MomentOfInertiaTest            import *
from ._SphericalHistogramTest         import *
from ._LabelTest                      import *
from ._DistanceTransformEdtTest       import *
from ._MaxCoveringRadiusTest          import *
from ._ConvexHullTest                 import *
from ._HistogramddTest                import *
from ._StructuringElementTest         import *

if (mango.haveRegistration):
    from .registrationTest import *

if (mango.haveRestricted):
    from .DiscreteGaussianTest import *