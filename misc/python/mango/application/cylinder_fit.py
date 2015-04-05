__doc__ = \
"""
===============================================================================
Tools for fitting cylinders to an image (:mod:`mango.application.cylinder_fit`)
===============================================================================

.. currentmodule:: mango.application.cylinder_fit

Functions and classes for fitting cylinders to an image. Useful
for identifying cylindrical containers in a tomographic image.

This module can be run as a script as follows::

   mpirun -np 32 python -m mango.application.cylinder_fit -n 3 path/to/tomoHiRes_nc

which will performs a multi-resolution cylinder fit of 3 cylinders and
generates an image file :samp:`path/to/segmentedHiRes_CYL_nc` which contains
the cylinder boundaries. Help for other options::

   python -m mango.application.cylinder_fit --help

Functions
=========

.. autosummary::
   :toctree: generated/
   
   cylinder_fit_gradient_metric - Factory method for creating a gradient-vector based cylinder fit metric.
   cylinder_fit - Finds a specified number of cylinders in an image.
   cylinder_fit_multi_res - Multi-resolution cylinder fitting.

Classes
=======

.. autosummary::
   :toctree: generated/
   
   CylinderFitGradientMetric - Gradient-vector based cylinder fit metric.
"""

import mango
import mango.data
import mango.io
import mango.math
import mango.optimize
import mango.mpi as mpi

import scipy as sp
import numpy as sp

if (mango.haveRestricted):
    import sys
    if sys.platform.startswith('linux'):
        import DLFCN as dl
        _flags = sys.getdlopenflags()
        sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
        from mango.restricted import _mango_restricted_core as _mango_restricted_core_so
        sys.setdlopenflags(_flags)
    else:
        from mango.restricted import _mango_restricted_core as _mango_restricted_core_so

import logging
import scipy as sp
import numpy as np

from mango.utils import ModuleObjectFactory as _ModuleObjectFactory

logger, rootLogger = mpi.getLoggers(__name__)

if (mango.haveRestricted):
    _moduleObjectFactory = _ModuleObjectFactory(_mango_restricted_core_so)


if (mango.haveRestricted):
    # Sphinx doc class only
    class CylinderFitGradientMetric(_mango_restricted_core_so._GradientCylinderMetric_float32):
        __doc__ = _mango_restricted_core_so._GradientCylinderMetric_float32.__doc__

class NonDistributedGradCylFitMetric:
    def __init__(self, metric):
        self.metric = metric
    
    def __getattr__(self, name):
        return getattr(self.metric, name)
    
    def __call__(self, x):
        return -sp.absolute(self.metric(x)) + self.metric.calcParameterPenalty(x)

class DistributedGradCylFitMetric(mango.optimize.DistributedMetricEvaluator):
    """
    Over-rides the :meth:`mango.optimize.DistributedMetricEvaluator.calcReduction`
    method, to return the negated absolute value of the MPI reduced metric value.
    """

    def calcReduction(self, localVal):
        """
        Returns the negated absolute value of the MPI reduced :samp:`localVal`.
        
        :type localVal: reducable :obj:`object`
        :param localVal: Local MPI process metric value, this value is MPI sum-reduced.
        """
        if (self.comm != None):
            metricVal = self.comm.reduce(localVal, root=self.root, op=mpi.SUM)
        else:
            metricVal = localVal

        if (metricVal != None):
            metricVal = -sp.absolute(metricVal)
            if (hasattr(self, "x") and (not (self.x is None)) and hasattr(self, "calcParameterPenalty")):
                metricVal += self.calcParameterPenalty(self.x)
        return metricVal


def cylinder_fit_gradient_metric(
    perimeter_sigma=1.0,
    num_slices=10,
    num_points_per_slice=90,
    normalise_gradients=False,
    dtype=None
):
    """
    Factory function for creating a metric which can be used to fit a cylinder
    to an image/gradient-vector-image. See the :obj:`CylinderFitGradientMetric`
    class constructor for argument doc.
    """
    if (dtype == None):
        if (mango.haveFloat16):
            dtype = "float16"
        else:
            dtype = "float32"

    metric =\
        _moduleObjectFactory.create(
            "_GradientCylinderMetric_",
            dtype = dtype,
            perimeter_sigma = perimeter_sigma,
            num_slices = num_slices,
            num_points_per_slice = num_points_per_slice,
            normalise_gradients = normalise_gradients
        )

    return metric

def cylinder_fit(
    input,
    numcyl,
    retcylimg=False,
    distributedMetricEvaluation=True,
    metric = None,
    radius_range_percent = (50.0, 100.0),
    centre_range_percent = (-8.0,   8.0),
    metricThreshRelTol   = 0.25
):
    """
    Searches for :samp:`{numcyl}` cylinders in :samp:`{input}` image
    which are approximately aligned with the :samp:`{input}` image z-axis.
    
    Perform exhaustive search over a grid of cylinder *radius*, *centre-x*, *centre-y*
    values, then uses numerical optimization (:func:`scipy.optimize.minimize`) on the best
    of the exhaustive search results to refine the cylinder fits.
    
    Returns list of cylinder parameters in global/absolute coordinates and in
    length units of :samp:`input.md.getVoxelSizeUnit()`.
    
    :type input: :obj:`mango.Dds`
    :param input: Search for cylinder boundaries in the gradient-vector
       image of this :samp:`{input}` image.
    :type numcyl: :obj:`mango.Dds`
    :param numcyl: Number of cylinders to find.
    :type retcylimg: :obj:`bool`
    :param retcylimg: If :samp:`True`, additionally returns a segmented image
       with the boundary of the cylinder fits labeled from :samp:`0` to :samp:`{numcyl}-1`.
    :type distributedMetricEvaluation: :obj:`bool`
    :param distributedMetricEvaluation: If :samp:`True`, the metric is evaluated in
       MPI distributed fashion (data parallelism). If :samp:`False`, the entire gradient-vector
       image is gathered to each MPI process, and task-parallelism is used as the paradigm to
       parallelise the cylinder fit searching.
    :type metric: :obj:`CylinderFitGradientMetric`
    :param metric: Metric object used to identify cylinder parameters
       which indicate the presence of a cylindrical boundary in the :samp:`input` image.
    :type radius_range_percent: 2 or 3 sequence of :obj:`float`
    :param radius_range_percent: Indicates the cylinder-radius exhaustive search grid
       as :samp:`(min_percent, max_percent, percent_step)`. This is a percentage
       of the minimum :samp:`{input}.shape` component (i.e. :samp:`numpy.min({input}.shape)`). 
       If the percent step size is omitted, then a percent equivalent of 2-voxels is chosen
       as the step-size.
    :type centre_range_percent: 2 or 3 sequence of :obj:`float`
    :param centre_range_percent: Indicates the cylinder-centre exhaustive search grid
       as :samp:`(min_percent, max_percent, percent_step)`. This is a percentage
       of the minimum :samp:`{input}.shape` component (i.e. :samp:`numpy.min({input}.shape)`). 
       If the percent step size is omitted, then a percent equivalent of 2-voxels is chosen
       as the step-size.
    :type metricThreshRelTol: :obj:`float`
    :param metricThreshRelTol: Relative tolerance for truncating the
       exhaustive search metric values. Ignore exhaustive search parameters
       which give metric values that have relative difference greater
       than :samp:`{metricThreshRelTol}` of the best (lowest) metric value in
       the entire search.
    :rtype: :obj:`list` of :samp:`[metricVal, cylinderParameters]` pairs
    :return: List of pairs :samp:`pairList = [[metricVal0, cylPrms0], [metricVal1, cylPrms1], ...]`
       where :samp:`cylPrms0=[[centrez0, centrey0, centrex0], radius0, axisLength0, [axisz0, axisy0, axisx0]]`, :samp:`cylPrms1=[[centrez1, centrey1, centrex1], radius1, axisLength1, [axisz1, axisy1, axisx1]]`, etc.
       If  :samp:`{retcylimg}` is :samp:`True`, additionally returns segmented :obj:`mango.Dds`
       image labeled with cylinder boundaries (i.e. returns :samp:`pairList, segDds`).
       The cylinder parameters are global/absolute coordinates and in
       length units of :samp:`input.md.getVoxelSizeUnit()`
    """
    fitter = \
        _CylinderFitter(
            input                       = input,
            numcyl                      = numcyl,
            distributedMetricEvaluation = distributedMetricEvaluation,
            metric                      = metric,
            radius_range_percent        = radius_range_percent,
            centre_range_percent        = centre_range_percent,
            axis_range_degrees          = None, # axis_range_degrees,
            metricThreshRelTol          = metricThreshRelTol
        )
    fitList = fitter.fit()
    retVal = fitList
    if (retcylimg):
        cylPrmList = [cylfit[1] for cylfit in fitList]
        cylImg = fitter.generateCylinderBoundaryImage(cylPrmList)
        retVal = [retVal, cylImg]
    
    return retVal

def cylinder_fit_multi_res(
    input,
    numcyl,
    retcylimg=False,
    resolutions=[128, 256, 1024],
    metric = None,
    radius_range_percent = (50.0, 100.0),
    centre_range_percent = (-8.0,   8.0),
    metricThreshRelTol   = 0.25
):
    """
    Multi-resolution version of :func:`cylinder_fit`.
    
    :type input: :obj:`mango.Dds`
    :param input: Search for cylinder boundaries in the gradient-vector
       image of this :samp:`{input}` image.
    :type numcyl: :obj:`mango.Dds`
    :param numcyl: Number of cylinders to find.
    :type retcylimg: :obj:`bool`
    :param retcylimg: If :samp:`True`, additionally returns a segmented image
       with the boundary of the cylinder fits labeled from :samp:`0` to :samp:`{numcyl}-1`.
    :type resolutions: sequence of :obj:`int`
    :param resolutions: The sequence of image shapes for each cylinder fit resolution.
        Exhaustive search and refinement is performed at the lowest resolution,
        just refinement at the higher resolutions.
    :type metric: :obj:`CylinderFitGradientMetric`
    :param metric: Metric object used to identify cylinder parameters
       which indicate the presence of a cylindrical boundary in the :samp:`input` image.
    :type radius_range_percent: 2 or 3 sequence of :obj:`float`
    :param radius_range_percent: Indicates the cylinder-radius exhaustive search grid
       as :samp:`(min_percent, max_percent, percent_step)`. This is a percentage
       of the minimum :samp:`{input}.shape` component (i.e. :samp:`numpy.min({input}.shape)`). 
       If the percent step size is omitted, then a percent equivalent of 2-voxels is chosen
       as the step-size.
    :type centre_range_percent: 2 or 3 sequence of :obj:`float`
    :param centre_range_percent: Indicates the cylinder-centre exhaustive search grid
       as :samp:`(min_percent, max_percent, percent_step)`. This is a percentage
       of the minimum :samp:`{input}.shape` component (i.e. :samp:`numpy.min({input}.shape)`). 
       If the percent step size is omitted, then a percent equivalent of 2-voxels is chosen
       as the step-size.
    :type metricThreshRelTol: :obj:`float`
    :param metricThreshRelTol: Relative tolerance for truncating the
       exhaustive search metric values. Ignore exhaustive search parameters
       which give metric values that have relative difference greater
       than :samp:`{metricThreshRelTol}` of the best (lowest) metric value in
       the entire search.
    :rtype: :obj:`list` of :samp:`[metricVal, cylinderParameters]` pairs
    :return: List of pairs :samp:`pairList = [[metricVal0, cylPrms0], [metricVal1, cylPrms1], ...]`
       where :samp:`cylPrms0=[[centrez0, centrey0, centrex0], radius0, axisLength0, [axisz0, axisy0, axisx0]]`, :samp:`cylPrms1=[[centrez1, centrey1, centrex1], radius1, axisLength1, [axisz1, axisy1, axisx1]]`, etc.
       If  :samp:`{retcylimg}` is :samp:`True`, additionally returns segmented :obj:`mango.Dds`
       image labeled with cylinder boundaries (i.e. returns :samp:`pairList, segDds`).

    """
    resolutions = list(resolutions)
    resolutions.sort()
    dsFactors = np.min(input.shape)/sp.array(resolutions, dtype="float64")
    if (dsFactors[-1] < 1):
        dsFactors[-1] = 1.0
    dsFactors = sp.array([dsFactor for dsFactor in dsFactors if dsFactor >= 1], dtype=dsFactors.dtype)
    i = 0
    rootLogger.info("Downsample factors = %s." % (dsFactors,))
    rootLogger.info("Downsampling (resolution=%s) input.shape=%s to shape=%s..." % (resolutions[i], input.shape, input.shape/dsFactors[i]))
    input.updateOverlapRegions()
    input.mirrorOuterLayersToBorder(False)
    dsInput = mango.image.gaussian_downsample(input, factor=[1.0/dsFactors[0],]*3)
    dsInput.updateHaloRegions()
    dsInput.mirrorOuterLayersToBorder(False)
    
    # Do data-parallel parallelism for all resolutions, having a copy of
    # the global downsampled gradient-vector image on all processes tends
    # to be too memory hungary.
    distributedMetricEvaluation = True
    rootLogger.info("Done downsampling, new shape=%s." % (dsInput.shape, ))
    cylPrmList = \
        cylinder_fit(
            input = dsInput,
            numcyl=numcyl,
            retcylimg = False,
            distributedMetricEvaluation = distributedMetricEvaluation,
            metric = metric,
            radius_range_percent = radius_range_percent,
            centre_range_percent = centre_range_percent,
            metricThreshRelTol = metricThreshRelTol
        )
    rootLogger.info("Lowest resolution fit cylinders=:\n%s" % ("\n".join(map(str, cylPrmList))))
    del dsInput
    dsInput = None
    fitter = None
    distributedMetricEvaluation = True
    i = 1
    if (metric == None):
        metric = cylinder_fit_gradient_metric()

    for dsFactor in dsFactors[1:]:
        metric.setNumSlices(metric.getNumSlices() + 10)
        metric.setNumPointsPerSlice(int(metric.getNumPointsPerSlice()*1.25))
        if (dsFactor > 1):
            rootLogger.info("Downsampling (resolution=%s) input.shape=%s to shape=%s..." % (resolutions[i], input.shape, input.shape/dsFactor))
            dsInput = mango.image.gaussian_downsample(input, factor=[1.0/dsFactor,]*3)
            rootLogger.info("Done downsampling, new shape=%s." % (dsInput.shape, ))
        else:
            rootLogger.info("Full resolution refinement input.shape=%s..." % (input.shape,))
            dsInput = input
        if (input.mpi.comm != None):
            input.mpi.comm.barrier()
        dsInput.updateHaloRegions()
        dsInput.mirrorOuterLayersToBorder(False)
        rootLogger.info("Cylinder-fit refinement for resolution image shape=%s." % (dsInput.shape,))
        fitter = \
            _CylinderFitter(
                input                       = dsInput,
                numcyl                      = numcyl,
                distributedMetricEvaluation = distributedMetricEvaluation,
                metric                      = metric,
                radius_range_percent        = radius_range_percent,
                centre_range_percent        = centre_range_percent,
                axis_range_degrees          = None,
                metricThreshRelTol          = metricThreshRelTol
            )
        if (i >= (len(dsFactors)-1)):
            fitter.minimize_options['xtol'] = 1.0e-4
            fitter.minimize_options['ftol'] = (fitter.minimize_options['xtol'])**2
        cylPrmList = fitter.refineFit(cylPrmList)
        rootLogger.info("Done cylinder-fit refinement for resolution image shape=%s." % (dsInput.shape,))
        rootLogger.info("Fit cylinders=:\n%s" % ("\n".join(map(str, cylPrmList))))
        i+=1
    
        del dsInput, fitter
        dsInput = None
        fitter = None

    retVal = cylPrmList
    if (retcylimg):
        cylPrmList = [cylfit[1] for cylfit in cylPrmList]
        cylImg = mango.copy(input, mtype="segmented")
        cylImg.setAllToValue(cylImg.mtype.maskValue())
        cylImg = fill_cylinder_boundaries(cylPrmList, cylImg)

        retVal = [retVal, cylImg]
    
    return retVal

class _SsOptimizer (mango.optimize.SingleStartOptimizer):
    def __init__(self, metric, minimize_options):
        self.metric = metric
        self.minimize_options = minimize_options
        self.minNumOptimizations = 3

    def copyResultInfo(self, spMinimizeRes, optimResult):
        optimResult.cost           = spMinimizeRes.fun
        optimResult.optim          = spMinimizeRes.x
        optimResult.numIterations  = spMinimizeRes.nit
        optimResult.numFuncEvals   = spMinimizeRes.nfev

    def optimize(self, resultList):
        """
        Optimization for set of single-radius metric evaluations.
        """
        bestResult = mango.optimize.OptimizeResult()

        if ((resultList != None) and (len(resultList) > 0)):
            resultList.sort(key = lambda resultPair: resultPair[0])
            
            currResult = resultList[0]
            res = sp.optimize.minimize(self.metric, currResult[1], method="Powell", options=self.minimize_options)
            rootLogger.debug("Optimized result = %s" % (res,))
            self.copyResultInfo(res, bestResult)
            bestResult.start = currResult[1]
            
            minNumOptimizations = min([self.minNumOptimizations, len(resultList)])
            numOptimizations = 1
            for currResult in resultList[1:]:
                res = sp.optimize.minimize(self.metric, currResult[1], method="Powell", options=self.minimize_options)
                rootLogger.debug("Optimized result = %s" % (res,))
                numOptimizations += 1
                if (res.fun <= bestResult.cost):
                    self.copyResultInfo(res, bestResult)
                    bestResult.start = currResult[1]
                elif (numOptimizations >= minNumOptimizations):
                    break
        return bestResult

def fill_cylinder_boundaries(cylprms, cylimg, annularWidth=None):
    if (annularWidth == None):
        anw = np.max(cylimg.md.getVoxelSize())*1.55
    else:
        anw = annularWidth
    
    cylImgMtype = None
    if hasattr(cylimg, "mtype"):
        cylImgMtype = cylimg.mtype
    cylimg.mtype = None

    for cylIdx in range(len(cylprms)):
        cylPrm = cylprms[cylIdx]
        c, r, al, a = cylPrm
        rMtx = mango.math.rotation_matrix_from_cross_prod(sp.array((1.0,0.,0.)), sp.array(a))
        mango.data.fill_annular_circular_cylinder(
            cylimg,
            c,
            r-0.5*anw,
            anw,
            al,
            fill=cylIdx,
            rotation=rMtx,
            unit=cylimg.md.getVoxelSizeUnit(),
            coordsys="abs"
        )
    if (cylImgMtype != None):
        cylimg.mtype = cylImgMtype
    
    return cylimg

class _CylinderFitter:
    """
    Searches for :samp:`numcyl` cylinders in :samp:`{input}` image
    which give the lowest :samp:`{metric}` value. 
    """
    def __init__(
        self,
        input,
        numcyl,
        distributedMetricEvaluation=True,
        metric = None,
        radius_range_percent = (50.0, 100.0),
        centre_range_percent = (-8.0,   8.0),
        axis_range_degrees   = ( 0.0,   0.0),
        metricThreshRelTol   = 1.0e-1
    ):
        self.input  = input
        self.numcyl = numcyl
        self.distributedMetricEvaluation = distributedMetricEvaluation
        self.metric = metric
        self.radius_range_percent = radius_range_percent
        self.centre_range_percent = centre_range_percent
        self.axis_range_degrees   = axis_range_degrees
        self.metricThreshRelTol   = metricThreshRelTol
        self.inVoxSz              = self.input.md.getVoxelSize()
        self.comm = input.mpi.comm
        self.root = 0
        self.minimize_options = {"xtol":0.01,"ftol":0.0001}

    def calcFullCylinderParameters(self, x):
        defaultPrms = self.metric.getDefaultX()
        prms = x
        fullPrms = prms.tolist() + self.metric.getDefaultX()[prms.size:].tolist()
        c  = [
            fullPrms[self.metric.CENTRE_Z_IDX],
            fullPrms[self.metric.CENTRE_Y_IDX],
            fullPrms[self.metric.CENTRE_X_IDX]
        ]
        c = sp.array(c)
        r  = fullPrms[self.metric.RADIUS_IDX]
        a  = [
            fullPrms[self.metric.AXIS_Z_IDX],
            fullPrms[self.metric.AXIS_Y_IDX],
            fullPrms[self.metric.AXIS_X_IDX]
        ]
        a = sp.array(a)
        aDot = sp.sum(a*a)
        if (aDot > 0):
            a /= sp.sqrt(aDot)

        al = fullPrms[self.metric.AXIS_LEN_IDX]
        
        return c, r, al, a

    def calcFitCylinderParameters(self, fullCylPrms):
        c, r, al, a = fullCylPrms
        prms = self.metric.getDefaultX()[0:6]

        az  = a[0]
        daz = self.metric.getDefaultX()[self.metric.AXIS_Z_IDX]
        af = daz/az
        prms[self.metric.RADIUS_IDX]   = r
        prms[self.metric.CENTRE_X_IDX] = c[2]
        prms[self.metric.CENTRE_Y_IDX] = c[1]
        prms[self.metric.AXIS_X_IDX]   = a[2]*af
        prms[self.metric.AXIS_Y_IDX]   = a[1]*af
        
        return prms

    def calcAnnularCylinderMaskParameters(self, x, annularWidth=None):
        c, r, al, a = self.calcFullCylinderParameters(x)
        if (annularWidth == None):
            anw = 4.0*self.metric.perimeter_sigma*np.min(self.inVoxSz)
        else:
            anw = annularWidth

        inr = r - 0.5*anw
        return c, inr, anw, al, a

    def generateCylinderBoundaryImage(self, cylPrmList, cylImg=None):
        if (cylImg == None):
            cylImgMtype = mango.mtype("segmented")
            cylImg = mango.empty_like(self.input, mtype=cylImgMtype)
            cylImg.setAllToValue(cylImg.mtype.maskValue())
        
        return fill_cylinder_boundaries(cylPrmList, cylImg)

    def calcExhaustiveSearchGrid(self):

        inVoxSz = self.input.md.getVoxelSize()
        inMinPt = self.input.origin*inVoxSz
        inMaxPt = (self.input.origin+self.input.shape)*inVoxSz
        radiusMax = np.min((inMaxPt-inMinPt)[1:3])*0.5
        radiusMin = 2*np.min(inVoxSz)
        radiusSpread = radiusMax-radiusMin
        radiusSearchRange = \
            (
                radiusMin + self.radius_range_percent[0]*radiusSpread/100.0,
                radiusMin + self.radius_range_percent[1]*radiusSpread/100.0
            )
        radiusMin, radiusMax = radiusSearchRange
        inCentrePt = 0.5*((inMaxPt+inMinPt)[1:3])
        cxySpread = np.min((inMaxPt-inMinPt)[1:3])
        cxMin = inCentrePt[1] + self.centre_range_percent[0]*cxySpread/100.0
        cxMax = inCentrePt[1] + self.centre_range_percent[1]*cxySpread/100.0
        cyMin = inCentrePt[0] + self.centre_range_percent[0]*cxySpread/100.0
        cyMax = inCentrePt[0] + self.centre_range_percent[1]*cxySpread/100.0
        
        defaultStepLen = None
        if (hasattr(self.metric, "perimeter_sigma")):
            defaultStepLen = 2.0*self.metric.perimeter_sigma*np.min(inVoxSz)
    
        if (len(self.radius_range_percent) > 2):
            self.radiusStepLen = radiusSpread*self.radius_range_percent[2]/100.0
        elif (defaultStepLen != None):
            self.radiusStepLen = defaultStepLen
        else:
            self.radiusStepLen = radiusSpread/100.0
    
        if (len(self.centre_range_percent) > 2):
            centreStepLen = cxySpread*self.centre_range_percent[2]/100.0
        elif (defaultStepLen != None):
            centreStepLen = defaultStepLen
        else:
            centreStepLen = cxySpread/100.0
    
        # do the initial exhaustive search over the
        # grid of (radius, centre_x, centre_y) parameter
        # values.
        xMin = sp.array([radiusMin, cxMin, cxMin, 0, 0])
        xMax = sp.array([radiusMax, cxMax, cxMax, 0, 0])
        xStp = sp.array([self.radiusStepLen, centreStepLen, centreStepLen, 1, 1])
        xShp = sp.array(np.maximum(sp.zeros_like(xMin), sp.floor((xMax-xMin)/xStp)), dtype="int64") + 1
        
        return xMin, xMax, xStp, xShp

    def maskGradientImageCylinders(self, cylList):
        if (len(cylList) > 0):
            c, inr, anw, al, a = \
                self.calcAnnularCylinderMaskParameters(cylList[-1][1])
            rootLogger.info(
                "Masking annular-cylinder (c, inr, anw, a, al)=(%s, %s, %s, %s, %s) in gradient image..."
                %
                (c, inr, anw, al, a)
            )
            self.metric.maskGradientAnnularCylinder(
                c,
                inr,
                anw,
                al,
                a
            )
            rootLogger.info("Done masking annular-cylinder .")

    def calcExhaustiveSearchResults(self, xMin, xMax, xStp, xShp):
        resultList = []
        numSearchPts = sp.prod(xShp)
        if (self.distributedMetricEvaluation):
            if ((self.comm == None) or (self.comm.Get_rank() == self.root)):
                for i in range(0, numSearchPts):
                    x = xMin + np.unravel_index(i, xShp)*xStp
                    resultList.append([self.metric(x), x])
                    rootLogger.debug(resultList[-1])
                self.metric.rootTerminate()
            else:
                self.metric.waitForEvaluate()
            if (self.comm != None):
                resultList = self.comm.bcast(resultList, self.root) 
        else:
            commSz = 1
            commRk = 0
            if (self.comm != None):
                commSz = self.comm.Get_size()
                commRk = self.comm.Get_rank()
            for i in range(commRk, numSearchPts, commSz):
                x = xMin + np.unravel_index(i, xShp)*xStp
                resultList.append([self.metric(x), x])
            if (self.comm != None):
                rListList = self.comm.allgather(resultList)
                resultList = []
                for rList in rListList:
                    resultList += rList

        return resultList

    def eliminatePoorResults(self, resultList):
        resultList.sort(key=lambda elem: elem[0])
        metricThresh = self.metricThreshRelTol*resultList[0][0]
        origNumResults = len(resultList)
        resultList = [result for result in resultList if result[0] < metricThresh]
        rootLogger.info(
            "Eliminated %s results from metric thresholding (orig num results = %s)."
            %
            (origNumResults-len(resultList), origNumResults)
        )
        return resultList

    def splitIntoPerRadiusResults(self, resultList):
        # Sort list, largest radius to smallest, smallest metric-value to largest
        # metric value for same-radius entries.
        radiusCmpTol = np.min(self.inVoxSz)*0.5
        resultList.sort(key=lambda elem: (-elem[1][self.metric.RADIUS_IDX], elem[0]))
        perRadiusLists = []
        if (len(resultList) > 0):
            perRadiusLists.append([resultList[0]])
            for result in resultList[1:]:
                if (
                    sp.absolute(
                        result[1][self.metric.RADIUS_IDX]
                        -
                        perRadiusLists[-1][0][1][self.metric.RADIUS_IDX]
                    )
                    >
                    radiusCmpTol
                ):
                    perRadiusLists.append([])
                    
                perRadiusLists[-1].append(result)
                    
        return perRadiusLists
        
    def calcBestRefinements(self, resultList):
        perRadiusResultLists = self.splitIntoPerRadiusResults(resultList)
        ssOptimizer = _SsOptimizer(self.metric, self.minimize_options)

        if (self.distributedMetricEvaluation and (self.comm != None)):
            if (self.comm.Get_rank() == self.root):
                rIdx = 0
                numResults = len(resultList)
                bestResults = []
                rootLogger.info("Optimizing...")
                for resultList in perRadiusResultLists:
                    optimResult = ssOptimizer.optimize(resultList)
                    bestResults.append([optimResult.cost, optimResult.optim])
        
                resultList = bestResults
                resultList.sort(key=lambda elem: elem[0])
                self.metric.rootTerminate()
            else:
                self.metric.waitForEvaluate()
            
            rootLogger.info("Done optimizing.")
        else:
            msOptimizer = mango.optimize.MultiStartOptimizer(ssOptimizer, self.root, self.comm)
            optimResults = msOptimizer.minimize(perRadiusResultLists)
            if (optimResults != None):
                resultList = [[optimResult.cost, optimResult.optim] for optimResult in optimResults]
            
        if (self.comm != None):
            resultList = self.comm.bcast(resultList, self.root)
        
        resultList.sort(key=lambda elem: elem[0])

        return resultList

    def initialiseMetric(self):
        rootLogger.info("Initialising metric object...")
        if (self.metric == None):
            self.metric = cylinder_fit_gradient_metric()
        self.metric.setGradientFromImage(self.input)
        if (self.distributedMetricEvaluation):
            # Distributed evaluation of metric, each MPI
            # process computes metric for local-sub-domain only
            # and the DistributedGradCylFitMetric wrapper
            # reduces the result to the self.metric.root ranked
            # process. 
            self.metric = DistributedGradCylFitMetric(self.metric, comm=self.input.mpi.comm)
            self.comm = self.metric.comm
            self.root = self.metric.root
        else:
            # Non-distributed evaluation of metric, give every MPI
            # process a copy of the entire gradient image.
            # Each MPI process computes metric for entire image.
            self.metric = NonDistributedGradCylFitMetric(self.metric)
            self.metric.gatherGradientDds()

        rootLogger.info("Done initialising metric object...")


    def fit(self):

        self.initialiseMetric()

        xMin, xMax, xStp, xShp = self.calcExhaustiveSearchGrid()
        numSearchPts = sp.prod(xShp)
    
        rootLogger.info("Grid search:")
        rootLogger.info("parameter x min   = %s" % (xMin,))
        rootLogger.info("parameter x max   = %s" % (xMax,))
        rootLogger.info("parameter x step  = %s" % (xStp,))
        rootLogger.info("parameter x shape = %s, %s metric evaluations" % (xShp, numSearchPts))
        rootLogger.info("Exhausive search...")
        cylList = []
        for cylIdx in range(0, self.numcyl):
            self.maskGradientImageCylinders(cylList)
            resultList = self.calcExhaustiveSearchResults(xMin, xMax, xStp, xShp)
            resultList = self.eliminatePoorResults(resultList)
            rootLogger.info("Done exhausive search.")

            resultList = self.calcBestRefinements(resultList)
            cylList.append(resultList[0])

        # Convert the parameter-vectors into 3 element centre-point, 3-element axis, etc.
        cylList = \
            [
                [resultPair[0], self.calcFullCylinderParameters(resultPair[1])]
                for
                resultPair in cylList
            ]

        return cylList

    def refineFit(self, cylFitList):
        self.initialiseMetric()
        inCylList = [[pair[0], self.calcFitCylinderParameters(pair[1])] for pair in cylFitList]
        cylList = []
        for result in inCylList:
            self.maskGradientImageCylinders(cylList)
            resultList = [result,]
            resultList = self.calcBestRefinements(resultList)
            cylList.append(resultList[0])

        # Convert the parameter-vectors into 3 element centre-point, 3-element axis, etc.
        cylList = \
            [
                [resultPair[0], self.calcFullCylinderParameters(resultPair[1])]
                for
                resultPair in cylList
            ]

        return cylList







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
            "Searches for cylinder-boundaries in the given image. Writes out" +\
            " a segmented data file with the cylinder boundaries identified as phases."
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
             'cmdLine':['-n', '--num-cylinders'],
             'dest':'numCylinders',
             'type':int,
             'metavar':'N',
             'default': 1,
             'action':'store',
             'help':"Look for this many cylinders in the image."
        }
    )

    argList.append(
        {
             'cmdLine':['-r', '--resolutions'],
             'dest':'resolutions',
             'type':str,
             'metavar':'R',
             'default':"256,512,1024",
             'action':'store',
             'help':"Comma separated list of num-voxels. Image is downsampled to" +\
                    " (approximately) these sizes for the cylinder fit. Exhaustive" +\
                    " search carried out at lowest resolution." +\
                    " Refinement cylinder fits" +\
                    " are carried out at the higher resolutions."
        }
    )

    argList.append(
        {
             'cmdLine':['--radius_trim_percent'],
             'dest':'radiusTrimPercent',
             'type':float,
             'metavar':'T',
             'default':95.0,
             'action':'store',
             'help':"When generating masked cylinder, trim the radius by this percentage."
        }
    )
    argList.append(
        {
             'cmdLine':['--axis_length_trim_percent'],
             'dest':'axisLengthTrimPercent',
             'type':float,
             'metavar':'T',
             'default':95.0,
             'action':'store',
             'help':"When generating masked cylinder, trim the axis-length (image z size) by this percentage."
        }
    )


    if (haveArgParse):
        parser = argparse.ArgumentParser(description=descStr)

        parser.add_argument(
            'fileName',
            metavar='F',
            type=str,
            nargs=1,
            help='Mango NetCDF image file/dir. Search for cylinders in this image.'
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

class ZeroToMaskFunc:
    def __init__(self, dstMtype):
        self.mskVal = dstMtype.maskValue()
    
    def __call__(self, x):
        if (x == 0):
            return self.mskVal
        return x

if (__name__ == "__main__"):
    import os
    import os.path
    import json

    argParser = getArgumentParser()
    if haveArgParse:
        args = argParser.parse_args()
        argv = args.fileName
    else:
        (args, argv) = argParser.parse_args()

    mpi.initialiseLoggers(
        [__name__, "mango.core", "mango.image", "mango.io", "mango.application"],
        logLevel=getattr(logging, args.loggingLevel)
    )
    mango.setLoggingVerbosityLevel(args.mangoVerbosity)
    ddsFileName = argv[0]
    ddsSplitFileName = mango.io.splitpath(ddsFileName)

    mpidims = [0,0,0]
    if ((mpi.size % 4) == 0):
        mpidims[1] = 2
        mpidims[2] = 2
    elif ((mpi.size % 2) == 0):
        mpidims[2] = 2

    resolutions = eval(args.resolutions)

    imgDds = mango.io.readDds(ddsFileName, halo=[8,8,8], mpidims=mpidims)
    
    cylPrmList, cylImg = \
        cylinder_fit_multi_res(
            imgDds,
            numcyl=args.numCylinders,
            retcylimg=True,
            resolutions=resolutions
        )

    mango.io.writeDds(
        os.path.join(
            ddsSplitFileName[0],
            "segmented" + ddsSplitFileName[2] + "_CYL" + ddsSplitFileName[3]
        ),
        cylImg
    )

    outputList = []
    for cylPrm in cylPrmList:
        metricVal = cylPrm[0]
        c, r, al, a = cylPrm[1]
        cylDict = dict()
        cylDict["metric_value"] = metricVal
        cylDict["centre"] = c.tolist()
        cylDict["radius"] = r
        cylDict["axis_length"] = al
        cylDict["axis"] = a.tolist()
        cylDict["length_unit"] = imgDds.md.getVoxelSizeUnit()
        
        outputList.append(cylDict)

    if ((imgDds.mpi.comm == None) or (imgDds.mpi.comm.Get_rank() == 0)):
        prmFileName = "cylprms_" + "segmented" + ddsSplitFileName[2] + "_CYL" + ".json"
        rootLogger.info("Writing cylinder parameters to file %s..." % prmFileName)
        open(prmFileName, 'w').write(json.dumps(outputList, indent=2, sort_keys=True))
    
    rootLogger.info("Generating masked image with smallest cylinder...")
    cylPrmList.sort(key=lambda e: e[1][1])
    c, r, al, a = cylPrmList[0][1]
    rootLogger.info("Generating masked image with smallest cylinder (c,r,al,a)=(%s, %s, %s, %s)..." % (c,r,al,a))
    
    mskCylPrms = [
        c,
        r*args.radiusTrimPercent/100.0,
        imgDds.shape[0]*imgDds.md.getVoxelSize()[0]*args.axisLengthTrimPercent/100.0,
        a
    ]
    c, r, al, a = mskCylPrms
    mskCylDict = dict()
    mskCylDict["centre"] = sp.array(c).tolist()
    mskCylDict["radius"] = r
    mskCylDict["axis_length"] = al
    mskCylDict["axis"] = sp.array(a).tolist()
    mskCylDict["length_unit"] = imgDds.md.getVoxelSizeUnit()

    if ((imgDds.mpi.comm == None) or (imgDds.mpi.comm.Get_rank() == 0)):
        prmFileName = "cylprms_" + "segmented" + ddsSplitFileName[2] + "_MSKcyl" + ".json"
        rootLogger.info("Writing mask cylinder parameters to file %s..." % prmFileName)
        open(prmFileName, 'w').write(json.dumps(mskCylDict, indent=2, sort_keys=True))

    rootLogger.info("Masking voxels outside cylinder...")
    rootLogger.info("Filling inside-cylinder voxels...")
    mskDds = mango.zeros_like(imgDds, mtype="segmented")
    mango.data.fill_circular_cylinder(
        mskDds,
        centre = mskCylPrms[0],
        radius = mskCylPrms[1],
        axislen = mskCylPrms[2],
        fill = 1,
        rotation = mango.math.rotation_matrix_from_cross_prod(sp.array([1.0,0.0,0.0]), sp.array(mskCylPrms[3])),
        unit = imgDds.md.getVoxelSizeUnit(),
        coordsys = "absolute"
    )
    rootLogger.info("Replacing zeros with mask value...")
    mskDds = mango.map_element_values(mskDds, ZeroToMaskFunc(mskDds.mtype))
    rootLogger.info("Copy mask voxels to original image...")
    mango.copy_masked(mskDds, imgDds)
    del mskDds
    rootLogger.info("Done masking voxels outside cylinder...")
    mango.io.writeDds(
        os.path.join(ddsSplitFileName[0], ddsSplitFileName[1] + ddsSplitFileName[2] + "_MSKcyl" + ddsSplitFileName[3]),
        imgDds
    )
    rootLogger.info("Auto-cropping, original shape = %s..." % (imgDds.shape,))
    imgDds = mango.image.auto_crop(imgDds)
    rootLogger.info("Done auto-cropping, new shape = %s..." % (imgDds.shape,))
    mango.io.writeDds(
        os.path.join(ddsSplitFileName[0], ddsSplitFileName[1] + ddsSplitFileName[2] + "_MSKcyl_AC" + ddsSplitFileName[3]),
        imgDds
    )
    

