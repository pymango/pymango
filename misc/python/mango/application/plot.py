__doc__ = \
"""
=======================================================================
Application Specific Plotting Utilities (:mod:`mango.application.plot`)
=======================================================================

.. currentmodule:: mango.application.plot

Application specific plotting utilities.

Functions
=========

.. autosummary::
   :toctree: generated/
   
   ternaryPlot - Creates basic "ternary" axes for plotting ternary data (or quaternary, etc.)

Classes
=======

.. autosummary::
   :toctree: generated/
   
   TernaryAxes - Creates :mod:`matplotlib` axes for plotting ternary data.
"""

import scipy as sp
import numpy as np

class TernaryAxes:
    """
    Object for generating matplotlib figure with *ternary* axes.
    """
    def __init__(
        self,
        basis = None,
        labels = None,
        sides = None,
        labelOffset = None,
        rotateLabels = None,
        edgeArgs = None,
        figArgs = None,
    ):
        self.basis        = basis
        self.labels       = labels
        self.sides        = sides
        self.labelOffset  = labelOffset
        self.rotateLabels = rotateLabels
        self.edgeArgs     = edgeArgs
        self.figArgs      = figArgs

    def createAxes(self, fig=None):
        """
        Returns a (:obj:`matplotlib.axes.Axes`, :obj:`matplotlib.figure.Figure`) pair
        with the plot containing labeled ternary axes.
        """
        
        from matplotlib import pyplot as plt
        
        if (fig == None):
            fig = plt.figure(**(self.figArgs))
        ax = fig.add_subplot(111)
   
        for i,l in enumerate(self.labels):
            if i >= self.sides:
                break
            x = self.basis[i,0]
            y = self.basis[i,1]
            if self.rotateLabels:
                angle = 180*sp.arctan(y/x)/sp.pi + 90
                if angle > 90 and angle <= 270:
                    angle = sp.mod(angle + 180,360)
            else:
                angle = 0
            ax.text(
                    x*(1 + self.labelOffset),
                    y*(1 + self.labelOffset),
                    l,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=angle
                )
    
        # Clear normal matplotlib axes graphics.
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_frame_on(False)

        # Plot border
        ax.plot(
            [self.basis[s,0] for s in list(range(self.sides)) + [0,]],
            [self.basis[s,1] for s in list(range(self.sides)) + [0,]],
            **(self.edgeArgs)
        )

        return ax, fig
    
def ternaryPlot(
            data,
            scaling=True,
            startAngle=89.9,
            rotateLabels=True,
            labels=('one','two','three'),
            sides=3,
            labelOffset=0.0666,
            edgeArgs={'color':'black','linewidth':2},
            figArgs = {'figsize':(8,8),'facecolor':'white','edgecolor':'white'},
        ):
    """
    This will create a basic *ternary* plot (or *quaternary*, *quinternary*, etc.)
    Converts ternary data to normalized for for plotting on the returned axes.
    
    :type data: :samp:`(N,3)` shaped :obj:`numpy.array`
    :param data: Data to be plotted. A :samp:`(N,3)` shaped :obj:`numpy.array`,
       where :samp:`N` is the number of data points.
    :type scaling: bool
    :param scaling: Scale data for ternary plot (i.e. a + b + c = 1)
    :type startAngle: float
    :param startAngle: Direction of first vertex.
    :type rotateLabels: bool
    :param rotateLabels: Orient labels perpendicular to vertices.
    :type labels: sequence of :obj:`str`
    :param labels: Labels for vertices.
    :type sides: int
    :param sides: Can accommodate more than 3 dimensions if desired.
    :type labelOffset: float
    :param labelOffset: Offset for label from vertex (percent of distance from origin).
    :type edgeArgs: dict
    :param edgeArgs: Any :obj:`matplotlib` keyword args for plots.
    :type figArgs: dict
    :param figArgs: Any :obj:`matplotlib` keyword args for figures.
    :rtype: (:obj:`numpy.array`, :obj:`TernaryAxes`)
    :return: A :samp:`(N,3)` shaped :obj:`numpy.array`, coordinates are
       transformed to *triangular* axes.

    For example::
    
       from matplotlib.pylab import *
       from mango.application.plot import ternaryPlot
       
       k = 0.5
       s = 1000
    
       data = vstack((
           array([k,0,0]) + rand(s,3), 
           array([0,k,0]) + rand(s,3), 
           array([0,0,k]) + rand(s,3)
       ))
       color = array([[1,0,0]]*s + [[0,1,0]]*s + [[0,0,1]]*s)
    
       newdata, ternAx = ternaryPlot(data)
       
       ax, fig = ternAx.createAxes()
       ax.scatter(
           newdata[:,0],
           newdata[:,1],
           s=2,
           alpha=0.5,
           color=color
       )
       show()
       
       
    """
    basis = sp.array(
                    [
                        [
                            sp.cos(2*s*sp.pi/sides + startAngle*sp.pi/180),
                            sp.sin(2*s*sp.pi/sides + startAngle*sp.pi/180)
                        ] 
                        for s in range(sides)
                    ]
                )

    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data for you.
        newdata = sp.dot((data.T / data.sum(-1)).T,basis)
    else:
        # Assumes data already sums to 1.
        newdata = sp.dot(data,basis)

    ternaryAxes =\
        TernaryAxes(
            basis        = basis,
            labels       = labels,
            sides        = sides,
            labelOffset  = labelOffset,
            rotateLabels = rotateLabels,
            edgeArgs     = edgeArgs,
            figArgs      = figArgs,
        )

    return newdata, ternaryAxes

__all__ = [s for s in dir() if not s.startswith('_')]

