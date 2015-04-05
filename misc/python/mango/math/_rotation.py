
import scipy as sp
import numpy as np
import mango.mpi as mpi
import math

logger, rootLogger = mpi.getLoggers(__name__)

def rotation_matrix(angle, axis, dim=3, dtype="float64"):
    """
    Returns rotation matrix for specified degree angle and
    coordinate axis of rotation.
    
    :type angle: :obj:`float`
    :param angle: Angle of rotation in degrees.
    :type axis: :obj:`int`
    :param axis: Index of the axis of rotation (for :samp:`{dim}=3`, :samp:`{axis}=0`
       is the z-axis, :samp:`{axis}=1` is the y-axis and:samp:`{axis}=2`
       is the x-axis.
    :type dim: :obj:`int`
    :param dim: Rotation spatial dimension.
    :rtype: :obj:`numpy.array`
    :return: A :samp:`(dim, dim)` shaped rotation matrix.
    """
    
    I = sp.eye(dim, dim, dtype=dtype)
    u = sp.zeros((dim,1), dtype=dtype)
    v = sp.zeros((dim,1), dtype=dtype)
    u[(axis+dim-2) % dim] = 1
    v[(axis+dim-1) % dim] = 1
    
#     rootLogger.debug("u          = %s" % str(u))
#     rootLogger.debug("u.T        = %s" % str(u.T))
#     rootLogger.debug("u.dot(u.T) = %s" % str(u.dot(u.T)))
# 
#     rootLogger.debug("v   = %s" % str(v))
#     rootLogger.debug("v.T = %s" % str(v.T))
    
    theta = sp.pi/180. * angle
    R = I + sp.sin(theta)*(v.dot(u.T) - u.dot(v.T)) + (sp.cos(theta) - 1)*(u.dot(u.T) + v.dot(v.T))
    
    rootLogger.debug("R = %s" % str(R))
    return R

def axis_angle_to_rotation_matrix(direction, angle):
    """
    Convert 3D axis and angle of rotation to 3x3 rotation matrix.
    
    :type direction: 3 sequence of :obj:`float`
    :param direction: Axis of rotation.
    :type angle: :obj:`float`
    :param angle: Radian angle of rotation about axis.
    :rtype: :obj:`numpy.array`
    :return: 3x3 rotation matrix.
    """
    d = np.array(direction, dtype=direction.dtype)
    eye = np.eye(3, 3, dtype=d.dtype)
    mtx = eye
    dNorm = np.linalg.norm(d)
    if ((angle != 0) and (dNorm > 0)):
        d /= dNorm
    
        ddt = np.outer(d, d)
        skew = np.array([[    0,  d[2],  -d[1]],
                         [-d[2],     0,   d[0]],
                         [ d[1], -d[0],     0]], dtype=d.dtype).T
    
        mtx = ddt + np.cos(angle) * (eye - ddt) + np.sin(angle) * skew
    return mtx

def axis_angle_from_rotation_matrix(rm):
    """
    Converts 3x3 rotation matrix to axis and angle representation.
    
    :type rm: 3x3 :obj:`float` matrix
    :param rm: Rotation matrix.
    :rtype: :obj:`tuple`
    :return: :samp:`(axis, radian_angle)` pair (angle in radians).
    """
    eps = (16*sp.finfo(rm.dtype).eps)
    aa = sp.array((0,0,1), dtype=rm.dtype) 
    theta = aa[0];
    c = (sp.trace(rm) - 1)/2;
    if (c > 1):
        c = 1;
    if (c < -1):
        c = -1;
    if (math.fabs(math.fabs(c)-1) >= eps):
        theta = math.acos(c);
        s = math.sqrt(1-c*c);
        inv2s = 1/(2*s);
        aa[0] = inv2s*(rm[2,1] - rm[1,2]);
        aa[1] = inv2s*(rm[0,2] - rm[2,0]);
        aa[2] = inv2s*(rm[1,0] - rm[0,1]);
    elif (c >= 0):
        theta = 0;
    else:
        rmI = (rm + sp.eye(3,3,dtype=rm.dtype));
        theta = np.pi;
        for i in range(0,3):
            n2 = np.linalg.norm(rmI[:,i]);
            if (n2 > 0):
                aa = col(rmI, i);
                break;

    return aa, theta

def rotation_matrix_from_cross_prod(a,b):
    """
    Returns the rotation matrix which rotates the
    vector :samp:`a` onto the the vector :samp:`b`.
    
    :type a: 3 sequence of :obj:`float`
    :param a: Vector to be rotated on to :samp:`{b}`.
    :type b: 3 sequence of :obj:`float`
    :param b: Vector.
    :rtype: :obj:`numpy.array`
    :return: 3D rotation matrix.
    """

    crs = np.cross(a,b)
    dotProd = np.dot(a,b)
    crsNorm = sp.linalg.norm(crs)
    eps = sp.sqrt(sp.finfo(a.dtype).eps)
    r = sp.eye(a.size, a.size, dtype=a.dtype)
    if (crsNorm > eps):
        theta = sp.arctan2(crsNorm, dotProd)
        r = axis_angle_to_rotation_matrix(crs, theta)
    elif (dotProd < 0):
        r = -r
    
    return r

def axis_angle_from_cross_prod(a,b):
    """
    Returns the :samp:`(axis, radian_angle)` rotation which rotates the
    vector :samp:`a` onto the the vector :samp:`b`.
    
    :type a: 3 sequence of :obj:`float`
    :param a: Vector to be rotated on to :samp:`{b}`.
    :type b: 3 sequence of :obj:`float`
    :param b: Vector.
    :rtype: :obj:`tuple`
    :return: samp:`(axis, radian_angle)` pair (angle in radians).
    """
    crs = np.cross(a,b)
    dotProd = np.dot(a,b)
    crsNorm = sp.linalg.norm(crs)
    theta = sp.arctan2(crsNorm, dotProd)
    
    return crs, theta
