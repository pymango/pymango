#!/usr/bin/env python
import logging
import mango.unittest
import scipy as sp
import numpy as np
import numpy.random
import mango.mpi as mpi
import mango
import mango.data
import mango.math
import mango.io
import os
import os.path

logger, rootLogger = mpi.getLoggers(__name__)

class FillTest(mango.unittest.TestCase):
    def setUp(self):
        dSz = 64
        mpidims = (1,1,1)
        if (mango.mpi.world != None):
            mpidims = mango.mpi.getCartShape(dimension=3)
        self.shape = (2*dSz*mpidims[0], dSz*1.25*mpidims[1], dSz*mpidims[2])
    
    def testCircularCylinderFill(self):
        dir = self.createTmpDir("testCircularCylinderFill")
        #dir = "."

        dds = mango.zeros(shape=self.shape, dtype="uint16")
        c = dds.shape*0.5
        r = np.min(dds.shape-3)*0.5
        axislen = dds.shape[0]*2
        
        mango.data.fill_circular_cylinder(dds, c, r, axislen, fill=1)
        mango.io.writeDds(os.path.join(dir, "tomoCyl0.nc"), dds)

        dds1 = mango.zeros(shape=self.shape, dtype="uint16", origin=(100, 55, 13))
        c = dds.shape*0.5
        r = np.min(dds.shape-3)*0.5
        axislen = dds.shape[0]*2
        
        mango.data.fill_circular_cylinder(dds1, c, r, axislen, fill=1)
        mango.io.writeDds(os.path.join(dir, "tomoCyl1.nc"), dds1)
        self.assertTrue(sp.all(dds.subd.asarray() == dds1.subd.asarray()))
        del dds1

        dds2 = mango.zeros(shape=self.shape, mtype="tomo", origin=(100, 55, 13))
        dds2.md.setVoxelSize((0.1,0.1,0.1), "cm")
        c = dds.shape*0.5
        r = np.min(dds.shape-3)*0.5
        axislen = dds.shape[0]*2
        
        mango.data.fill_circular_cylinder(dds2, c, r, axislen, fill=1, unit="mm")
        mango.io.writeDds(os.path.join(dir, "tomoCyl2.nc"), dds2)
        self.assertTrue(sp.all(dds.subd.asarray() == dds2.subd.asarray()))
        del dds2

        dds3 = mango.zeros(shape=self.shape, mtype="tomo", origin=(100, 55, 13))
        dds3.md.setVoxelSize((2.5,2.5,2.5), "um")
        c = dds.shape*0.5*dds3.md.getVoxelSize("mm")
        r = np.min(dds.shape-3)*0.5*dds3.md.getVoxelSize("mm")[0]
        axislen = dds.shape[0]*2*dds3.md.getVoxelSize("mm")[0]
        
        mango.data.fill_circular_cylinder(dds3, c, r, axislen, fill=1, unit="mm")
        mango.io.writeDds(os.path.join(dir, "tomoCyl3.nc"), dds3)
        self.assertTrue(sp.all(dds.subd.asarray() == dds3.subd.asarray()))
        del dds3

        dds4 = mango.zeros(shape=self.shape, mtype="tomo", origin=(100, 55, 13))
        dds4.md.setVoxelSize((2.5,2.5,2.5), "um")
        c = (dds4.origin + dds.shape*0.5)*dds4.md.getVoxelSize("mm")
        r = np.min(dds.shape-3)*0.5*dds4.md.getVoxelSize("mm")[0]
        axislen = dds.shape[0]*2*dds4.md.getVoxelSize("mm")[0]
        
        mango.data.fill_circular_cylinder(dds4, c, r, axislen, fill=1, unit="mm", coordsys="abs")
        mango.io.writeDds(os.path.join(dir, "tomoCyl4.nc"), dds4)
        self.assertTrue(sp.all(dds.subd.asarray() == dds4.subd.asarray()))
        del dds4

    def testCircularCylinderFillRotated(self):
        dir = self.createTmpDir("testCircularCylinderFillRotated")
        #dir = "."
        
        dds = mango.zeros(shape=self.shape, mtype="segmented")
        c = dds.shape*0.5
        r = np.min(dds.shape-3)*0.5
        axislen = dds.shape[0]*2
        
        rm = np.dot(mango.math.rotation_matrix(-10, 1), mango.math.rotation_matrix(20, 2))
        mango.data.fill_circular_cylinder(dds, c, r, axislen, fill=1, rotation=rm)
        mango.io.writeDds(os.path.join(dir, "segmentedCyl0_rotmtx.nc"), dds)

        (axis, angle) = mango.math.axis_angle_from_rotation_matrix(rm)
        rm = mango.math.axis_angle_to_rotation_matrix(axis, angle)
        
        dds = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_circular_cylinder(dds, c, r, axislen, fill=1, rotation=rm)
        mango.io.writeDds(os.path.join(dir, "segmentedCyl0_rotmtxaa.nc"), dds)
        
        dds1 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_circular_cylinder(dds1, c, r, axislen, fill=1, rotation=(axis, angle))
        mango.io.writeDds(os.path.join(dir, "segmentedCyl1_axang.nc"), dds1)
        self.assertTrue(sp.all(dds.subd.asarray() == dds1.subd.asarray()))
        del dds1

        dds1 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_circular_cylinder(dds1, c, r, axislen, fill=1, rotation=(axis.tolist(), angle))
        mango.io.writeDds(os.path.join(dir, "segmentedCyl2_axlistang.nc"), dds1)
        self.assertTrue(sp.all(dds.subd.asarray() == dds1.subd.asarray()))
        del dds1
        
        dds2 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_circular_cylinder(dds2, c, r, axislen, fill=1, rotation=(angle, axis.tolist()))
        mango.io.writeDds(os.path.join(dir, "segmentedCyl3_angaxlist.nc"), dds2)
        self.assertTrue(sp.all(dds.subd.asarray() == dds2.subd.asarray()))
        del dds2

        dds3 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_circular_cylinder(dds3, c, r, axislen, fill=1, rotation=(angle, axis))
        mango.io.writeDds(os.path.join(dir, "segmentedCyl4_angax.nc"), dds3)
        self.assertTrue(sp.all(dds.subd.asarray() == dds3.subd.asarray()))
        del dds3

        dds4 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_circular_cylinder(dds4, c, r, axislen, fill=1, rotation=angle*axis)
        mango.io.writeDds(os.path.join(dir, "segmentedCyl5_axis.nc"), dds4)
        self.assertTrue(sp.all(dds.subd.asarray() == dds4.subd.asarray()))
        del dds4

        dds5 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_circular_cylinder(dds5, c, r, axislen, fill=1, rotation=(angle*axis).tolist())
        mango.io.writeDds(os.path.join(dir, "segmentedCyl4_axislist.nc"), dds5)
        self.assertTrue(sp.all(dds.subd.asarray() == dds5.subd.asarray()))
        del dds5

    def testCircularAnnularCylinderFillRotated(self):
        dir = self.createTmpDir("testCircularAnnularCylinderFillRotated")
        #dir = "."

        
        dds = mango.zeros(shape=self.shape, mtype="segmented")
        c = dds.shape*0.5
        r = np.min(dds.shape-3)*0.35
        w = np.min(dds.shape-3)*0.1
        axislen = dds.shape[0]*2.
        
        rm = np.dot(mango.math.rotation_matrix(-10, 1), mango.math.rotation_matrix(20, 2))
        mango.data.fill_annular_circular_cylinder(dds, c, r, w, axislen, fill=1, rotation=rm)
        mango.io.writeDds(os.path.join(dir, "segmentedAnnularCyl0_rotmtx.nc"), dds)

        (axis, angle) = mango.math.axis_angle_from_rotation_matrix(rm)
        rm = mango.math.axis_angle_to_rotation_matrix(axis, angle)
        
        dds = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_annular_circular_cylinder(dds, c, r, w, axislen, fill=1, rotation=rm)
        mango.io.writeDds(os.path.join(dir, "segmentedAnnularCyl0_rotmtxaa.nc"), dds)
        
        dds1 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_annular_circular_cylinder(dds1, c, r, w, axislen, fill=1, rotation=(axis, angle))
        mango.io.writeDds(os.path.join(dir, "segmentedAnnularCyl1_axang.nc"), dds1)
        self.assertTrue(sp.all(dds.subd.asarray() == dds1.subd.asarray()))
        del dds1

        dds1 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_annular_circular_cylinder(dds1, c, r, w, axislen, fill=1, rotation=(axis.tolist(), angle))
        mango.io.writeDds(os.path.join(dir, "segmentedAnnularCyl2_axlistang.nc"), dds1)
        self.assertTrue(sp.all(dds.subd.asarray() == dds1.subd.asarray()))
        del dds1
        
        dds2 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_annular_circular_cylinder(dds2, c, r, w, axislen, fill=1, rotation=(angle, axis.tolist()))
        mango.io.writeDds(os.path.join(dir, "segmentedAnnularCyl3_angaxlist.nc"), dds2)
        self.assertTrue(sp.all(dds.subd.asarray() == dds2.subd.asarray()))
        del dds2

        dds3 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_annular_circular_cylinder(dds3, c, r, w, axislen, fill=1, rotation=(angle, axis))
        mango.io.writeDds(os.path.join(dir, "segmentedAnnularCyl4_angax.nc"), dds3)
        self.assertTrue(sp.all(dds.subd.asarray() == dds3.subd.asarray()))
        del dds3

        dds4 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_annular_circular_cylinder(dds4, c, r, w, axislen, fill=1, rotation=angle*axis)
        mango.io.writeDds(os.path.join(dir, "segmentedAnnularCyl5_axis.nc"), dds4)
        self.assertTrue(sp.all(dds.subd.asarray() == dds4.subd.asarray()))
        del dds4

        dds5 = mango.zeros(shape=self.shape, mtype="segmented")
        mango.data.fill_annular_circular_cylinder(dds5, c, r, w, axislen, fill=1, rotation=(angle*axis).tolist())
        mango.io.writeDds(os.path.join(dir, "segmentedAnnularCyl4_axislist.nc"), dds5)
        self.assertTrue(sp.all(dds.subd.asarray() == dds5.subd.asarray()))
        del dds5

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.data", "mango.dataTest"],
        logLevel=logging.INFO
    )
    mango.unittest.main()
