import mango.mpi as mpi
import mango.unittest
import mango.data
import mango.image
import mango.image.registration
import scipy as sp
from scipy import linalg
import logging
import os
import os.path

logger, rootLogger = mpi.getLoggers(__name__)

class TransformTest(mango.unittest.TestCase):

    def testAffineTransform(self):
        tmpDir = self.createTmpDir(os.path.join("TransformTest","testAffineTransform"))
        dds = mango.data.createCheckerDds(shape=(64,128,128), checkShape=(8,16,16), mtype="segmented", black=0, white=1)
        dds.md.setVoxelSize((1,1,1))
        dds.md.setVoxelSizeUnit("millimetre")
        mtrx = sp.zeros((3,3), dtype="float64")
        mtrx[0,0] = 1.0
        mtrx[1,1] = 0.5
        mtrx[2,2] = 0.25
        shift = (0, 0, 0)

        tDds = \
            mango.image.affine_transform(
                dds,
                mtrx,
                shift,
                interptype=mango.image.registration.InterpolationType.NEAREST_NEIGHBOUR
            )
        mango.io.writeDds(os.path.join(tmpDir,"segmentedCheck.nc"), dds)
        mango.io.writeDds(os.path.join(tmpDir,"segmentedCheckTransformed.nc"), tDds)
        
        if ((tDds.mpi.comm == None) or (tDds.mpi.size <= 1)):
            offset = sp.array(tDds.asarray().shape)*0.5
            spTrnsDds = mango.zeros_like(tDds)
            spTrnsDds.asarray()[...] = sp.ndimage.affine_transform(dds.asarray(), linalg.inv(mtrx), linalg.inv(mtrx).dot(mtrx.dot(offset.T)-offset.T), order=0, mode="constant", cval=dds.mtype.maskValue())
            mango.io.writeDds(os.path.join(tmpDir,"segmentedCheckTransformedScipy.nc"), spTrnsDds)

            self.assertTrue(sp.all(spTrnsDds.asarray() == tDds.asarray()))
            

    def testRotationTransform(self):
        tmpDir = self.createTmpDir(os.path.join("TransformTest","testRotationTransform"))
        dds = mango.data.createCheckerDds(shape=(64,128,128), checkShape=(8,32,16), mtype="segmented", black=0, white=1)
        dds = mango.copy(dds, origin=dds.origin-64, shape = dds.shape+128)
        dds.mtype = mango.mtype("segmented")
        dds.md.setVoxelSize((1,1,1))
        dds.md.setVoxelSizeUnit("millimetre")
        rootLogger.info("dds.origin = %s, dds.shape=%s" % (dds.origin,dds.shape))
        rootLogger.info("dds.subd.origin = %s, dds.subd.shape=%s" % (dds.subd.origin,dds.subd.shape))

        tDds = \
            mango.image.rotate(
                dds,
                30,
                interptype=mango.image.registration.InterpolationType.NEAREST_NEIGHBOUR
            )
        rootLogger.info("mango.count_non_masked(dds)       = %7d" % (mango.count_non_masked(dds),))
        rootLogger.info("mango.count_non_masked(tDds)      = %7d" % (mango.count_non_masked(tDds),))
        mango.io.writeDds(os.path.join(tmpDir,"segmentedCheck.nc"), dds)
        mango.io.writeDds(os.path.join(tmpDir,"segmentedCheckRotated.nc"), tDds)
        
        if ((dds.mpi.comm == None) or (dds.mpi.size <= 1)):
            offset = sp.array(dds.asarray().shape)*0.5
            rootLogger.info("offset = %s" % str(offset))
            mtrx = mango.image.rotation_matrix(30, 0)
            spTrnsDds = mango.zeros_like(dds)
            #spTrnsDds.asarray()[...] = sp.ndimage.rotate(dds.asarray(), -30, axes=(2,1), reshape=False, order=0, mode="constant", cval=dds.mtype.maskValue())
            spTrnsDds.asarray()[...] = sp.ndimage.affine_transform(dds.asarray(), linalg.inv(mtrx), linalg.inv(mtrx).dot(mtrx.dot(offset.T)-offset.T), order=0, mode="constant", cval=dds.mtype.maskValue())
            rootLogger.info("mango.count_non_masked(spTrnsDds) = %7d" % (mango.count_non_masked(spTrnsDds),))
            mango.io.writeDds(os.path.join(tmpDir,"segmentedCheckRotatedScipy.nc"), spTrnsDds)

            self.assertGreater(
                0.01*sp.product(spTrnsDds.shape),
                sp.sum(sp.where(
                    spTrnsDds.asarray() != tDds.asarray(),
                    1,
                    0
                ))
            )
            self.assertTrue(sp.all(spTrnsDds.asarray() == tDds.asarray()))

            

if __name__ == "__main__":
    mango.setLoggingVerbosityLevel("high")
    mpi.initialiseLoggers(
        [__name__, "mango.mpi", "mango.image", "mango.imageTest"],
        logLevel=logging.DEBUG
    )
    mango.unittest.main()
