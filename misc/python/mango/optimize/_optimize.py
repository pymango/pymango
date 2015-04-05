
try:
    from io import BytesIO
except ImportError as e:
    import StringIO as BytesIO
    
try:
    import cPickle as pickle
except ImportError as e:
    import pickle

from mango import mpi
haveMpi4py = mpi.haveMpi4py 
import scipy as sp
import scipy.optimize

logger, rootLogger = mpi.getLoggers(__name__)

class DistributedMetricEvaluator(object):
    """
    Wrapper class for functions which are evaluated by combining (MPI-reducing) a
    result from each MPI process. See also the :func:`distributed_minimize` function.
    
    Example::
    
       import mango.mpi
       import mango.optimize
       import scipy as sp
       import scipy.optimize
       
       def my_func(x):
          return (mango.mpi.world.Get_rank()+1) * (x * x + x + 1)
    
       # dfunc sums/reduces (rank+1)*(x*x + x + 1) from all processes to the dfunc.root process
       dfunc = mango.optimize.DistributedMetricEvaluator(my_func) 
       
       if (dfunc.comm.Get_rank() == dfunc.root):
           x0 = 8
           res = scipy.optimize.minimize(dfunc, x0, method="Powell")
           dfunc.rootTerminate()
           print("res.x = %s, res.fun = %s" % (res.x, res.fun))
       else:
           dfunc.waitForEvaluate()
       
    
    """
    #: Instruction to evaluate local MPI function value, see :meth:`waitForEvaluate`
    EVALUATE  = 0
    #: Instruction to terminate the wait-loop in :meth:`waitForEvaluate`
    TERMINATE = 1
    def __init__(self, funcObj, comm=mpi.world, root=0):
        self.funcObj = funcObj
        self.comm = comm
        self.root = root
        self.x = None
    
    def __getattr__(self, name):
        """
        Python magic to forward attribute access to self.funcObj
        """
        return getattr(self.funcObj, name)

    def evaluate(self, x):
        """
        Evaluate the function at :samp:`x` on the local MPI process.
        """
        self.x = x
        return self.funcObj(x)

    def calcReduction(self, localVal):
        """
        Converts the *local* result :samp:`{localVal}` returned from
        the :meth:`evaluate` method to a MPI-reduced result.
        
        :type localVal: reducable :obj:`object`
        :param localVal: Object returned from the :meth:`evaluate` method.
        """
        if (self.comm != None):
            metricVal = self.comm.reduce(localVal, root=self.root, op=mpi.SUM)
        return metricVal
    
    def rootEvaluate(self, x):
        """
        Broadcasts :samp:`x`
        to all processes and then does local evaluation (:samp:`self.evaluate(x)`).
        MPI reduces (:samp:`op=mpi.SUM`) the results from all MPI processes
        and returns the reduced result.
        Should only be called by the :samp:`self.root`-rank process.
        
        :type x: broadcastable :obj:`object`
        :param x: Function parameter.
        
        :return: Reduction of the :meth:`evaluate` values from MPI processes.

        """
        if (self.comm != None):
            (instr, x) = self.comm.bcast((self.EVALUATE, x), root=self.root)
        metricVal = self.evaluate(x)
        metricVal = self.calcReduction(metricVal)
        
        return metricVal
    
    def waitForEvaluate(self):
        """
        Loops
        waiting for an :samp:`self.EVALUATE` broadcast of the :samp:`x`
        parameter from the :samp:`self.root`-rank process. Exit's loop when
        a :samp:`(self.TERMINATE,x)` broadcast is received.
        Should only be called by the non-:samp:`self.root`-rank MPI processes.
        
        """
        (instr, x) = self.comm.bcast(root=self.root)
        while (instr != self.TERMINATE):
            metricVal = self.evaluate(x)
            metricVal = self.calcReduction(metricVal)
            (instr, x) = self.comm.bcast(root=self.root)
        
    def rootTerminate(self):
        """
        Issues
        a :samp:`(self.TERMINATE,x)` broadcast so other processes exit
        the :meth:`waitForEvaluate` loop.
        Should only be called from the :samp:`self.root`-rank process. 
        """
        if (self.comm != None):
            (instr, x) = self.comm.bcast((self.TERMINATE, None))    
    
    def __call__(self, x):
        """
        Method which makes this object behave like a function,
        should only be called from the :samp:`self.root`-rank process.
        """
        if (self.comm != None):
            if (self.root == self.comm.Get_rank()):
                metricVal = self.rootEvaluate(x)
            else:
                raise RuntimeError("__call__ method called from non-root process, rank=%s, self.root=%s" % (self.comm.Get_rank(), self.root))
        else:
            metricVal = self.rootEvaluate(x)

        return metricVal


def distributed_minimize(fun, *args, **kwargs):
    """
    Wrapper for the :func:`scipy.optimize.minimize` function for distributed function evaluation
    on multiple MPI processors. The :samp:`{fun}` argument should be an instance
    of :obj:`DistributedMetricEvaluator`. This function should be called from all MPI processes.
    
    :type fun: :obj:`DistributedMetricEvaluator`
    :param fun: Function which is to be minimized.
    :type bcastres: :obj:`bool`
    :param bcastres: If :samp:`True` (default is :samp:`True`) the result
       object (:obj:`scipy.optimize.OptimizeResult`) returned
       by :func:`scipy.optimize.minimize` is broadcast from
       the :samp:`{fun}.root` process to all other :samp:`{fun}.comm` MPI processes.
       If :samp:`False` only the :samp:`{fun}.root` process returns the result and other
       processes return the :samp:`None` value.
    
    Example::
    
       import mango.mpi
       import scipy as sp
       import mango.optimize
       
       def my_func(x):
          return (mango.mpi.world.Get_rank()+1) * (x * x + x + 1)
    
       # dfunc sums/reduces (rank+1)*(x*x + x + 1) from all processes to the dfunc.root process
       dfunc = mango.optimize.DistributedMetricEvaluator(my_func) 
       x0 = 8
       res = mango.optimize.minimize(dfunc, x0, method="Powell")
       print("res.x = %s, res.fun = %s" % (res.x, res.fun))

    """
    res = None
    if ((fun.comm == None) or ((fun.comm != None) and (fun.root == fun.comm.Get_rank()))):
        res = sp.optimize.minimize(fun, *args, **kwargs)
        fun.rootTerminate()
    else:
        fun.waitForEvaluate()
    
    if ((fun.comm != None) and (("bcastres" not in kwargs.keys()) or kwargs["bcastres"])):
        res = fun.comm.bcast(res, fun.root)
    
    return res


class OptimizeResult:
    """
    Optimization result object returned by SingleStartOptimizer.optimize method.
    """
    def __init__(self):
        self.optim          = None
        self.start          = None
        self.cost           = None
        self.numIterations  = None
        self.numFuncEvals   = None
        self.startIndex     = None

    def getStartIndex(self):
        return self.startIndex
    
    def getNumIterations(self):
        return self.numIterations

    def getNumFuncEvals(self):
        return self.numFuncEvals
    
    def getStart(self):
        return self.start
    
    def getOptim(self):
        return self.optim
    
    def getCost(self):
        return self.cost
        
    def __str__(self):
        optimFmtStr = "%16.8f"
        startFmtStr = "%16.8f"

        if (hasattr(self.optim, "__len__")):
            optimStr = "("
            optimStr += optimFmtStr % self.optim[0]
            for comp in self.optim[1:]:
                optimStr += optimFmtStr % comp
            optimStr += ")"
        else:
            optimStr = optimFmtStr % self.optim
            
        if (hasattr(self.start, "__len__")):
            startStr = "("
            startStr += startFmtStr % self.start[0]
            for comp in self.start[1:]:
                startStr += startFmtStr % comp
            startStr += ")"
        else:
            startStr = startFmtStr % self.start

        return \
            (
                "cost=%16.8f,optim=%s,start=%s,startIdx=%6d,numFuncEval=%8d,numIterations=%6d"
                %
                (
                    self.cost,
                    optimStr,
                    startStr,
                    self.startIndex,
                    self.numFuncEvals,
                    self.numIterations
                )
            )



class SingleStartOptimizer:
    """
    Base class for single start optimizers.
    """
    def __init__(self):
        pass
    
    def cmpOptimizeResult(self, r0, r1):
        """
           Compare two OptimizeResult objects, used for sorting
           list of OptimizeResult objects.
        """
        d = r0.cost - r1.cost
        if (d < 0):
            return -1
        elif (d > 0):
            return 1
        return 0

    def optimize(self, startPrms):
        raise NotImplementedError("Derived class does not implement optimize method")

    def minimize(self, startPrms):
        return self.optimize(startPrms)

class MultiStartOptimizer:
    """
    Runs an optimization, using multiple starting values, in an attempt to find
    a global minimum. Can take advantage of mpi4py as long as the
    start-parameter-objects can be pickled (and subsequently sent to remote MPI processes).
    
    The self.rootRank master MPI process uses asynchronous communication to hand out
    individual (single-start) optimization jobs to remote worker MPI processes. 
    """
    def __init__(self, singleStartOptimizer, rootRank=0, mpiComm=None):
        """
        Initialise.
        
        :type singleStartOptimizer: SingleStartOptimizer
        :param singleStartOptimizer: An object which can perform an optimization of a single
           start-parameter object.
        :type rootRank: int
        :param rootRank: Rank of the process which controls the handing out of
           individual (single-start) optimization jobs
        """
        self.ssOptimizer        = singleStartOptimizer
        self.prmIterator        = None
        self.rootRank           = rootRank
        self.terminatePrmArray  = None
        self.byteArrayRecvSize  = 2**19
        self.resultList         = None
        if (mpi.haveMpi4py and (mpiComm == None)):
            mpiComm = mpi.world
        self.mpiComm = mpiComm
        self.numMpiProcsForAsync = 4
        
        self.START_PARAM_TAG = 0
        self.OPTIM_RESULT_TAG = 1

    def setNumMpiProcessesForAsyncProcessing(self, numMpiProcs):
        """
        Set the number of MPI processes which triggers the use of
        asynchronous master/slave processing of the multi-start optimizations.
        """
        self.numMpiProcsForAsync = numMpiProcs

    def broadcast(self, prmIterator):
        if ((self.mpiComm == None) or (self.mpiComm.Get_rank() == self.rootRank)):
            self.prmIterator = prmIterator
        else:
            self.prmIterator = None

        if (self.mpiComm != None):
            self.prmIterator = self.mpiComm.bcast(self.prmIterator, root=self.rootRank)

    def broadcastResultList(self):
        if (self.mpiComm != None):
            self.resultList = self.mpiComm.bcast(self.resultList, root=self.rootRank)

    def pickleToByteArray(self, obj):
        bytesIO = BytesIO()
        pickle.dump(obj, bytesIO)
        return bytearray(bytesIO.getvalue())

    def unpickleFromByteArray(self, ba):
        bytesIO = BytesIO(ba)
        return pickle.load(bytesIO)

    def doAsyncMpiMasterSends(self):
        if (self.mpiComm.Get_rank() == self.rootRank):
            optimReqDict = dict()
            SREQ = 0
            SBUF = 1
            RREQ = 2
            RBUF = 3
            worldSz = self.mpiComm.Get_size()
            workerSz = worldSz-1

            for rank in range(1, worldSz):
                optimReqDict[rank] = [None, bytearray(), None, bytearray()]
            
            prmIdx = 0
            startRankIdx = 0
            resultList = []
            completeCount = 0
            numOptimsPerMpiProc = len(self.prmIterator)
            for startPrm in self.prmIterator:
                foundWorker = False
                while (not foundWorker):
                    for rankIdx in range(startRankIdx, workerSz):
                        rank = rankIdx + 1
                        optimReq = optimReqDict[rank]
                        if ((optimReq[RREQ] == None) or (optimReq[RREQ].Test())):
                            if ((optimReq[RREQ] != None)):
                                result = self.unpickleFromByteArray(optimReq[RBUF])
                                if (result.optim != None):
                                    resultList.append(result)
                                completeCount += 1
                                logger.info("Completed %5d of %5d optimizations." % (completeCount, numOptimsPerMpiProc))
                                if (len(resultList) > 0):
                                    resultList.sort(self.ssOptimizer.cmpOptimizeResult)
                                    logger.info("Best result so far:\n%s" % (str(resultList[0],)))
                            foundWorker = True
                            optimReq[SBUF] = self.pickleToByteArray([prmIdx, startPrm])
                            optimReq[RBUF] = bytearray(self.byteArrayRecvSize)
                            optimReq[SREQ] = self.mpiComm.Isend(optimReq[SBUF], dest = rank, tag = self.START_PARAM_TAG)
                            optimReq[RREQ] = self.mpiComm.Irecv(optimReq[RBUF], source = rank, tag = self.OPTIM_RESULT_TAG)
                            startRankIdx = rankIdx+1
                            break
                    startRankIdx = 0
                prmIdx += 1
                
            waitingForOptimResult = True
            while (waitingForOptimResult):
                waitingForOptimResult = False
                for rankIdx in range(0, workerSz):
                    rank = rankIdx + 1
                    optimReq = optimReqDict[rank]
                    if (optimReq[SREQ] == None):
                        optimReq[SBUF] = self.pickleToByteArray([None, None])
                        optimReq[SREQ] = self.mpiComm.Isend(optimReq[SBUF], dest = rank, tag = self.START_PARAM_TAG)

                    if (optimReq[RREQ] != None):
                        waitingForOptimResult = True
                        if (optimReq[RREQ].Test()):
                            result = self.unpickleFromByteArray(optimReq[RBUF])
                            if (result.optim != None):
                                resultList.append(result)
                            completeCount += 1
                            logger.info("Completed %5d of %5d optimizations." % (completeCount, numOptimsPerMpiProc))
                            if (len(resultList) > 0):
                                resultList.sort(self.ssOptimizer.cmpOptimizeResult)
                                logger.info("Best result so far:\n%s" % (str(resultList[0],)))

                            optimReq[RREQ] = None
                            optimReq[RBUF] = None
                            optimReq[SBUF] = self.pickleToByteArray([None, None])
                            optimReq[SREQ] = self.mpiComm.Isend(optimReq[SBUF], dest = rank, tag = self.START_PARAM_TAG)
            resultList.sort(self.ssOptimizer.cmpOptimizeResult)
            mpi.Request.Waitall([optimReqDict[rankIdx][SREQ] for rankIdx in range(1, self.mpiComm.Get_size())])
            self.resultList = resultList

    def doAsyncMpiWorkerRecvs(self):
        if (self.mpiComm.Get_rank() != self.rootRank):
            recvdTerminate = False
            while (not recvdTerminate):
                rBa = bytearray(self.byteArrayRecvSize)
                rReq = self.mpiComm.Irecv(rBa, source = self.rootRank, tag = self.START_PARAM_TAG)
                rReq.Wait()
                [prmIdx, startPrm] = self.unpickleFromByteArray(rBa)
                if ((prmIdx != None) and (startPrm != None)):
                    result = self.ssOptimizer.minimize(startPrm)
                    result.startIndex = prmIdx
                    sReq = \
                        self.mpiComm.Isend(
                            self.pickleToByteArray(result),
                            dest = self.rootRank,
                            tag = self.OPTIM_RESULT_TAG
                        )
                    sReq.Wait()
                else:
                    recvdTerminate = True
                
    def doAsyncMpiMultiStart(self):
        self.doAsyncMpiMasterSends()
        self.doAsyncMpiWorkerRecvs()
    
    def doEvenDivisionMultiStart(self):
        prmIdx = 0
        resultList = []
        if (self.mpiComm != None):
            worldSz = self.mpiComm.Get_size()
            myRank = self.mpiComm.Get_rank()
        else:
            worldSz = mpi.size
            myRank = mpi.rank

        numOptimsPerMpiProc =\
            sp.sum(
                sp.ones((len(self.prmIterator),), dtype="int32")[
                    sp.where(
                        (sp.array(range(0, len(self.prmIterator)), dtype="int32") % worldSz)
                        ==
                        myRank
                    )
                ]
            )
        logger.info("")
        completeCount = 0
        for startPrm in self.prmIterator:
            if ((prmIdx % worldSz) == myRank):
                logger.info("Optimizing %5d of %5d optimizations..." % (completeCount+1, numOptimsPerMpiProc))                
                result = self.ssOptimizer.optimize(startPrm)
                result.startIndex = prmIdx
                if (result.optim != None):
                    resultList.append(result)
                completeCount += 1
                logger.info("Completed  %5d of %5d optimizations." % (completeCount, numOptimsPerMpiProc))
            prmIdx += 1
        if (self.mpiComm != None):
            resultList = self.mpiComm.reduce(resultList, op=mpi.SUM, root=self.rootRank)
        if (myRank == self.rootRank):
            resultList.sort(cmp=self.ssOptimizer.cmpOptimizeResult)
            self.resultList = resultList
        else:
            self.resultList = None

    def optimize(self, startPrmIterator):
        self.broadcast(startPrmIterator)

        if ((self.mpiComm != None) and (self.mpiComm.Get_size() >= self.numMpiProcsForAsync)):
            self.doAsyncMpiMultiStart()
        else:
            self.doEvenDivisionMultiStart()
        
        if (self.mpiComm != None):
            self.mpiComm.Barrier()
        self.broadcastResultList()

        return self.resultList

    def minimize(self, startPrmIterator):
        return self.optimize(startPrmIterator)
