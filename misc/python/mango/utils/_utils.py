import mango.mpi

logger,rootLogger = mango.mpi.getLoggers(__name__)

class ModuleObjectFactory:
    """
    Factory for creating objects of classes from a specified module.
    Useful for multiple instantiations of C++ template classes.  
    """
    def __init__(self, module):
        """
        Initialise specifying module from which class instances (objects)
        are to be constructed.
        """
        self.module = module
    
    def getClass(self, classNamePrefix, dtype="float64"):
        dtypeStr = str(dtype).lower()
        cls = None
        clsName = classNamePrefix + dtypeStr
        if (hasattr(self.module, clsName)):
            cls = getattr(self.module, clsName)
        else:
            raise ValueError("No class named %s in module %s." % (clsName, self.module.__name__))
        
        return cls
    
    def create(self, classNamePrefix, dtype="float64", **kwargs):
        cls = self.getClass(classNamePrefix, dtype)
        obj = cls(**kwargs)
        cls.dtype = dtype

        return obj

def checkForException(potentialException, extraErrorMessage=None, loggerFunction=None):
    """
    Simply checks if the :samp:`importedModule` argument is derived from
    :obj:`Exception`. If it is an exception then a :samp:`raise potentialException`
    is executed.
    :type potentialException: object
    :param potentialException: Check if this object is an :obj:`Exception`.
    :type extraErrorMessage: str
    :param extraErrorMessage: If :samp:`potentialException` is an instance
       of :obj:`Exception`, use the :samp:`loggerFunction` to log the message.
    :type loggerFunction: callable
    :param loggerFunction: If :samp:`potentialException` is an instance
       of :obj:`Exception`, then :samp:`loggerFunction(extraErrorMessage)`
       is called before raising the `potentialException` exception.
    """
    if (isinstance(potentialException, Exception)):
        if (loggerFunction != None and extraErrorMessage != None):
            loggerFunction(extraErrorMessage)
        raise potentialException
