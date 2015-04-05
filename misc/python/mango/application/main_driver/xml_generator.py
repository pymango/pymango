__doc__ = \
"""
=============================================================================================================================
Routines for Run-Time Generation of Main Driver Filter XML Descriptions ((:mod:`mango.application.main_driver.xml_generator`)  
=============================================================================================================================

.. currentmodule:: mango.application.main_driver.xml_generator

Functions for generating XML filter descriptions (:samp:`<install_prefix>/share/pymango.xml`). 
Simply inspects all main-driver registered filter classes and accumlates the
individual XML descriptions (using the :meth:`mango.application.main_driver.MainDriverFilter.getXmlFileName`
or :meth:`mango.application.main_driver.MainDriverFilter.getXmlString` methods).

.. seealso::

   Class :obj:`mango.application.main_driver.MainDriverFilter`
      Base class for main-driver filters.
     
Main Functions
==============

.. autosummary::
   :toctree: generated/

   generateXmlFromCommandLineArgs - The main routine called from :samp:`__main__`.   
   generatePyMangoFilterDescriptionXml - Main work-horse, generates the complete XML string for all registered main-driver filters.
   generatePerFilterClassXmlString - Returns the XML description for a specified sequence of main-driver filter classes.
   
    
Helper Functions
================

Most of the helper functions which return a :obj:`str` look for the relevant method attribute
on a filter class before calling the method, if the attribute isn't present, then :samp:`None`
is returned.

.. autosummary::
   :toctree: generated/
   
   getXmlString - Returns filter description XML string from the :meth:`mango.application.main_driver.MainDriverFilter.getXmlString` method.
   readXmlStringFromXmlFileName - Returns filter description XML string by reading file returned by :meth:`mango.application.main_driver.MainDriverFilter.getXmlFileName` method.
   getParmSectionName - Returns the parameter-section-name string by calling the :meth:`mango.application.main_driver.MainDriverFilter.parmSectionName` method.
   checkXmlFromFile - Reads XML filter description from file and validates the XML.
   checkXmlFromString - Validates an XML string.
   makeXmlDoc - Turns improper XML string into proper XML document string.
"""

import logging
import re

try:
    # Import the set class, python2 only, python3 has set as a built-in.
    import sets.Set as set
except:
    pass

import inspect
import mango
import mango.mpi as mpi
from mango.application.main_driver import MainDriverFilter
from mango.application.main_driver import getAllRegisteredMainDriverFilterClasses

# Import XML element-tree parser, use this to validate XML.
import xml.etree.ElementTree as ET
      
logger, rootLogger = mpi.getLoggers(__name__)

_xmlDocRe = re.compile("\\s*\\<\\?\\s*xml\\s*version\\s*=.*\\?\\>\\s+(.*)")
def makeXmlDoc(xmlStr):
    """
    Turns XML tag string into XML document which can be validated by :obj:`xml.etree.ElementTree`
    parser. Simply adds the :samp:`'<? xml version=\"1.0\"?>'` header string and
    global enclosing tags of :samp:`<mango>...</mango>`.
    :type xmlStr: :obj:`str`
    :param xmlStr: String of XML.
    :rtype: :obj:`str`
    :return: XML document string.
    """
    xmlDocStr = xmlStr
    if (re.match(_xmlDocRe, xmlDocStr) == None):
        xmlDocStr = ("<?xml version=\"1.0\"?>\n<mango>\n%s\n</mango>" % xmlStr)
    
    return xmlDocStr

def checkXmlFromString(xmlDocumentStr, parmSectName=None):
    """
    Validates specified XML string by creating a :obj:`xml.etree.ElementTree`
    by calling :samp:`xml.etree.ElementTree.fromstring({xmlDocumentStr})`.
    
    :type xmlDocumentStr: :obj:`str`
    :param xmlDocumentStr: XML string to validate.
    :type parmSectName: :obj:`str`
    :param parmSectName: If not :samp:`None`, checks that at least one
       of the *analysis*, *inplace*, *creation* or *generator* child nodes
       has a :samp:`name` attribute equal to the :samp:`{parmSectName}`.
    :rtype: :obj:`bool`
    :return: :samp:`True` if XML is valid. 
    """
    root = ET.fromstring(xmlDocumentStr)
    
    if (parmSectName is not None):
        filtChildRe = re.compile("analysis|inplace|creation|generator")
        nameAttribList = []
        for child in root:
            m = re.match(filtChildRe, child.tag)
            if (m != None):
                if ("name" in child.attrib.keys()):
                    nameAttribList.append(child.attrib["name"])
        
        if (parmSectName not in nameAttribList):
            rootLogger.warn(
                "Could not find matching class parmSectionName()=%s in XML attributes: name=\"%s\"."
                %
                (parmSectName, nameAttribList)
            )


    return True

def checkXmlFromFile(xmlFileName, parmSectName=None):
    """
    Reads entire file into string and calls :obj:`checkXmlFromString`
    to validate the XML.
    
    :type xmlFileName: :obj:`str`
    :param xmlFileName: XML string read from this file.
    :type parmSectName: :obj:`str`
    :param parmSectName: If not :samp:`None`, checks that at least one
       of the *analysis*, *inplace*, *creation* or *generator* child nodes
       has a :samp:`name` attribute equal to the :samp:`{parmSectName}`.
    :rtype: :obj:`bool`
    :return: :samp:`True` if XML is valid.
    """
    xmlDocStr = makeXmlDoc(open(xmlFileName, 'r').read())
    return checkXmlFromString(xmlDocStr, parmSectName)

def getParmSectionName(filterCls):
    """
    Returns :samp:`{filterCls}.parmSectionName()`.
    
    :return: :samp:`None` if :samp:`{filterCls}` does not have a :samp:`'parmSectionName'` attribute.
    """
    parmSectName = None
    if (hasattr(filterCls, "parmSectionName")):
        if (inspect.isroutine(getattr(filterCls, "parmSectionName"))):
            parmSectName = filterCls().parmSectionName()
        
    return parmSectName

def readXmlStringFromXmlFileName(filterCls):
    """
    Returns the string read from the file named :samp:`{filterCls}.getXmlFileName()`.
    
    :return: :samp:`None` if :samp:`{filterCls}` does not have a :samp:`'getXmlFileName'` attribute.
    """
    xmlFileName = None
    if (hasattr(filterCls, "getXmlFileName")):
        if (inspect.isroutine(getattr(filterCls, "getXmlFileName"))):
            xmlFileName = filterCls().getXmlFileName()
    
    xmlString = None
    if (xmlFileName is not None):
        rootLogger.info("Reading XML from file %s for class %s..." % (xmlFileName, str(filterCls)))
        if (checkXmlFromFile(xmlFileName, getParmSectionName(filterCls))):
            xmlString = open(xmlFileName, 'r').read()
        else:
            rootLogger.warn("XML from file %s (class %s) did not validate." % (xmlFileName, str(filterCls)))
    
    return xmlString

def getXmlString(filterCls):
    """
    Returns the string :samp:`{filterCls}.getXmlString()`.
    
    :return: :samp:`None` if :samp:`{filterCls}` does not have a :samp:`'getXmlString'` attribute.
    """
    xmlString = None
    tmpXmlString = None
    if (hasattr(filterCls, "getXmlString")):
        if (inspect.isroutine(getattr(filterCls, "getXmlString"))):
            tmpXmlString = filterCls().getXmlString()

    if (
        (tmpXmlString is not None)
        and
        (checkXmlFromString(makeXmlDoc(tmpXmlString), getParmSectionName(filterCls)))
    ):
        xmlString = tmpXmlString
    
    return xmlString


def generatePerFilterClassXmlString(filterClsList):
    """
    Returns XML string describing the filters/filter-parameters of the given
    main-driver filter classes.
    
    :type filerClsList: sequence of :obj:`mango.application.main_driver.MainDriverFilter` class objects.
    :param filerClsList: Generate (and validate) XML for this list of classes.
    :return: String of XML describing the main-driver filters in the given :samp:`filterClsList` sequence.
    """
    xmlStringSet = set();
    allFiltersXmlString = "\n"
    for filterCls in filterClsList:
        rootLogger.info("Generating/reading XML for class %s..." % str(filterCls))
        if (inspect.isclass(filterCls)):
            if (not issubclass(filterCls, MainDriverFilter)):
                rootLogger.warn("Filter class %s is not a subclass of %s. " % (str(filterCls), str(MainDriverFilter)))
                # raise RuntimeError("Encountered class which is not subclass of MainDriverFilter object in filterClsList: %s" % filterCls)
                pass
            
            xmlString = readXmlStringFromXmlFileName(filterCls)
            if (xmlString == None):
                xmlString   = getXmlString(filterCls)
                
            if ((xmlString is None)):
                rootLogger.warn("Could not determine XML filter description for class %s. " % (str(filterCls),))
            else:
                if (xmlString not in xmlStringSet):
                    allFiltersXmlString += xmlString
                    xmlStringSet.add(xmlString)
                    
            
            
        else:
            raise RuntimeError("Encountered non-class object in filterClsList: %s" % filterCls)
        
    allFiltersXmlString += "\n"
    
    return allFiltersXmlString

def generatePyMangoFilterDescriptionXml(xmlStringPart1 = None, xmlStringPart2 = None, filterClsList = None):
    """
    Generate the XML filter description string (e.g. the pymango.xml string) associated with
    the *pymango* executable. Resulting XML string suitable for *qmango* parsing.
    
    :type xmlStringPart1: :obj:`str`
    :param xmlStringPart1: Prefix XML string prepended to the individual filter XML descriptions.
    :type xmlStringPart2: :obj:`str`
    :param xmlStringPart2: Suffix XML string appended to the individual filter XML descriptions.
    :type filerClsList: sequence of :obj:`mango.application.main_driver.MainDriverFilter` class objects.
    :param filerClsList: Generate (and validate) XML for this list of classes. If :samp:`None`, generates
       the class list using the :obj:`mango.application.main_driver.getAllRegisteredMainDriverFilterClasses`
       function.

    :return: String of XML describing the main-driver filters in the given :samp:`filterClsList` sequence.
    """
    
    if (xmlStringPart1 == None):
        xmlStringPart1 = ""
        
    if (xmlStringPart2 == None):
        xmlStringPart2 = ""

    if (filterClsList is None):
        filterClsList = getAllRegisteredMainDriverFilterClasses()
    
    return xmlStringPart1 + generatePerFilterClassXmlString(filterClsList) + xmlStringPart2


def generateXmlFromCommandLineArgs(args):
    """
    Main routine used to generate the :samp:`'<install_prefix>/share/pymango.xml'` XML file
    which describes main-driver filters and data-types. XML description are generated
    for filter classes which are registered with the :obj:`mango.application.main_driver` module.
    
    The :samp:`{args}` argument is required to have the following attributes:
    
    :samp:`{args.loggingLevel}` (:obj:`str`)
        The logging level string, one of :samp:`'INFO', 'DEBUG', 'WARN', 'ERROR', 'CRITICAL', 'NONE'`.
    
    :samp:`{args.xmlPart1}` (:obj:`str`)
       XML string pre-pended to individual filter XML descriptions.

    :samp:`{args.xmlPart2}` (:obj:`str`)
       XML string appended after the individual filter XML descriptions.

    :samp:`{args.outputFileName}` (:obj:`str`)
       Name of the file in which the generated XML string is written.
    
    :type args: object
    :param args: Parameter object, see above.
    """
    logLevel=getattr(logging, args.loggingLevel)
    
    mpi.initialiseLoggers(
        [__name__, "mango.core", "mango.image", "mango.io", "mango.application"],
        logLevel=getattr(logging, args.loggingLevel)
    )

    xmlPartsFileNames = [args.xmlPart1, args.xmlPart2]
    xmlPartsStrings = [None,]*len(xmlPartsFileNames)
    for i in range(0, len(xmlPartsFileNames)):    
        xmlPartFileName= xmlPartsFileNames[i]
        if (xmlPartFileName != None):
            xmlPartsStrings[i] = open(xmlPartFileName, 'r').read()
    
    xmlString = generatePyMangoFilterDescriptionXml(xmlPartsStrings[0], xmlPartsStrings[1])
    
    if (args.outputFileName != None):
        if (mpi.rank == 0):
            rootLogger.info("Writing XML to file %s..." % args.outputFileName)
            open(args.outputFileName, 'w').write(xmlString)
            rootLogger.info("Done writing XML to file %s." % args.outputFileName)
    else:
        print(xmlString)
    
    
__all__ = [s for s in dir() if not s.startswith('_')]
