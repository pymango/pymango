PROJECT(pymango)
SET(PROJECT_VERSION "r7058")

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE
  )
ENDIF()

FIND_PACKAGE(PythonInterp)
IF (PYTHONINTERP_FOUND AND CMAKE_VERSION VERSION_GREATER 2.8.7)
  FIND_PACKAGE(PythonLibs ${PYTHON_VERSION_STRING})
ELSE()
  FIND_PACKAGE(PythonLibs)
ENDIF()

IF (PYTHON_EXECUTABLE)
  SET(
    PYTHON_SITE_PACKAGES_INSTALL_DIR
    ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages
    CACHE INTERNAL ""
  )
  SET(
    MANGO_PYTHON_SITE_PACKAGES_INSTALL_DIR
    ${PYTHON_SITE_PACKAGES_INSTALL_DIR}/mango
    CACHE INTERNAL ""
  )
ENDIF()

ADD_SUBDIRECTORY(misc)
ADD_SUBDIRECTORY(registration)
ADD_SUBDIRECTORY(doc)

