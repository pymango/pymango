
IF (PYTHON_SITE_PACKAGES_INSTALL_DIR)
  #
  # Installation order should not be important here, never
  # want to over-write any of the open python modules
  # with restricted python modules, can't guarantee
  # order of install...
  #
  INSTALL(
    DIRECTORY mango
    DESTINATION ${PYTHON_SITE_PACKAGES_INSTALL_DIR}
    COMPONENT python
  )
ENDIF()
