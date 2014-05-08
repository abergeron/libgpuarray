# - Try to find sqlite3 
# Find sqlite3 headers, libraries and the answer to all questions.
#
#  SQLITE3_FOUND               True if sqlite3 got found
#  SQLITE3_INCLUDE_DIR         Location of sqlite3 headers 
#  SQLITE3_LIBRARIES           List of libaries to use sqlite3

# This file is public domain, use as you like

find_package(PkgConfig)
pkg_check_modules(PC_SQLITE3 QUIET sqlite3)

find_path(SQLITE3_INCLUDE_DIR sqlite3.h
          HINTS ${PC_SQLITE3_INCLUDEDIR} ${PC_SQLITE3_INCLUDE_DIRS})

find_library(SQLITE3_LIBRARY sqlite3 libsqlite3
             HINTS ${PC_SQLITE3_INCLUDEDIR} ${PC_SQLITE3_LIBRARY_DIRS})

set(SQLITE3_INCLUDE_DIRS ${SQLITE3_INCLUDE_DIR})
set(SQLITE3_LIBRARIES ${SQLITE3_LIBRARY})

find_package_handle_standard_args(Sqlite3 DEFAULT_MSG
                                  SQLITE3_LIBRARY SQLITE3_INCLUDE_DIR)

MARK_AS_ADVANCED( SQLITE3_LIBRARY SQLITE3_INCLUDE_DIR )

