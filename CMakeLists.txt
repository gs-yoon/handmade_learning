cmake_minimum_required(VERSION 2.8)
project( handmade_learning )
#find_package( OpenCV 4.0 REQUIRED )

#include_directories( ${OpenCV_INCLUDE_DIRS} )
add_library(activate_function STATIC
${CMAKE_SOURCE_DIR}/src/includes/essential/activate_functions.cpp)

include_directories( ${CMAKE_SOURCE_DIR}/dependencies/eigen-3.4.0
                        ${CMAKE_SOURCE_DIR}/src/includes/essential
                        ${CMAKE_SOURCE_DIR}/src/includes/definition )


file(GLOB SOURCES
    "src/*.h"
    "src/*.cpp"
)

add_executable( handmade_learning ${SOURCES} )

target_link_libraries( handmade_learning 
    PRIVATE
    activate_function)
    
#target_link_libraries( handmade_learning ${OpenCV_LIBS} )

