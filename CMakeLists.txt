cmake_minimum_required(VERSION 2.8)
project( handmade_learning )
find_package( OpenCV 4.0 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories( ${CMAKE_SOURCE_DIR}/dependencies/eigen-3.4.0
                        ${CMAKE_SOURCE_DIR}/src/essential
                        ${CMAKE_SOURCE_DIR}/src/includes/definition )


file(GLOB SOURCES
    "src/*.cpp"
    "src/essential/*.cpp"
)

add_executable( handmade_learning ${SOURCES} )

target_link_libraries( handmade_learning ${OpenCV_LIBS} )

