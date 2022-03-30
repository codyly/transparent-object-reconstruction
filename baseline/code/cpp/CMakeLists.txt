cmake_minimum_required(VERSION 2.8)
project( ArTagPart )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( ArTagPart test.cpp )
target_link_libraries( ArTagPart ${OpenCV_LIBS} )
