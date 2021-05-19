cmake_minimum_required(VERSION 3.8)
project(lab5_Grandin_Matteo_2020374)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} 
lab5.cpp
PanoramicImage.h
PanoramicImage.cpp
panoramic_utils.h
)

file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})