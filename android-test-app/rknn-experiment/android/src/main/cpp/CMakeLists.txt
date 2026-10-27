cmake_minimum_required(VERSION 3.22.1)
project(experimental_rknn_enhancer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED CONFIG)
set(EXPERIMENT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES IMPORTED_LOCATION
    "${EXPERIMENT_ROOT}/vendor/librknnrt.so")
add_library(experimental_rknn_enhancer SHARED ai_filter_enhancer.cpp jni_bridge.cpp)
target_include_directories(experimental_rknn_enhancer PRIVATE
    "${EXPERIMENT_ROOT}/vendor")
target_link_libraries(experimental_rknn_enhancer rknnrt OpenCV::opencv_java4 android jnigraphics log)
