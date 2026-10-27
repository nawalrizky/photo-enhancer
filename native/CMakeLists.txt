cmake_minimum_required(VERSION 3.18.1)
project(photo_enhancer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define paths for ONNX Runtime Mobile and OpenCV Android SDK
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "Path to ONNX Runtime Android SDK")
set(OPENCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv" CACHE PATH "Path to OpenCV Android SDK")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_DIR}/include
    ${OPENCV_DIR}/sdk/native/jni/include
)

# Source files
set(SOURCE_FILES
    src/native_enhancer.cpp
    src/jni_bridge.cpp
)

# Create shared library (.so)
add_library(photo_enhancer SHARED ${SOURCE_FILES})

# Import ONNX Runtime prebuilt shared library (.so)
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION
    ${ONNXRUNTIME_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so
)

# Import OpenCV prebuilt static/shared libraries
find_package(OpenCV REQUIRED HINTS ${OPENCV_DIR}/sdk/native/jni)

# Link libraries
target_link_libraries(photo_enhancer
    onnxruntime
    ${OpenCV_LIBS}
    android
    jnigraphics
    log
)
