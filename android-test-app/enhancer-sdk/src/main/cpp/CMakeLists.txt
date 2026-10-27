cmake_minimum_required(VERSION 3.22.1)
project(ai_filter_enhancer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(APP_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/../../../../app/src/main")
set(ONNXRUNTIME_ROOT "${APP_MAIN}/cpp/onnxruntime")
set(ONNXRUNTIME_LIB "${APP_MAIN}/jniLibs/${ANDROID_ABI}/libonnxruntime.so")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers missing. Run ./scripts/setup_native_deps.sh")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIB}")
    message(FATAL_ERROR "libonnxruntime.so missing. Run ./scripts/setup_native_deps.sh")
endif()

find_package(OpenCV REQUIRED CONFIG)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIB}")

add_library(ai_filter_enhancer SHARED
    ai_filter_enhancer.cpp
    jni_bridge.cpp
)

target_include_directories(ai_filter_enhancer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(ai_filter_enhancer
    onnxruntime
    OpenCV::opencv_java4
    android
    jnigraphics
    log
)
