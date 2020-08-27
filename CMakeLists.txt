cmake_minimum_required(VERSION 3.10)
project(Piz LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(HEADERS 
    include/camera.h
    include/draw.h
    include/framebuffer.h
    include/image.h
    include/light.h
    include/loader.h
    include/macros.h
    include/material.h
    include/maths.h
    include/matrix.h
    include/object.h
    include/pathtracer.h
    include/plane.h
    include/platform.h
    include/polygon.h
    include/program.h
    include/scene.h
    include/vector.h
)

set(SOURCES 
    src/camera.cu
    src/draw.cu
    src/framebuffer.cu
    src/image.cu
    src/material.cu
    src/maths.cu
    src/matrix.cu
    src/object.cu
    src/pathtracer.cu
    src/plane.cu
    src/scene.cu
    src/vector.cu
)

if(WIN32)
    set(SOURCES ${SOURCES} platform/win32.cpp)
endif()

# ==========================
# DEMO
# ==========================

add_library(common INTERFACE)

# set(TARGET Demo_Interactive Demo_PathTracer)

add_executable(Demo_Interactive ${HEADERS} ${SOURCES} src/demo.cu)
add_executable(Demo_PathTracer ${HEADERS} ${SOURCES} src/demo_pathtracer.cu)

target_include_directories(Demo_Interactive PUBLIC include)
target_include_directories(Demo_PathTracer PUBLIC include)

# ==========================
# Compile options
# ==========================

find_package(OpenMP REQUIRED)

if(MSVC)
    target_compile_options(common INTERFACE /fp:fast)
    target_compile_options(common INTERFACE /openmp)
endif()

# ==========================
# Link libraries
# ==========================

target_link_libraries(Demo_Interactive common)
target_link_libraries(Demo_PathTracer common)
# target_compile_options(example PUBLIC -std=c++1y -Wall -Wfloat-conversion)
