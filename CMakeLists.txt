cmake_minimum_required(VERSION 3.10)
project(Piz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

set(HEADERS 
    include/camera.h
    include/draw.h
    include/framebuffer.h
    include/image.h
    include/light.h
    include/loader.h
    include/macros.h
    include/maths.h
    include/matrix.h
    include/object.h
    include/pathtracer.h
    include/plane.h
    include/platform.h
    include/polygon.h
    include/program.h
    include/vector.h
)

set(SOURCES 
    src/camera.cpp
    src/draw.cpp
    src/framebuffer.cpp
    src/image.cpp
    src/maths.cpp
    src/matrix.cpp
    src/object.cpp
    src/pathtracer.cpp
    src/plane.cpp
    src/vector.cpp
)

if(WIN32)
    set(SOURCES ${SOURCES} platform/win32.cpp)
endif()

# ==========================
# DEMO
# ==========================

add_library(common INTERFACE)

# set(TARGET Demo_Interactive Demo_PathTracer)

add_executable(Demo_Interactive ${HEADERS} ${SOURCES} src/demo.cpp)
add_executable(Demo_PathTracer ${HEADERS} ${SOURCES} src/demo_pathtracer.cpp)

target_include_directories(Demo_Interactive PUBLIC include)
target_include_directories(Demo_PathTracer PUBLIC include)

# ==========================
# Compile options
# ==========================

if(MSVC)
    target_compile_options(common INTERFACE /fp:fast)
endif()

# ==========================
# Link libraries
# ==========================

target_link_libraries(Demo_Interactive common)
target_link_libraries(Demo_PathTracer common)
# target_compile_options(example PUBLIC -std=c++1y -Wall -Wfloat-conversion)
