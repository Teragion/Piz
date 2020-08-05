cmake_minimum_required(VERSION 3.10)
project(Piz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

set(HEADERS 
    include/camera.h
    include/framebuffer.h
    include/loader.h
    include/macros.h
    include/maths.h
    include/matrix.h
    include/object.h
    include/plane.h
    include/platform.h
    include/polygon.h
    include/vector.h
)

set(SOURCES 
    src/camera.cpp
    src/demo.cpp
    src/matrix.cpp
    src/plane.cpp
    src/vector.cpp
)

if(WIN32)
    set(SOURCES ${SOURCES} platform/win32.cpp)
endif()

# ==========================
# DEMO
# ==========================

set(TARGET Demo)

add_executable(${TARGET} ${HEADERS} ${SOURCES})

target_include_directories(Demo PUBLIC include)

# ==========================
# Compile options
# ==========================

# ==========================
# Link libraries
# ==========================

# target_compile_options(example PUBLIC -std=c++1y -Wall -Wfloat-conversion)
