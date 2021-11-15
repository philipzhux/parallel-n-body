cmake_minimum_required(VERSION 3.2)
project(csc4005_imgui CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 20)

find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA 9.0 REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(OpenGL_GL_PREFERENCE "GLVND")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/opt/rh/devtoolset-10/root/usr/bin/g++ -gencode=arch=compute_75,code=[sm_75,compute_75]")
find_package(OpenGL REQUIRED)

include_directories(
        include
        imgui
        imgui/backends
        ${SDL2_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS})

file(GLOB IMGUI_SRC
        imgui/*.cpp
        imgui/backends/imgui_impl_sdl.cpp
        imgui/backends/imgui_impl_opengl2.cpp
        imgui/misc/freetype/imgui_freetype.cpp
        imgui/misc/cpp/imgui_stdlib.cpp
        )
add_library(core STATIC ${IMGUI_SRC})
file(GLOB CSC4005_PROJECT_SRC src/*.cpp src/*.c)
file(GLOB CUDA_SRC src/cuda/*.cpp src/cuda/*.c src/cuda/*.cu)
add_executable(nb_cuda ${CUDA_SRC})
set_target_properties(
        nb_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
add_executable(nb_mpi ${CSC4005_PROJECT_SRC})
add_executable(nb_pthread ${CSC4005_PROJECT_SRC})
add_executable(nb_omp ${CSC4005_PROJECT_SRC})
get_filename_component(FONT_PATH imgui/misc/fonts/DroidSans.ttf ABSOLUTE)
target_link_libraries(core PUBLIC
        Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})
target_link_libraries(nb_mpi core)
target_compile_options(nb_mpi PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)
target_compile_definitions(nb_mpi PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(nb_mpi PRIVATE -DMPI)
target_link_libraries(nb_pthread core)
target_compile_options(nb_pthread PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_compile_definitions(nb_pthread PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(nb_pthread PRIVATE -DPTHREAD)
target_link_libraries(nb_omp core)
target_compile_options(nb_omp PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_compile_definitions(nb_omp PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(nb_omp PRIVATE -DOPENMP)
target_link_libraries(nb_cuda core)
target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)
target_compile_definitions(nb_cuda PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_link_libraries(nb_cuda ${CUDA_LIBRARIES})
target_compile_options(nb_omp PRIVATE
          $<$<COMPILE_LANGUAGE:C>:"${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}">
          $<$<COMPILE_LANGUAGE:CXX>:"${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}">)
target_link_options(nb_omp PRIVATE "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
if (WIN32)
    target_link_options(nb_cuda BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
endif()
if (WIN32)
    target_link_options(mpi BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
endif()