cmake_minimum_required(VERSION 3.2)
project(csc4005_imgui)

set(CMAKE_CXX_STANDARD 17)

find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(OpenGL_GL_PREFERENCE "GLVND")
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
add_executable(mpi ${CSC4005_PROJECT_SRC})
add_executable(pthread ${CSC4005_PROJECT_SRC})
get_filename_component(FONT_PATH imgui/misc/fonts/DroidSans.ttf ABSOLUTE)
target_link_libraries(core PUBLIC
        Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})
target_link_libraries(mpi core)
target_compile_options(mpi PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)
target_compile_definitions(mpi PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(mpi PRIVATE -DMPI)
target_link_libraries(pthread core)
target_compile_options(pthread PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_compile_definitions(pthread PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(pthread PRIVATE -DPTHREAD)
if (WIN32)
    target_link_options(mpi BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
endif()