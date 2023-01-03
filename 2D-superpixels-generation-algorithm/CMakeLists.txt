cmake_minimum_required(VERSION 3.16)
project(CS271-hw2 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)

include_directories(
    src
    include
)

ADD_DEFINITIONS(-D_AFXDLL)
SET(CMAKE_MFC_FLAG 2)

# set source files and add executable
file(GLOB SRC_FILES src/*.cpp)
set(CMAKE_BUILD_TYPE "Release")
add_executable(main ${SRC_FILES} ${INTERNAL_SRC_FILES})
target_include_directories(main PRIVATE include)
target_compile_features(main PRIVATE cxx_std_14)
target_compile_definitions(main PRIVATE ${DEFINES})
target_link_libraries(main PRIVATE ${LIBRARIES})
