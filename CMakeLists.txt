cmake_minimum_required(VERSION 3.5)

project(QAM)

set(CMAKE_CXX_STANDARD 23)
set(BUILD_SHARED_LIBS ON)
set(BUILD_TESTING OFF)

if(NOT DEFINED PYTHON_EXECUTABLE OR NOT DEFINED CMAKE_INSTALL_PREFIX OR NOT DEFINED PYTHON_INCLUDE_DIR)
    message(FATAL_ERROR "Please specify PYTHON_EXECUTABLE and CMAKE_INSTALL_PREFIX and PYTHON_INCLUDE_DIR")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

find_package(Boost REQUIRED)
find_library(PTHREAD NAMES pthread REQUIRED)
# find_library(DL NAMES dl REQUIRED)
# find_library(UTIL NAMES util REQUIRED)
find_library(Z_LIB NAMES z)
find_library(BZ2_LIB NAMES bz2)
find_library(LZMA_LIB NAMES lzma)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_Declare(
    kalman
    GIT_REPOSITORY https://github.com/mherb/kalman.git
)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_Declare(
    websocketpp
    GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
    GIT_TAG 0.8.2
)
FetchContent_Declare(
    boost_asio
    GIT_REPOSITORY https://github.com/boostorg/asio.git
    GIT_TAG boost-1.86.0
)

FetchContent_MakeAvailable(pybind11 kalman eigen websocketpp boost_asio)

file(GLOB FST_SOURCES ${FST_DIR}/src/lib/*.cc ${FST_DIR}/src/script/*.cc)

set(Eigen3_INCLUDE_DIR ${FETCHCONTENT_BASE_DIR}/eigen-src)
set(kalman_INCLUDE_DIR ${FETCHCONTENT_BASE_DIR}/kalman-src/include)
set(websocketpp_INCLUDE_DIR ${FETCHCONTENT_BASE_DIR}/websocketpp-src)
pybind11_add_module(_zctc SHARED ${CMAKE_SOURCE_DIR}/src/binding.cpp ${FST_SOURCES})
include_directories(${CMAKE_SOURCE_DIR}/include ${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${FST_DIR}/src/include ${Eigen3_INCLUDE_DIR} ${kalman_INCLUDE_DIR} ${websocketpp_INCLUDE_DIR}) #${FETCHCONTENT_BASE_DIR}/kenlm_external-src ${FETCHCONTENT_BASE_DIR}/threadpool_external-src)
message("Include directories are ${Eigen3_INCLUDE_DIR} ${kalman_INCLUDE_DIR} ${websocketpp_INCLUDE_DIR}")

target_link_libraries(_zctc PUBLIC ${PYTHON_LIBRARIES} kenlm_filter kenlm_builder kenlm_util kenlm pthread dl util)
if(TARGET z)
    target_link_libraries(_zctc PUBLIC z)
endif()
if(TARGET bz2)
    target_link_libraries(_zctc PUBLIC bz2)
endif()
if(TARGET lzma)
    target_link_libraries(_zctc PUBLIC lzma)
endif()

install(TARGETS _zctc LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
# set_target_properties(_zctc kenlm_filter kenlm_builder kenlm_util kenlm PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})

# add_subdirectory(${FETCHCONTENT_BASE_DIR}/websocketpp-src ${FETCHCONTENT_BASE_DIR}/boost_asio-src)

add_executable(bolt ${CMAKE_SOURCE_DIR}/src/demo.cpp)

target_include_directories(bolt PUBLIC ${websocketpp_INCLUDE_DIR} ${boost_asio_INCLUDE_DIR})

install(TARGETS bolt RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
