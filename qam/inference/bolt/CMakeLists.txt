cmake_minimum_required(VERSION 3.5)

project(BOLT)

set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -fPIC -DKENLM_MAX_ORDER=6")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

include(FetchContent)

FetchContent_Declare(
    websocketpp
    GIT_REPOSITORY https://github.com/zaphoyd/websocketpp
    GIT_TAG 0.8.2
)
FetchContent_Declare(
    boost_asio
    GIT_REPOSITORY https://github.com/boostorg/asio
    GIT_TAG boost-1.85.0
)
FetchContent_MakeAvailable(websocketpp boost_asio)

add_subdirectory(${FETCHCONTENT_BASE_DIR}/websocketpp-src ${FETCHCONTENT_BASE_DIR}/boost_asio-src)

add_executable(bolt ${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp)

target_include_directories (bolt PUBLIC ${FETCHCONTENT_BASE_DIR}/websocketpp-src ${FETCHCONTENT_BASE_DIR}/boost_asio-src/include)

install(TARGETS bolt RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
