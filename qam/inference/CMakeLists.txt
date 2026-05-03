FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
# FetchContent_Declare(
#     eigen
#     GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#     GIT_TAG 3.4.0
# )
# FetchContent_Declare(
#     websocketpp
#     GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
#     GIT_TAG 0.8.2
# )

FetchContent_MakeAvailable(pybind11) # eigen websocketpp)

# set(Eigen3_INCLUDE_DIR ${FETCHCONTENT_BASE_DIR}/eigen-src)
# set(websocketpp_INCLUDE_DIR ${FETCHCONTENT_BASE_DIR}/websocketpp-src)

# pybind11_add_module(bolt SHARED ${CMAKE_SOURCE_DIR}/bolt/lib/binding.cpp)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/bolt/include
    # ${pybind11_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    # ${Eigen3_INCLUDE_DIR}
    # ${websocketpp_INCLUDE_DIR}
    ${BOOST_DIR}
    ${ONNXRUNTIME_DIR}/include
)

# install(TARGETS bolt LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
# set_target_properties(bolt PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})

# add_subdirectory(${FETCHCONTENT_BASE_DIR}/websocketpp-src ${FETCHCONTENT_BASE_DIR}/boost_asio-src)

add_executable(bolt ${CMAKE_CURRENT_SOURCE_DIR}/bolt/bin/main.cpp)
target_link_libraries(bolt PUBLIC ${PYTHON_LIBRARIES} pthread dl util ${ONNXRUNTIME_DIR}/libs/libonnxruntime.so ${ONNXRUNTIME_DIR}/libs/libonnxruntime_providers_shared.so)

install(TARGETS bolt RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
