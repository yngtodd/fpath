## Build the fpath application and its test suite.
##
cmake_minimum_required(VERSION 3.12)

project(fpath
    VERSION 0.0.1.0
    LANGUAGES CXX CUDA
)

#set(Torch_DIR "{CMAKE_SOURCE_DIR}/lib/libtorch/share/cmake/Torch")
#set(CMAKE_PREFIX_PATH ${Torch_DIR}) 

find_package(Torch REQUIRED)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(EXECUTABLE fpath) 

set(API_SOURCE_FILES
    # API sources
)

set(CORE_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/core/configure.cpp    
    ${CMAKE_SOURCE_DIR}/src/core/logging.cpp
)

set(DATA_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/data/random.cpp
)

set(MODEL_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/models/yoonkim.cpp
)

set(APP_SOURCE_FILES
    ${API_SOURCE_FILES}
    ${CORE_SOURCE_FILES}
    ${DATA_SOURCE_FILES}
    ${MODEL_SOURCE_FILES}
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)

add_executable(${EXECUTABLE})

target_sources(${EXECUTABLE}
PRIVATE
    ${API_SOURCE_FILES}
    ${CORE_SOURCE_FILES}
    ${DATA_SOURCE_FILES}
    ${MODEL_SOURCE_FILES}
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)

target_include_directories(${EXECUTABLE}
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${EXECUTABLE} cudnn ${TORCH_LIBRARIES})

target_compile_features(${EXECUTABLE}
PRIVATE
    cxx_std_14
)

add_subdirectory(test)
