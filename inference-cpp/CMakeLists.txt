# Mensagem: MODIFICAR PARA O PRÓPRIO PC (depois envio uma versão genérica)

cmake_minimum_required(VERSION 3.20)
project(RTDVSR_CPP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(/EHsc)

find_package(CUDAToolkit REQUIRED)

set(OpenCV_DIR "C:/MY-SPACE/OpenCV/opencv/build")
find_package(OpenCV REQUIRED)

set(TENSORRT_ROOT "C:/MY-SPACE/TensorRT-10.16.1.11" CACHE PATH "Raiz do TensorRT C++")
include_directories(${TENSORRT_ROOT}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

link_directories(${TENSORRT_ROOT}/lib)

set(SOURCES
    src/RTDVSRInferencer.cpp
    src/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer_10
    nvinfer_plugin_10
)