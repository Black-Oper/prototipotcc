#include "RTDVSRInferencer.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

RTDVSRInferencer::RTDVSRInferencer(const std::string& enginePath, int widthLR, int heightLR, int scaleFactor, int hiddenDim)
    : m_widthLR(widthLR),
      m_heightLR(heightLR),
      m_scaleFactor(scaleFactor),
      m_hiddenDim(hiddenDim) {
    
    m_widthSR = m_widthLR * m_scaleFactor;
    m_heightSR = m_heightLR * m_scaleFactor;

    m_inputSizeBytes = 1 * 3 * m_heightLR * m_widthLR * sizeof(float);
    m_outputSizeBytes = 1 * 3 * m_heightSR * m_widthSR * sizeof(float);
    m_stateSizeBytes = 1 * m_hiddenDim * m_heightLR * m_widthLR * sizeof(float);

    cudaStreamCreate(&m_stream);
    loadEngine(enginePath);
    allocateBuffers();
}

RTDVSRInferencer::~RTDVSRInferencer() {
    freeBuffers();
    if (m_stream) cudaStreamDestroy(m_stream);
    if (m_context) delete m_context;
    if (m_engine) delete m_engine;
    if (m_runtime) delete m_runtime;
}

void RTDVSRInferencer::loadEngine(const std::string& enginePath) {
    // ----------------------------------------------------------------------
    // INICIALIZAÇÃO DOS PLUGINS (Resolve o erro do ModulatedDeformConv2d)
    // ----------------------------------------------------------------------
    initLibNvInferPlugins(&gLogger, "");

    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Não foi possível abrir o arquivo do motor TensorRT: " + enginePath);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    m_runtime = nvinfer1::createInferRuntime(gLogger);
    if (!m_runtime) {
        throw std::runtime_error("Falha ao criar o IRuntime do TensorRT.");
    }

    m_engine = m_runtime->deserializeCudaEngine(buffer.data(), size);
    if (!m_engine) {
        throw std::runtime_error("Falha ao desserializar o motor CUDA.");
    }

    m_context = m_engine->createExecutionContext();
    if (!m_context) {
        throw std::runtime_error("Falha ao criar o ExecutionContext do TensorRT.");
    }
}

void RTDVSRInferencer::allocateBuffers() {
    cudaMalloc(&d_inputLR, m_inputSizeBytes);
    cudaMalloc(&d_outputSR, m_outputSizeBytes);
    cudaMalloc(&d_stateA, m_stateSizeBytes);
    cudaMalloc(&d_stateB, m_stateSizeBytes);

    resetState();
}

void RTDVSRInferencer::freeBuffers() {
    if (d_inputLR) cudaFree(d_inputLR);
    if (d_outputSR) cudaFree(d_outputSR);
    if (d_stateA) cudaFree(d_stateA);
    if (d_stateB) cudaFree(d_stateB);
}

void RTDVSRInferencer::resetState() {
    if (d_stateA) cudaMemsetAsync(d_stateA, 0, m_stateSizeBytes, m_stream);
    if (d_stateB) cudaMemsetAsync(d_stateB, 0, m_stateSizeBytes, m_stream);
    cudaStreamSynchronize(m_stream);
    m_pingPong = false;
}

bool RTDVSRInferencer::processFrame(const cv::Mat& inputLR, cv::Mat& outputSR) {
    if (inputLR.empty() || inputLR.cols != m_widthLR || inputLR.rows != m_heightLR) {
        std::cerr << "Dimensões do frame de entrada inválidas." << std::endl;
        return false;
    }

    if (outputSR.empty() || outputSR.cols != m_widthSR || outputSR.rows != m_heightSR) {
        outputSR = cv::Mat(m_heightSR, m_widthSR, CV_8UC3);
    }

    // Preprocessamento HWC (BGR uint8) -> CHW (RGB float32 [0, 1]) na CPU
    cv::Mat rgbMat;
    cv::cvtColor(inputLR, rgbMat, cv::COLOR_BGR2RGB);

    std::vector<float> hostInput(3 * m_heightLR * m_widthLR);
    int planeSize = m_heightLR * m_widthLR;

    for (int h = 0; h < m_heightLR; ++h) {
        for (int w = 0; w < m_widthLR; ++w) {
            cv::Vec3b pixel = rgbMat.at<cv::Vec3b>(h, w);
            int idx = h * m_widthLR + w;
            hostInput[0 * planeSize + idx] = pixel[0] / 255.0f; // R
            hostInput[1 * planeSize + idx] = pixel[1] / 255.0f; // G
            hostInput[2 * planeSize + idx] = pixel[2] / 255.0f; // B
        }
    }

    // Copia entrada para a VRAM
    cudaMemcpyAsync(d_inputLR, hostInput.data(), m_inputSizeBytes, cudaMemcpyHostToDevice, m_stream);

    // Alternância do Ping-Pong buffer de estado
    float* currentStateInput  = m_pingPong ? d_stateB : d_stateA;
    float* currentStateOutput = m_pingPong ? d_stateA : d_stateB;

    // Associa os ponteiros de memória de VRAM aos nomes dos tensores no TensorRT 10.x
    m_context->setTensorAddress("x", d_inputLR);
    m_context->setTensorAddress("prev_state", currentStateInput);
    m_context->setTensorAddress("sr", d_outputSR);
    m_context->setTensorAddress("state", currentStateOutput);

    // Executa a inferência assíncrona na GPU
    bool status = m_context->enqueueV3(m_stream);
    if (!status) {
        return false;
    }

    // Copia resultado de volta para a CPU
    std::vector<float> hostOutput(3 * m_heightSR * m_widthSR);
    cudaMemcpyAsync(hostOutput.data(), d_outputSR, m_outputSizeBytes, cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // Pós-processamento CHW (RGB float32) -> HWC (BGR uint8)
    int srPlaneSize = m_heightSR * m_widthSR;
    for (int h = 0; h < m_heightSR; ++h) {
        for (int w = 0; w < m_widthSR; ++w) {
            int idx = h * m_widthSR + w;
            float r = hostOutput[0 * srPlaneSize + idx] * 255.0f;
            float g = hostOutput[1 * srPlaneSize + idx] * 255.0f;
            float b = hostOutput[2 * srPlaneSize + idx] * 255.0f;

            outputSR.at<cv::Vec3b>(h, w) = cv::Vec3b(
                static_cast<uchar>(std::clamp(b, 0.0f, 255.0f)),
                static_cast<uchar>(std::clamp(g, 0.0f, 255.0f)),
                static_cast<uchar>(std::clamp(r, 0.0f, 255.0f))
            );
        }
    }

    // Inverte o estado do Ping-Pong para o próximo frame
    m_pingPong = !m_pingPong;

    return true;
}